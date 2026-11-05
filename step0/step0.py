# Software4 step 0: feasibility check on a Kaggle T4.
# Tests: 1 load + fp16 sanity, 2 baseline score, 3 one adapter, 4 two adapters, 5 sparse memory layer.
# Writes /kaggle/working/results.json after every test.
import json, os, random, re, subprocess, sys, time, traceback

OUT = "/kaggle/working/results.json" if os.path.isdir("/kaggle/working") else "results.json"
R = {"started": time.strftime("%Y-%m-%d %H:%M:%S")}

def save():
    with open(OUT, "w") as f:
        json.dump(R, f, indent=2)

def log(*a):
    print(*a, flush=True)

def sh(cmd):
    log("$", cmd)
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    log((p.stdout + p.stderr)[-1500:])
    return p.returncode

MID = os.environ.get("S4_MODEL", "Qwen/Qwen3.5-2B")
R["model"] = MID

# ---------- setup ----------
sh("pip install -q -U 'transformers>=5.0' peft accelerate")
sh("pip uninstall -y torchao")  # Kaggle's old torchao breaks peft's LoRA dispatch
fla_rc = sh("pip install -q -U flash-linear-attention")
SMOKE = r'''
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
try:
    m = AutoModelForCausalLM.from_pretrained("%s", dtype=torch.float16, device_map={"": 0})
except Exception:
    from transformers import AutoModelForImageTextToText
    m = AutoModelForImageTextToText.from_pretrained("%s", dtype=torch.float16, device_map={"": 0})
t = AutoTokenizer.from_pretrained("%s")
x = t("Hello, my name is", return_tensors="pt").to(0)
y = m.generate(**x, max_new_tokens=8, do_sample=False)
assert torch.isfinite(m(**x).logits).all()
print("SMOKE_OK", t.decode(y[0]))
''' % (MID, MID, MID)
with open("smoke.py", "w") as f:
    f.write(SMOKE)
fast = False
if fla_rc == 0:
    p = subprocess.run([sys.executable, "smoke.py"], capture_output=True, text=True)
    fast = "SMOKE_OK" in p.stdout
    log("smoke with fla:", fast, (p.stdout + p.stderr)[-1500:])
    if not fast:
        sh("pip uninstall -y flash-linear-attention fla-core")
R["setup"] = {"fla_installed": fla_rc == 0, "fast_path_smoke_ok": fast}

import torch
import torch.nn as nn
import transformers, peft
from transformers import AutoTokenizer, AutoModelForCausalLM

R["setup"].update({
    "torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__,
    "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
})
save()
log(R["setup"])

tok = AutoTokenizer.from_pretrained(MID)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
IM_END = "<|im_end|>"

def chat(user):
    msgs = [{"role": "user", "content": user}]
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except Exception:
        return f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

def load(dtype, dev):
    try:
        return AutoModelForCausalLM.from_pretrained(MID, dtype=dtype, device_map={"": dev})
    except Exception as e:
        log("CausalLM load failed, trying ImageTextToText:", repr(e)[:300])
        from transformers import AutoModelForImageTextToText
        return AutoModelForImageTextToText.from_pretrained(MID, dtype=dtype, device_map={"": dev})

@torch.no_grad()
def generate(model, prompts, max_new=48, bs=16):
    tok.padding_side = "left"
    outs = []
    for i in range(0, len(prompts), bs):
        enc = tok([chat(p) for p in prompts[i:i + bs]], return_tensors="pt", padding=True).to(model.device)
        y = model.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id)
        outs += tok.batch_decode(y[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    return [o.strip() for o in outs]

# ---------- toy skills (the model cannot know these rules) ----------
WORDS = ("apple river stone cloud tiger music garden window pencil rocket candle forest ocean silver dragon "
         "planet bridge castle mirror shadow thunder valley meadow lantern harbor island jungle desert winter "
         "summer autumn spring copper marble velvet coffee butter honey pepper ginger lemon mango cherry orange "
         "violet purple yellow basket bottle button carpet cookie engine falcon guitar hammer helmet jacket "
         "kettle ladder magnet needle oyster parrot pillow puzzle rabbit saddle tablet tunnel wallet zebra "
         "anchor badger camera dentist eagle feather goblin hockey igloo jigsaw koala lizard monkey noodle "
         "pirate quilt raven salmon turtle umbrella vulture walrus yogurt acorn barrel cactus donkey elbow "
         "fossil glacier hermit insect jester kitten lobster mammoth nectar onion pebble quartz robot spider "
         "trumpet unicorn vessel wizard").split()

def skill_a(rng):
    w = rng.sample(WORDS, rng.randint(3, 5))
    return "glossify: " + ", ".join(w), " ~ ".join(x.capitalize() for x in sorted(w)) + f" [{len(w)}]"

def skill_b(rng):
    n = [rng.randint(1, 99) for _ in range(rng.randint(3, 5))]
    return "numbify: " + ", ".join(map(str, n)), f"sum={sum(n)} max={max(n)}"

def make(fn, n, seed):
    rng = random.Random(seed)
    return [fn(rng) for _ in range(n)]

A_TR, A_TE = make(skill_a, 600, 1), make(skill_a, 100, 2)
B_TR, B_TE = make(skill_b, 600, 3), make(skill_b, 100, 4)

def skill_acc(model, data):
    preds = generate(model, [q for q, _ in data], max_new=40)
    return round(sum(p == a for p, (_, a) in zip(preds, data)) / len(data), 3), preds[:3]

# ---------- GSM8K slice ----------
def gsm8k(n=100):
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="test").select(range(n))
    return [(r["question"], r["answer"].split("####")[-1].strip().replace(",", "")) for r in ds]

def num(s):
    m = re.findall(r"-?\d[\d,]*\.?\d*", s)
    return m[-1].replace(",", "").rstrip(".") if m else None

def gsm_acc(model, data):
    prompts = [q + "\nThink step by step briefly, then end with: The answer is N." for q, _ in data]
    preds = generate(model, prompts, max_new=384)
    ok = 0
    for p, (_, a) in zip(preds, data):
        tail = p.split("answer is")[-1] if "answer is" in p else p
        g = num(tail)
        try:
            ok += float(g) == float(a)
        except (TypeError, ValueError):
            pass
    return round(ok / len(data), 3)

def run(name, fn):
    t = time.time()
    try:
        R[name] = fn()
    except Exception as e:
        R[name] = {"error": repr(e), "trace": traceback.format_exc()[-2500:]}
    R[name] = R[name] if isinstance(R[name], dict) else {"value": R[name]}
    R[name]["seconds"] = round(time.time() - t)
    save()
    log(name, json.dumps(R[name])[:1500])

# ---------- test 1: load + fp16 sanity + speed ----------
model = None
def t1():
    global model
    model = load(torch.float16, 0)
    model.eval()
    res = {"params_b": round(sum(p.numel() for p in model.parameters()) / 1e9, 3), "class": type(model).__name__}
    text = " ".join(open(transformers.__file__.replace("__init__.py", "modeling_utils.py")).read().split()[:3000])
    enc = tok([text[:12000]], return_tensors="pt", truncation=True, max_length=1024).to(0)
    with torch.no_grad():
        out = model(**enc, output_hidden_states=True)
    res["logits_finite_1k"] = bool(torch.isfinite(out.logits).all())
    res["max_abs_hidden"] = round(max(h.abs().max().item() for h in out.hidden_states), 1)
    # fp16 vs fp32 agreement on the second GPU
    if torch.cuda.device_count() > 1:
        m32 = load(torch.float32, 1)
        with torch.no_grad():
            l32 = m32(**{k: v.to(1) for k, v in enc.items()}).logits.float().cpu()
        l16 = out.logits.float().cpu()
        res["fp16_vs_fp32_top1_agree"] = round((l16.argmax(-1) == l32.argmax(-1)).float().mean().item(), 4)
        res["fp16_vs_fp32_max_logit_diff"] = round((l16 - l32).abs().max().item(), 3)
        del m32
        torch.cuda.empty_cache()
    # speed: batch 1, 256 new tokens
    x = tok(chat("Write a short story about a robot."), return_tensors="pt").to(0)
    torch.cuda.synchronize(); t = time.time()
    with torch.no_grad():
        y = model.generate(**x, max_new_tokens=256, min_new_tokens=256, do_sample=False)
    torch.cuda.synchronize()
    res["decode_tok_per_s_bs1"] = round((y.shape[1] - x["input_ids"].shape[1]) / (time.time() - t), 1)
    res["sample"] = generate(model, ["What is the capital of France? One word."], max_new=10)[0]
    res["peak_mem_gb"] = round(torch.cuda.max_memory_allocated(0) / 1e9, 2)
    return res
run("t1_load", t1)
if model is None:
    log("model failed to load; stopping"); sys.exit(0)

GSM = gsm8k(100)

# ---------- test 2: baseline ----------
run("t2_baseline", lambda: {
    "gsm8k_100": gsm_acc(model, GSM),
    "skill_a": skill_acc(model, A_TE),
    "skill_b": skill_acc(model, B_TE),
})

# ---------- training helper ----------
def batches(data, bs, rng):
    tok.padding_side = "right"
    idx = list(range(len(data))); rng.shuffle(idx)
    for i in range(0, len(idx), bs):
        ids, labs = [], []
        for j in idx[i:i + bs]:
            q, a = data[j]
            p = tok(chat(q), add_special_tokens=False)["input_ids"]
            full = tok(chat(q) + a + IM_END, add_special_tokens=False)["input_ids"]
            ids.append(full); labs.append([-100] * len(p) + full[len(p):])
        L = max(map(len, ids))
        att = [[1] * len(x) + [0] * (L - len(x)) for x in ids]
        ids = [x + [tok.pad_token_id] * (L - len(x)) for x in ids]
        labs = [x + [-100] * (L - len(x)) for x in labs]
        yield (torch.tensor(ids).cuda(), torch.tensor(att).cuda(), torch.tensor(labs).cuda())

def train(params, data, epochs=2, lr=2e-4, bs=16):
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda")
    rng = random.Random(0); losses = []
    model.train()
    for ep in range(epochs):
        for ids, att, labs in batches(data, bs, rng):
            with torch.autocast("cuda", dtype=torch.float16):
                loss = model(input_ids=ids, attention_mask=att, labels=labs).loss
            if not torch.isfinite(loss):
                losses.append(float("nan")); opt.zero_grad(); continue
            scaler.scale(loss).backward()
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt); scaler.update(); opt.zero_grad()
            losses.append(loss.item())
    model.eval()
    return {"loss_first": round(losses[0], 3), "loss_last": round(losses[-1], 3),
            "nan_steps": sum(l != l for l in losses), "steps": len(losses),
            "peak_mem_gb": round(torch.cuda.max_memory_allocated(0) / 1e9, 2)}

# ---------- test 3: one adapter (skill A) ----------
from peft import LoraConfig, get_peft_model
def t3():
    global model
    lin = sorted({n.split(".")[-1] for n, m in model.named_modules()
                  if isinstance(m, nn.Linear) and ".layers." in n and "visual" not in n and "mtp" not in n})
    rx = r"^(?!.*(visual|mtp)).*\.layers\.\d+\..*\.(" + "|".join(lin) + r")$"
    cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, target_modules=rx)
    model = get_peft_model(model, cfg, adapter_name="A")
    R["lora_targets"] = lin
    params = [p for p in model.parameters() if p.requires_grad]
    tr = train(params, A_TR)
    return {"targets": lin, "trainable_m": round(sum(p.numel() for p in params) / 1e6, 2), "train": tr,
            "skill_a": skill_acc(model, A_TE), "skill_b": skill_acc(model, B_TE),
            "gsm8k_100": gsm_acc(model, GSM)}
run("t3_adapter_A", t3)

# ---------- test 4: second adapter (skill B), then both together ----------
def t4():
    cfg = model.peft_config["A"]
    model.add_adapter("B", cfg)
    model.set_adapter("B")
    for n, p in model.named_parameters():
        p.requires_grad = ".B." in n and "lora_" in n
    params = [p for p in model.parameters() if p.requires_grad]
    tr = train(params, B_TR)
    res = {"train": tr, "B_only_skill_b": skill_acc(model, B_TE)}
    model.base_model.set_adapter(["A", "B"])
    res.update({"both_skill_a": skill_acc(model, A_TE), "both_skill_b": skill_acc(model, B_TE),
                "both_gsm8k_100": gsm_acc(model, GSM)})
    return res
run("t4_two_adapters", t4)

# ---------- test 5: sparse memory layer (skill A), adapters off ----------
class Mem(nn.Module):
    def __init__(s, d, n=8192, k=32):
        super().__init__()
        s.keys = nn.Parameter(torch.randn(n, d) / d ** 0.5)
        s.vals = nn.Parameter(torch.zeros(n, d))
        s.k = k
    def forward(s, x):
        q = x.float()
        q = q / (q.norm(dim=-1, keepdim=True) + 1e-6)
        sc, i = (q @ s.keys.t()).topk(s.k, dim=-1)
        w = sc.softmax(-1).unsqueeze(-1)
        return (w * s.vals[i]).sum(-2).to(x.dtype)

def t5():
    layers = [m for n, m in model.named_modules() if isinstance(m, nn.ModuleList) and len(m) == 24 and "visual" not in n][0]
    tgt = layers[len(layers) // 2].mlp
    d = model.config.get_text_config().hidden_size
    mem = Mem(d).cuda()
    h = tgt.register_forward_hook(lambda mod, inp, out: out + mem(inp[0]))
    for p in model.parameters():
        p.requires_grad = False
    import contextlib
    ctx = model.disable_adapter() if hasattr(model, "disable_adapter") else contextlib.nullcontext()
    with ctx:
        tr = train(list(mem.parameters()), A_TR, epochs=3, lr=1e-3)
        res = {"layer": len(layers) // 2, "slots": 8192, "topk": 32, "train": tr,
               "skill_a": skill_acc(model, A_TE), "skill_b": skill_acc(model, B_TE),
               "gsm8k_100": gsm_acc(model, GSM)}
    h.remove()
    return res
run("t5_memory_layer", t5)

R["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
save()
log("DONE")
