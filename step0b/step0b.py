# Software4 step 0b: can replay stop forgetting, and does a router stop experts interfering?
# Tests: r1 expert A + replay, r2 expert B + replay, r3 both-on vs router, r4 memory layer + replay.
# Writes /kaggle/working/results.json after every test.
import json, os, random, re, subprocess, sys, time, traceback
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

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
            # micro-batches of MB (replay answers are long; 248K-vocab logits are big), same effective batch
            MB, tot, bad = 2, 0.0, False
            for s in range(0, ids.shape[0], MB):
                a = att[s:s + MB]; L = min(int(a.sum(1).max()), 384)
                with torch.autocast("cuda", dtype=torch.float16):
                    loss = model(input_ids=ids[s:s + MB, :L], attention_mask=a[:, :L], labels=labs[s:s + MB, :L]).loss
                if not torch.isfinite(loss):
                    bad = True; break
                n = min(MB, ids.shape[0] - s) / ids.shape[0]
                scaler.scale(loss * n).backward(); tot += loss.item() * n
                del loss
            if bad:
                losses.append(float("nan")); opt.zero_grad(); continue
            loss = torch.tensor(tot)
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt); scaler.update(); opt.zero_grad()
            losses.append(loss.item())
    model.eval()
    return {"loss_first": round(losses[0], 3), "loss_last": round(losses[-1], 3),
            "nan_steps": sum(l != l for l in losses), "steps": len(losses),
            "peak_mem_gb": round(torch.cuda.max_memory_allocated(0) / 1e9, 2)}

# ---------- load + baseline ----------
model = load(torch.float16, 0); model.eval()
GSM = gsm8k(100)
run("b0_baseline", lambda: {"gsm8k_100": gsm_acc(model, GSM)})

# ---------- replay set: general prompts answered by the pre-update model ----------
REPLAY = []
def make_replay():
    from datasets import load_dataset
    g = load_dataset("openai/gsm8k", "main", split="train").shuffle(seed=0).select(range(200))
    qs = [r["question"] + "\nThink step by step briefly, then end with: The answer is N." for r in g]
    src = {"gsm8k_train": len(qs)}
    try:
        al = load_dataset("tatsu-lab/alpaca", split="train").shuffle(seed=0)
        al = [r["instruction"] for r in al if not r["input"]][:200]
        qs += al; src["alpaca"] = len(al)
    except Exception as e:
        src["alpaca_error"] = repr(e)[:200]
    ans = generate(model, qs, max_new=256)
    REPLAY.extend((q, a) for q, a in zip(qs, ans) if a)
    return {"sources": src, "n": len(REPLAY), "sample": REPLAY[0][1][:200]}
run("b1_replay_set", make_replay)

def mix(new):
    m = list(new) + REPLAY
    random.Random(7).shuffle(m)
    return m

# ---------- r1: expert A + replay ----------
from peft import LoraConfig, get_peft_model
def r1():
    global model
    lin = sorted({n.split(".")[-1] for n, m in model.named_modules()
                  if isinstance(m, nn.Linear) and ".layers." in n and "visual" not in n and "mtp" not in n})
    rx = r"^(?!.*(visual|mtp)).*\.layers\.\d+\..*\.(" + "|".join(lin) + r")$"
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, target_modules=rx), adapter_name="A")
    params = [p for p in model.parameters() if p.requires_grad]
    tr = train(params, mix(A_TR))
    return {"replay_frac": round(len(REPLAY) / (len(REPLAY) + len(A_TR)), 2), "train": tr,
            "skill_a": skill_acc(model, A_TE), "gsm8k_100": gsm_acc(model, GSM)}
run("r1_expert_A_replay", r1)

# ---------- r2: expert B + replay (A frozen) ----------
def r2():
    model.add_adapter("B", model.peft_config["A"])
    model.set_adapter("B")
    for n, p in model.named_parameters():
        p.requires_grad = ".B." in n and "lora_" in n
    tr = train([p for p in model.parameters() if p.requires_grad], mix(B_TR))
    return {"train": tr, "skill_b": skill_acc(model, B_TE), "gsm8k_100": gsm_acc(model, GSM)}
run("r2_expert_B_replay", r2)

# ---------- r3: both experts summed vs a router ----------
@torch.no_grad()
def embed(prompts, bs=16):
    tok.padding_side = "left"; out = []
    with model.disable_adapter():
        for i in range(0, len(prompts), bs):
            enc = tok([chat(p) for p in prompts[i:i + bs]], return_tensors="pt", padding=True).to(0)
            h = model(**enc, output_hidden_states=True).hidden_states[-1].float()
            m = enc["attention_mask"].unsqueeze(-1).float()
            e = (h * m).sum(1) / m.sum(1)
            out.append(torch.nn.functional.normalize(e, dim=-1).cpu())
    return torch.cat(out)

def r3():
    res = {}
    model.base_model.set_adapter(["A", "B"])
    res["summed"] = {"skill_a": skill_acc(model, A_TE)[0], "skill_b": skill_acc(model, B_TE)[0],
                     "gsm8k_100": gsm_acc(model, GSM)}
    # router: nearest centroid of the frozen core's prompt embedding (A, B, or none)
    cents = torch.stack([embed([q for q, _ in A_TR[:100]]).mean(0), embed([q for q, _ in B_TR[:100]]).mean(0),
                         embed([q for q, _ in REPLAY[:100]] + [q for q, _ in REPLAY[-100:]]).mean(0)])
    names = ["A", "B", None]
    gsm_prompts = [q + "\nThink step by step briefly, then end with: The answer is N." for q, _ in GSM]
    tests = {"skill_a": ([q for q, _ in A_TE], "A"), "skill_b": ([q for q, _ in B_TE], "B"), "gsm8k_100": (gsm_prompts, None)}
    routed = {}
    for k, (ps, want) in tests.items():
        r = (embed(ps) @ cents.t()).argmax(-1).tolist()
        routed[k] = ([names[i] for i in r], want)
    res["router_acc"] = {k: round(sum(c == w for c in ch) / len(ch), 3) for k, (ch, w) in routed.items()}
    def routed_eval(k):
        ps = tests[k][0]; choice = routed[k][0]; preds = [None] * len(ps)
        for name in ["A", "B", None]:
            idx = [i for i, c in enumerate(choice) if c == name]
            if not idx: continue
            sub = [ps[i] for i in idx]
            mx = 384 if k == "gsm8k_100" else 40
            if name is None:
                with model.disable_adapter():
                    o = generate(model, sub, max_new=mx)
            else:
                model.base_model.set_adapter(name); o = generate(model, sub, max_new=mx)
            for i, p in zip(idx, o): preds[i] = p
        return preds
    pa = routed_eval("skill_a"); pb = routed_eval("skill_b"); pg = routed_eval("gsm8k_100")
    ok_g = 0
    for p, (_, a) in zip(pg, GSM):
        g = num(p.split("answer is")[-1] if "answer is" in p else p)
        try: ok_g += float(g) == float(a)
        except (TypeError, ValueError): pass
    res["routed"] = {"skill_a": round(sum(p == a for p, (_, a) in zip(pa, A_TE)) / 100, 3),
                     "skill_b": round(sum(p == a for p, (_, a) in zip(pb, B_TE)) / 100, 3),
                     "gsm8k_100": round(ok_g / len(GSM), 3)}
    return res
run("r3_summed_vs_router", r3)

# ---------- r4: sparse memory layer + replay (adapters off) ----------
class Mem(nn.Module):
    def __init__(s, d, n=8192, k=32):
        super().__init__()
        s.keys = nn.Parameter(torch.randn(n, d) / d ** 0.5)
        s.vals = nn.Parameter(torch.zeros(n, d))
        s.k = k
    def forward(s, x):
        q = x.float(); q = q / (q.norm(dim=-1, keepdim=True) + 1e-6)
        sc, i = (q @ s.keys.t()).topk(s.k, dim=-1)
        return (sc.softmax(-1).unsqueeze(-1) * s.vals[i]).sum(-2).to(x.dtype)

def r4():
    layers = [m for n, m in model.named_modules() if isinstance(m, nn.ModuleList) and len(m) == 24 and "visual" not in n][0]
    tgt = layers[len(layers) // 2].mlp
    mem = Mem(model.config.get_text_config().hidden_size).cuda()
    h = tgt.register_forward_hook(lambda mod, inp, out: out + mem(inp[0]))
    for p in model.parameters(): p.requires_grad = False
    with model.disable_adapter():
        tr = train(list(mem.parameters()), mix(A_TR), epochs=3, lr=1e-3)
        res = {"train": tr, "skill_a": skill_acc(model, A_TE), "gsm8k_100": gsm_acc(model, GSM)}
    h.remove()
    return res
# r4 finished in run 2 (results_run2.json); skipped to save GPU quota

R["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
save()
log("DONE")
