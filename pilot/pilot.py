# Software4 pilot (build step 1b): 3 sleep cycles, 3 new skills, experts+router vs memory layer vs text peers.
# See pilot/DESIGN.md. Roles (S4_ROLE): driver (default), replay, baseline, experts, memory, peers, selftest.
# S4_SMOKE=1: 2 training steps on a worst-case batch per arm + tiny eval. S4_TINY=1: CPU run on a tiny random model.
import contextlib, glob, hashlib, json, math, os, random, re, shutil, subprocess, sys, time, traceback
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

ENV = os.environ.get
TINY = ENV("S4_TINY") == "1"
# Kaggle cannot set env vars, so the default is edited in this file: "1" = smoke run, "0" = full run.
SMOKE = ENV("S4_SMOKE", "0") == "1"
SEED = int(ENV("S4_SEED", "0"))
ROLE = ENV("S4_ROLE", "driver")
MID = ENV("S4_MODEL", "Qwen/Qwen3.5-2B")
OUTROOT = ENV("S4_OUT") or ("/kaggle/working" if os.path.isdir("/kaggle/working") else ".")
WORK = os.path.join(OUTROOT, "pilot")
INPUT_ROOT = ENV("S4_INPUT", "/kaggle/input")
SKILL_KEYS = ["s1", "s2", "s3"]


def make_cfg():
    c = dict(n_train=100, n_test=100, n_gate=30, n_gsm_test=100, n_gsm_gate=30, n_arc_test=100, n_arc_gate=30,
             steps=80, bs=16, mb=2, maxlen=384, replay_src=200, replay_per_sleep=150, replay_new=256,
             gsm_new=384, arc_new=8, skill_new=40, note_ex=10, note_new=100, n_sleeps=3,
             lr_exp=2e-4, lr_mem=1e-3, lora_r=16, lora_alpha=32, mem_slots=8192, mem_topk=32, gate_drop_pts=10.0,
             rag_k=3, mode="full")
    if SMOKE:
        c.update(n_test=8, n_gate=4, n_gsm_test=8, n_gsm_gate=4, n_arc_test=8, n_arc_gate=4, steps=2,
                 replay_src=8, replay_new=64, gsm_new=64, note_new=60, n_sleeps=2, mode="smoke")
    if TINY:
        c.update(n_train=12, n_test=6, n_gate=4, n_gsm_test=6, n_gsm_gate=4, n_arc_test=6, n_arc_gate=4, steps=3,
                 bs=4, replay_src=8, replay_per_sleep=8, replay_new=10, gsm_new=8, arc_new=4, skill_new=8,
                 note_ex=4, note_new=8, mem_slots=64, mem_topk=4, lora_r=4, lora_alpha=8, maxlen=96,
                 mode="tiny", n_sleeps=3)
    return c


CFG = make_cfg()
FP = hashlib.sha1(json.dumps({"seed": SEED, "model": MID, "cfg": CFG}, sort_keys=True).encode()).hexdigest()[:10]


def log(*a):
    print(f"[{ROLE}]", *a, flush=True)


def sh(cmd):
    log("$", cmd)
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    log((p.stdout + p.stderr)[-1500:])
    return p.returncode


def jclean(o):
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {str(k): jclean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jclean(v) for v in o]
    return o


def save_json(path, obj):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(jclean(obj), f, indent=2)
    os.replace(tmp, path)


def find_saved(rel):
    """Path of WORK/rel, copied from an attached earlier run (/kaggle/input/**/pilot/rel) if needed."""
    p = os.path.join(WORK, rel)
    if os.path.exists(p):
        return p
    for c in sorted(glob.glob(os.path.join(INPUT_ROOT, "**", "pilot", rel), recursive=True)):
        os.makedirs(os.path.dirname(p), exist_ok=True)
        shutil.copy(c, p)
        log("reused from earlier run:", c)
        return p
    return None


def load_json_fp(rel, **must_match):
    p = find_saved(rel)
    if not p:
        return None
    try:
        d = json.load(open(p))
    except Exception:
        return None
    if d.get("fp") != FP:
        log("ignoring saved", rel, "(settings differ)")
        return None
    for k, v in must_match.items():
        if d.get(k) != v:
            log("ignoring saved", rel, f"({k} differs)")
            return None
    return d


class Res(dict):
    def __init__(self, role):
        super().__init__()
        self.path = os.path.join(WORK, f"results_{role}.json")
        self["role"] = role
        self["fp"] = FP
        self["started"] = time.strftime("%Y-%m-%d %H:%M:%S")

    def save(self):
        save_json(self.path, self)


def is_bad(x):
    return isinstance(x, dict) and ("error" in x or "skipped" in x)


def step(R, name, fn, deps=()):
    """Run fn() into R[name]. If a dependency failed or was skipped, mark this one skipped (no numbers)."""
    for d in deps:
        if d not in R or is_bad(R[d]):
            R[name] = {"skipped": f"depends on {d}, which failed or was skipped"}
            R.save()
            log(name, "SKIPPED (dep", d, ")")
            return False
    t = time.time()
    try:
        out = fn()
        out = out if isinstance(out, dict) else {"value": out}
        ok = True
    except BaseException as e:
        if isinstance(e, KeyboardInterrupt):
            raise
        out = {"error": repr(e), "trace": traceback.format_exc()[-2500:]}
        ok = False
        log(name, "FAILED", repr(e))
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
    out["seconds"] = round(time.time() - t)
    R[name] = out
    R.save()
    log(name, json.dumps(jclean(out))[:600])
    return ok


# =====================================================================================
# Tasks: 3 skills, one template, same word lists, differing only in the invented procedure name
# =====================================================================================
WORDS = ("apple river stone cloud tiger music garden window pencil rocket candle forest ocean silver dragon "
         "planet bridge castle mirror shadow thunder valley meadow lantern harbor island jungle desert winter "
         "summer autumn spring copper marble velvet coffee butter honey pepper ginger lemon mango cherry orange "
         "violet purple yellow basket bottle button carpet cookie engine falcon guitar hammer helmet jacket "
         "kettle ladder magnet needle oyster parrot pillow puzzle rabbit saddle tablet tunnel wallet zebra "
         "anchor badger camera dentist eagle feather goblin hockey igloo jigsaw koala lizard monkey noodle "
         "pirate quilt raven salmon turtle umbrella vulture walrus yogurt acorn barrel cactus donkey elbow "
         "fossil glacier hermit insect jester kitten lobster mammoth nectar onion pebble quartz robot spider "
         "trumpet unicorn vessel wizard").split()

NAMES = ["zorvel", "quillon", "mardeth"]


def proc_zorvel(w):
    return " ~ ".join(x.capitalize() for x in sorted(w)) + f" [{len(w)}]"


def proc_quillon(w):
    return "".join(x[0].upper() for x in w) + "/" + w[-1]


def proc_mardeth(w):
    return "-".join(reversed(w)) + " #" + str(sum(x[0] < "n" for x in w))


PROCS = [proc_zorvel, proc_quillon, proc_mardeth]
RULES = [
    "zorvel sorts the words alphabetically, capitalizes each word, joins them with ' ~ ' (a space, a tilde, a space), "
    "and then appends the number of words in square brackets after a space, for example ' [4]'.",
    "quillon writes the first letter of each word, in the given order, in capitals with no separators, then a slash, "
    "then the last word of the list exactly as given.",
    "mardeth writes the words in reverse order, joined by hyphens, then a space, then '#' followed by the number of "
    "words whose first letter comes before 'n' in the alphabet (a to m).",
]


def prompt_for(name, w):
    return f"Run the procedure {name} on this list: {', '.join(w)}. Reply with the result only."


def build_skill_data(seed=SEED, cfg=None):
    """data[i] = {'train': [...], 'test': [...], 'gate': [...]} of (prompt, answer). Separate seeds per split,
    and no word list (as a set) is reused anywhere, so all sets are disjoint."""
    cfg = cfg or CFG
    used, data = set(), []
    for s, name in enumerate(NAMES):
        d = {}
        for si, (split, n) in enumerate([("train", cfg["n_train"]), ("test", cfg["n_test"]), ("gate", cfg["n_gate"])]):
            rng = random.Random(seed * 100003 + 1000 * (s + 1) + 17 * (si + 1))
            items = []
            while len(items) < n:
                w = rng.sample(WORDS, rng.randint(3, 5))
                key = tuple(sorted(w))
                if key in used:
                    continue
                used.add(key)
                items.append((prompt_for(name, w), PROCS[s](w)))
            d[split] = items
        data.append(d)
    return data


def grade_skill(pred, gold):
    return pred is not None and pred.strip() == gold


def num(s):
    m = re.findall(r"-?\d[\d,]*\.?\d*", s)
    return m[-1].replace(",", "").rstrip(".") if m else None


def gsm_prompt(q):
    return q + "\nThink step by step briefly, then end with: The answer is N."


def grade_gsm(pred, gold):
    tail = pred.split("answer is")[-1] if "answer is" in pred else pred
    g = num(tail)
    try:
        return float(g) == float(gold)
    except (TypeError, ValueError):
        return False


def arc_prompt(q, texts):
    lines = [f"{'ABCDE'[i]}. {t}" for i, t in enumerate(texts)]
    return q + "\n" + "\n".join(lines) + "\nAnswer with the letter of the correct choice only."


def arc_letter(text):
    t = text.strip()
    m = (re.match(r"^\W*([A-E])\b", t) or re.search(r"answer is\W*([A-E])\b", t) or re.search(r"\b([A-E])\b", t))
    return m.group(1) if m else None


def grade_arc(pred, gold):
    return arc_letter(pred) == gold


def load_gsm_test():
    """GSM8K test: first n_gsm_test rows are the canary, the next n_gsm_gate rows are the gate set."""
    a, b = CFG["n_gsm_test"], CFG["n_gsm_gate"]
    if TINY:
        rng = random.Random(11)
        rows = [(f"Tom has {x} apples and gets {y} more. How many apples now?", str(x + y))
                for x, y in [(rng.randint(1, 50), rng.randint(1, 50)) for _ in range(a + b)]]
    else:
        from datasets import load_dataset
        ds = load_dataset("openai/gsm8k", "main", split="test").select(range(a + b))
        rows = [(r["question"], r["answer"].split("####")[-1].strip().replace(",", "")) for r in ds]
    return rows[:a], rows[a:a + b]


def load_arc_test():
    """ARC-Challenge test (not in replay): shuffled with a fixed seed, first n_arc_test = canary, next = gate."""
    a, b = CFG["n_arc_test"], CFG["n_arc_gate"]
    if TINY:
        rng = random.Random(12)
        rows = []
        for i in range(a + b):
            w = rng.sample(WORDS, 4)
            k = rng.randrange(4)
            rows.append((arc_prompt(f"Which of these is the word number {i}?", w), "ABCD"[k]))
    else:
        from datasets import load_dataset
        ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test").shuffle(seed=0).select(range(a + b))
        rows = []
        for r in ds:
            labels, texts = r["choices"]["label"], r["choices"]["text"]
            rows.append((arc_prompt(r["question"], texts), "ABCDE"[labels.index(r["answerKey"])]))
    return rows[:a], rows[a:a + b]


def load_replay_prompts():
    n = CFG["replay_src"]
    if TINY:
        rng = random.Random(13)
        g = [gsm_prompt(f"Sam has {rng.randint(1, 30)} pens and loses {rng.randint(1, 9)}. How many are left?") for _ in range(n)]
        al = [f"Write one short sentence about a {rng.choice(WORDS)}." for _ in range(n)]
        return g + al, {"gsm8k_train": n, "alpaca": n}
    from datasets import load_dataset
    g = load_dataset("openai/gsm8k", "main", split="train").shuffle(seed=0).select(range(n))
    qs = [gsm_prompt(r["question"]) for r in g]
    src = {"gsm8k_train": len(qs)}
    try:
        al = load_dataset("tatsu-lab/alpaca", split="train").shuffle(seed=0)
        al = [r["instruction"] for r in al if not r["input"]][:n]
        qs += al
        src["alpaca"] = len(al)
    except Exception as e:
        src["alpaca_error"] = repr(e)[:200]
    return qs, src


# =====================================================================================
# Go / no-go (pure python, fixed before running; also tested by the selftest role)
# =====================================================================================
LEARN_PTS, KEEP_PTS, CANARY_PTS, TIE_PTS = 30.0, 10.0, 10.0, 9.0


def pts(x):
    return round(100.0 * x, 6)


def arm_judgement(base, sleeps, n=3):
    """base: {'s1','s2','s3','gsm','arc'}; sleeps: list of per-sleep records with 'test'. Returns criteria dict."""
    if len(sleeps) < n or any(is_bad(s) or not isinstance(s.get("test"), dict) or is_bad(s["test"]) for s in sleeps[:n]):
        return {"evaluable": False, "reason": "a sleep or its test scoring failed or was skipped"}
    try:
        final = sleeps[n - 1]["test"]
        learned, kept, rows = True, True, {}
        for i, k in enumerate(SKILL_KEYS[:n]):
            own = sleeps[i]["test"][k]
            gain = pts(final[k]) - pts(base[k])
            drop = pts(own) - pts(final[k])
            rows[k] = {"base": pts(base[k]), "after_own_sleep": pts(own), "final": pts(final[k]),
                       "gain_over_base_pts": round(gain, 3), "drop_since_own_sleep_pts": round(drop, 3),
                       "learned": gain >= LEARN_PTS - 1e-9, "kept": drop <= KEEP_PTS + 1e-9}
            learned &= rows[k]["learned"]
            kept &= rows[k]["kept"]
        old = {}
        for k in ("gsm", "arc"):
            d = pts(base[k]) - pts(final[k])
            old[k] = {"base": pts(base[k]), "final": pts(final[k]), "drop_pts": round(d, 3), "ok": d <= CANARY_PTS + 1e-9}
        old_ok = all(v["ok"] for v in old.values())
        mean = sum(final[k] for k in SKILL_KEYS[:n]) / n
        return {"evaluable": True, "learned": learned, "kept": kept, "old_skills_ok": old_ok,
                "meets_all": bool(learned and kept and old_ok), "skills": rows, "canaries": old,
                "mean_skill_pts": pts(mean)}
    except (KeyError, TypeError) as e:
        return {"evaluable": False, "reason": f"missing numbers: {e!r}"}


def text_peer_summary(peers, n=3):
    out = {}
    for name in ("rag", "notes"):
        try:
            c = peers[name][f"cycle{n}"]
            if is_bad(c) or any(c.get(k) is None for k in SKILL_KEYS[:n]):
                continue
            out[name] = {k: pts(c[k]) for k in SKILL_KEYS[:n]}
            out[name]["mean"] = round(sum(c[k] for k in SKILL_KEYS[:n]) / n * 100, 3)
        except (KeyError, TypeError):
            continue
    return out


def go_nogo(res):
    n = res.get("meta", {}).get("n_sleeps", 3)
    if n < 3:
        return {"computed": False, "reason": f"only {n} sleeps run (smoke mode); go/no-go needs 3"}
    base = (res.get("baseline") or {}).get("test")
    if not base or is_bad(base):
        return {"computed": False, "reason": "baseline missing"}
    peers = text_peer_summary(res.get("peers") or {}, n)
    best = max(peers.items(), key=lambda kv: kv[1]["mean"]) if peers else None
    out = {"computed": True, "rules": {"learned": f">= {LEARN_PTS} pts above plain base, each skill",
                                        "kept": f"<= {KEEP_PTS} pts below its score right after its own sleep, each skill",
                                        "old_skills": f"GSM8K and ARC each <= {CANARY_PTS} pts below base",
                                        "tie_band": f"within {TIE_PTS} pts of best text peer (mean of 3 skills) counts as a tie"},
           "text_peers_cycle3": peers, "best_text_peer": best[0] if best else None, "arms": {}}
    for arm in ("experts", "memory"):
        a = res.get(arm) or {}
        j = arm_judgement(base, a.get("sleeps", []), n) if not is_bad(a) else {"evaluable": False, "reason": "arm failed or skipped"}
        if j.get("evaluable") and best:
            diff = j["mean_skill_pts"] - best[1]["mean"]
            j["vs_best_text_peer"] = {"peer": best[0], "peer_mean_pts": best[1]["mean"], "diff_pts": round(diff, 3),
                                      "verdict": "beats" if diff > TIE_PTS else ("loses" if diff < -TIE_PTS else "ties")}
        elif j.get("evaluable"):
            j["vs_best_text_peer"] = {"verdict": "unknown (text peers failed or skipped)"}
        out["arms"][arm] = j
    out["go"] = any(j.get("meets_all") for j in out["arms"].values())
    out["go_arms"] = [a for a, j in out["arms"].items() if j.get("meets_all")]
    out["note"] = ("Flag separately: an arm can pass go while losing to the best text peer; that decision is the user's.")
    return out


def merge_results(statuses=None):
    res = {"meta": {"seed": SEED, "fp": FP, "model": MID, "n_sleeps": CFG["n_sleeps"], "cfg": CFG}}
    for role in ("replay", "baseline", "experts", "memory", "peers"):
        p = os.path.join(WORK, f"results_{role}.json")
        if os.path.exists(p):
            try:
                res[role] = json.load(open(p))
            except Exception as e:
                res[role] = {"error": f"unreadable results file: {e!r}"}
        else:
            res[role] = {"error": "no results file", "exit": (statuses or {}).get(role)}
    bp = os.path.join(WORK, "baseline.json")
    base = json.load(open(bp)) if os.path.exists(bp) else None
    res["baseline"] = base if base else {"error": "baseline.json missing", "role_results": res["baseline"]}
    res["go_nogo"] = go_nogo(res)
    res["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    save_json(os.path.join(OUTROOT, "results.json"), res)
    return res


# =====================================================================================
# Driver
# =====================================================================================
def detect_gpus():
    if TINY:
        return 2
    try:
        p = subprocess.run("nvidia-smi -L", shell=True, capture_output=True, text=True)
        return max(1, len([l for l in p.stdout.splitlines() if l.startswith("GPU")]))
    except Exception:
        return 1


def spawn(role, gpu):
    env = dict(os.environ, S4_ROLE=role)
    if not TINY:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    log("spawn", role, "on GPU", gpu)
    me = os.path.abspath(globals().get("__file__") or sys.argv[0])
    return subprocess.Popen([sys.executable, me], env=env)


def driver():
    os.makedirs(WORK, exist_ok=True)
    D = {"setup": {}}
    if not TINY:
        sh("pip install -q -U 'transformers>=5.0' peft accelerate")
        sh("pip uninstall -y torchao")  # Kaggle's old torchao breaks peft's LoRA dispatch
        fla_rc = sh("pip install -q -U flash-linear-attention")
        smoke_py = ('import torch\nfrom transformers import AutoTokenizer, AutoModelForCausalLM\n'
                    f'MID="{MID}"\ntry:\n m=AutoModelForCausalLM.from_pretrained(MID,dtype=torch.float16,device_map={{"":0}})\n'
                    'except Exception:\n from transformers import AutoModelForImageTextToText\n'
                    ' m=AutoModelForImageTextToText.from_pretrained(MID,dtype=torch.float16,device_map={"":0})\n'
                    't=AutoTokenizer.from_pretrained(MID)\nx=t("Hello, my name is",return_tensors="pt").to(0)\n'
                    'y=m.generate(**x,max_new_tokens=8,do_sample=False)\nassert torch.isfinite(m(**x).logits).all()\n'
                    'print("SMOKE_OK",t.decode(y[0]))\n')
        with open("fla_smoke.py", "w") as f:
            f.write(smoke_py)
        fast = False
        if fla_rc == 0:
            p = subprocess.run([sys.executable, "fla_smoke.py"], capture_output=True, text=True)
            fast = "SMOKE_OK" in p.stdout
            log("smoke with fla:", fast, (p.stdout + p.stderr)[-800:])
            if not fast:
                sh("pip uninstall -y flash-linear-attention fla-core")
        D["setup"] = {"fla_installed": fla_rc == 0, "fast_path_smoke_ok": fast}
    ngpu = detect_gpus()
    D["setup"]["ngpu"] = ngpu
    save_json(os.path.join(WORK, "driver.json"), D)
    status = {}

    def run_jobs(jobs):
        """jobs: list of (role, gpu). Run concurrently if enough GPUs, else one after another."""
        if ngpu >= 2:
            ps = [(r, spawn(r, g % ngpu)) for r, g in jobs]
            for r, p in ps:
                status[r] = p.wait()
        else:
            for r, g in jobs:
                status[r] = spawn(r, 0).wait()

    run_jobs([("replay", 0), ("baseline", 1)])
    # phase 2: experts on GPU 0, memory on GPU 1; text peers start on whichever GPU frees first
    if ngpu >= 2:
        procs = {"experts": (spawn("experts", 0), 0), "memory": (spawn("memory", 1), 1)}
        peers_started = False
        while procs:
            time.sleep(5)
            for r in list(procs):
                p, g = procs[r]
                if p.poll() is not None:
                    status[r] = p.returncode
                    del procs[r]
                    if not peers_started:
                        procs["peers"] = (spawn("peers", g), g)
                        peers_started = True
    else:
        run_jobs([("experts", 0), ("memory", 0), ("peers", 0)])
    res = merge_results(status)
    log("GO/NO-GO:", json.dumps(res["go_nogo"], indent=1))
    log("DONE")


# =====================================================================================
# Worker: model, generation, training
# =====================================================================================
tok = MODEL = DEV = None
torch = nn = None
NAN_STEPS = 0
PEAK_GB = 0.0
EMB_CACHE = {}
IM_END = "<|im_end|>"


def init_worker():
    global torch, nn, tok, DEV
    import torch as _t
    import torch.nn as _nn
    torch, nn = _t, _nn
    DEV = "cuda" if (torch.cuda.is_available() and not TINY) else "cpu"
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    torch.manual_seed(SEED)


def chat(user):
    msgs = [{"role": "user", "content": user}]
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except Exception:
        return f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def build_tiny():
    from transformers import AutoConfig, AutoModelForCausalLM
    cfg = AutoConfig.from_pretrained(MID)
    tc = cfg.get_text_config()
    nl = 4
    over = dict(hidden_size=64, intermediate_size=128, num_hidden_layers=nl, num_attention_heads=4, num_key_value_heads=2,
                head_dim=16, linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=16,
                linear_value_head_dim=16, layer_types=["linear_attention"] * (nl - 1) + ["full_attention"],
                max_position_embeddings=1024)
    for k, v in over.items():
        if hasattr(tc, k):
            setattr(tc, k, v)
    for k in ("num_nextn_predict_layers", "mtp_num_hidden_layers"):
        if hasattr(tc, k):
            setattr(tc, k, 0)
    vc = getattr(cfg, "vision_config", None)
    if vc is not None:
        for k, v in dict(depth=1, hidden_size=32, intermediate_size=64, num_heads=2, out_hidden_size=64).items():
            if hasattr(vc, k):
                setattr(vc, k, v)
    torch.manual_seed(SEED)
    m = AutoModelForCausalLM.from_config(cfg, dtype=torch.float32)
    return m


def load_model():
    if TINY:
        return build_tiny().to(DEV).eval()
    from transformers import AutoModelForCausalLM
    try:
        m = AutoModelForCausalLM.from_pretrained(MID, dtype=torch.float16, device_map={"": 0})
    except Exception as e:
        log("CausalLM load failed, trying ImageTextToText:", repr(e)[:300])
        from transformers import AutoModelForImageTextToText
        m = AutoModelForImageTextToText.from_pretrained(MID, dtype=torch.float16, device_map={"": 0})
    return m.eval()


def text_cfg(model):
    return model.config.get_text_config()


def generate(model, prompts, max_new, bs=16):
    tok.padding_side = "left"
    outs = []
    with torch.no_grad():
        for i in range(0, len(prompts), bs):
            enc = tok([chat(p) for p in prompts[i:i + bs]], return_tensors="pt", padding=True).to(model.device)
            y = model.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id)
            outs += tok.batch_decode(y[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    return [o.strip() for o in outs]


def embed(model, prompts, off=None, bs=16):
    """Mean-pooled last hidden state of the frozen core (adapters off), normalised. Cached per text."""
    todo = [p for p in dict.fromkeys(prompts) if p not in EMB_CACHE]
    tok.padding_side = "left"
    with torch.no_grad():
        for i in range(0, len(todo), bs):
            chunk = todo[i:i + bs]
            enc = tok([chat(p) for p in chunk], return_tensors="pt", padding=True).to(model.device)
            with (off() if off else contextlib.nullcontext()):
                h = model(**enc, output_hidden_states=True).hidden_states[-1].float()
            m = enc["attention_mask"].unsqueeze(-1).float()
            e = torch.nn.functional.normalize((h * m).sum(1) / m.sum(1), dim=-1).cpu()
            for p, v in zip(chunk, e):
                EMB_CACHE[p] = v
    return torch.stack([EMB_CACHE[p] for p in prompts])


# ---------- systems under test ----------
class PlainSystem:
    """Base model (or base + memory layer hook): no routing."""
    def __init__(self, model, prefix=None):
        self.model, self.prefix = model, prefix

    def gen(self, prompts, max_new, tag=None):
        if self.prefix:
            prompts = [self.prefix(p) for p in prompts]
        return generate(self.model, prompts, max_new)


class ExpertSystem:
    """Frozen core + accepted experts; nearest-centroid router (adapters off) picks one expert or none."""
    def __init__(self, model, ks, skill_data, replay_prompts):
        self.model, self.ks = model, list(ks)
        self.names = [f"e{k}" for k in self.ks]
        off = model.disable_adapter
        cents = [embed(model, [q for q, _ in skill_data[k - 1]["train"]], off).mean(0) for k in self.ks]
        cents.append(embed(model, replay_prompts, off).mean(0))
        self.cents = torch.nn.functional.normalize(torch.stack(cents), dim=-1)
        self.labels = self.names + ["none"]
        self.routes = {}

    def route(self, prompts):
        e = embed(self.model, prompts, self.model.disable_adapter)
        return (e @ self.cents.t()).argmax(-1).tolist()

    def gen(self, prompts, max_new, tag=None):
        r = self.route(prompts)
        if tag:
            self.routes[tag] = [self.labels[i] for i in r]
        preds = [""] * len(prompts)
        for gi in sorted(set(r)):
            idx = [i for i, c in enumerate(r) if c == gi]
            sub = [prompts[i] for i in idx]
            if gi == len(self.names):
                with self.model.disable_adapter():
                    o = generate(self.model, sub, max_new)
            else:
                self.model.set_adapter(self.names[gi])
                o = generate(self.model, sub, max_new)
            for i, p in zip(idx, o):
                preds[i] = p
        return preds

    def router_report(self, seen_skills):
        conf, per, tot, ok = {}, {}, 0, 0
        excluded = []
        for tag, routed in self.routes.items():
            if tag.startswith("s"):
                want = f"e{tag[1:]}"
                if want not in self.names:
                    excluded.append(tag)
                    continue
            else:
                want = "none"
            row = conf.setdefault(want, {})
            for c in routed:
                row[c] = row.get(c, 0) + 1
            n = len(routed)
            good = sum(c == want for c in routed)
            per[want] = round(good / n, 3)
            tot += n
            ok += good
        macro = round(sum(per.values()) / len(per), 3) if per else None
        return {"overall": round(ok / tot, 3) if tot else None, "macro_per_class": macro, "per_class": per,
                "confusion_true_to_routed": conf, "skills_without_expert_excluded": excluded}


def evaluate(system, data, keys, split, gsm, arc):
    """keys: list like ['s1','s2']. gsm/arc: lists of (prompt-or-question, gold) already for this split."""
    out, samples = {}, {}
    for k in keys:
        items = data[int(k[1:]) - 1][split]
        preds = system.gen([q for q, _ in items], CFG["skill_new"], tag=k)
        out[k] = round(sum(grade_skill(p, a) for p, (_, a) in zip(preds, items)) / len(items), 3)
        samples[k] = preds[:2]
    preds = system.gen([gsm_prompt(q) for q, _ in gsm], CFG["gsm_new"], tag="gsm")
    out["gsm"] = round(sum(grade_gsm(p, a) for p, (_, a) in zip(preds, gsm)) / len(gsm), 3)
    preds = system.gen([q for q, _ in arc], CFG["arc_new"], tag="arc")
    out["arc"] = round(sum(grade_arc(p, a) for p, (_, a) in zip(preds, arc)) / len(arc), 3)
    samples["arc"] = preds[:2]
    return out, samples


# ---------- training ----------
def make_batch(items):
    tok.padding_side = "right"
    ids, labs = [], []
    for q, a in items:
        p = tok(chat(q), add_special_tokens=False)["input_ids"]
        full = tok(chat(q) + a + IM_END, add_special_tokens=False)["input_ids"]
        ids.append(full)
        labs.append([-100] * len(p) + full[len(p):])
    L = max(map(len, ids))
    att = [[1] * len(x) + [0] * (L - len(x)) for x in ids]
    ids = [x + [tok.pad_token_id] * (L - len(x)) for x in ids]
    labs = [x + [-100] * (L - len(x)) for x in labs]
    return (torch.tensor(ids).to(DEV), torch.tensor(att).to(DEV), torch.tensor(labs).to(DEV))


def worst_case_batch():
    """Smoke mode: every row hits the length cap, so memory use is the true maximum."""
    q = " ".join(["alpha"] * 40)
    a = " ".join(["beta"] * (CFG["maxlen"] + 50))
    return [(q, a)] * CFG["bs"]


def batch_stream(data, steps, bs, rng):
    if SMOKE:
        for _ in range(steps):
            yield worst_case_batch()
        return
    idx, n = list(range(len(data))), 0
    while n < steps:
        rng.shuffle(idx)
        for i in range(0, len(idx), bs):
            yield [data[j] for j in idx[i:i + bs]]
            n += 1
            if n >= steps:
                return


def train(model, params, data, steps, lr, seed):
    global NAN_STEPS, PEAK_GB
    use_cuda = DEV == "cuda"
    if use_cuda:
        torch.cuda.reset_peak_memory_stats()
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)
    rng = random.Random(seed)
    losses, MB, CAP = [], CFG["mb"], CFG["maxlen"]
    model.train()
    for batch in batch_stream(data, steps, CFG["bs"], rng):
        ids, att, labs = make_batch(batch)
        tot, bad = 0.0, False
        for s in range(0, ids.shape[0], MB):
            a = att[s:s + MB]
            L = min(int(a.sum(1).max()), CAP)
            lab = labs[s:s + MB, :L]
            if int((lab != -100).sum()) == 0:
                continue
            ctx = torch.autocast("cuda", dtype=torch.float16) if use_cuda else contextlib.nullcontext()
            with ctx:
                loss = model(input_ids=ids[s:s + MB, :L], attention_mask=a[:, :L], labels=lab).loss
            if not torch.isfinite(loss):
                bad = True
                break
            n = min(MB, ids.shape[0] - s) / ids.shape[0]
            scaler.scale(loss * n).backward()
            tot += loss.item() * n
            del loss
        if bad:
            losses.append(float("nan"))
            opt.zero_grad()
            continue
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad()
        losses.append(tot)
    model.eval()
    nan = sum(l != l for l in losses)
    NAN_STEPS += nan
    peak = round(torch.cuda.max_memory_allocated(0) / 1e9, 2) if use_cuda else None
    if peak:
        PEAK_GB = max(PEAK_GB, peak)
    return {"loss_first": losses[0] if losses else None, "loss_last": losses[-1] if losses else None,
            "nan_steps": nan, "steps": len(losses), "peak_mem_gb": peak, "n_train_rows": len(data),
            "lr": lr, "batch": CFG["bs"], "micro_batch": MB, "max_len": CAP}


# ---------- shared data for workers ----------
class World:
    def __init__(self):
        self.skills = build_skill_data()
        self.gsm_test, self.gsm_gate = load_gsm_test()
        self.arc_test, self.arc_gate = load_arc_test()
        self.base = None
        self.replay = None   # list of (q, a)

    def load_baseline(self):
        d = load_json_fp("baseline.json")
        if not d:
            raise RuntimeError("baseline.json missing (baseline step failed or was skipped)")
        self.base = d

    def load_replay(self):
        d = load_json_fp("replay.json")
        if not d:
            raise RuntimeError("replay.json missing (replay step failed or was skipped)")
        self.replay = [tuple(x) for x in d["items"]]
        self.replay_sha = hashlib.sha1(json.dumps(d["items"]).encode()).hexdigest()[:10]

    def replay_prompts(self):
        return [q for q, _ in self.replay]


# =====================================================================================
# Roles
# =====================================================================================
def role_replay():
    R = Res("replay")

    def go():
        d = load_json_fp("replay.json")
        if d:
            return {"reused": True, "n": len(d["items"]), "sources": d.get("sources")}
        model = load_model()
        qs, src = load_replay_prompts()
        ans = generate(model, qs, CFG["replay_new"])   # written by the pre-update (here: base) model
        if TINY:   # a random-weight model emits empty text; keep rows so the flow can be tested
            ans = [a or "ok" for a in ans]
        items = [(q, a) for q, a in zip(qs, ans) if a]
        if not items:
            raise RuntimeError("replay generation produced no non-empty answers")
        save_json(os.path.join(WORK, "replay.json"), {"fp": FP, "items": items, "sources": src})
        return {"n": len(items), "sources": src, "sample": items[0][1][:200]}
    step(R, "replay_set", go)


def role_baseline():
    R = Res("baseline")
    W = World()

    def go():
        d = load_json_fp("baseline.json")
        if d:
            return {"reused": True}
        model = load_model()
        sysm = PlainSystem(model)
        test, st = evaluate(sysm, W.skills, SKILL_KEYS, "test", W.gsm_test, W.arc_test)
        gate, sg = evaluate(sysm, W.skills, SKILL_KEYS, "gate", W.gsm_gate, W.arc_gate)
        save_json(os.path.join(WORK, "baseline.json"), {"fp": FP, "test": test, "gate": gate, "samples": st})
        return {"test": test, "gate": gate, "samples": st}
    step(R, "baseline_scores", go)


class ExpertsArm:
    name = "experts"

    def __init__(self, W):
        from peft import LoraConfig, get_peft_model
        self.W = W
        model = load_model()
        lin = sorted({n.split(".")[-1] for n, m in model.named_modules()
                      if isinstance(m, nn.Linear) and ".layers." in n and "visual" not in n and "mtp" not in n})
        rx = r"^(?!.*(visual|mtp)).*\.layers\.\d+\..*\.(" + "|".join(lin) + r")$"
        self.lcfg = LoraConfig(r=CFG["lora_r"], lora_alpha=CFG["lora_alpha"], lora_dropout=0.0, target_modules=rx)
        torch.manual_seed(SEED + 1)
        self.model = get_peft_model(model, self.lcfg, adapter_name="e1")
        self.accepted = []
        self.lr = CFG["lr_exp"]

    def ensure(self, name, k):
        if name not in self.model.peft_config:
            torch.manual_seed(SEED + k)
            self.model.add_adapter(name, self.lcfg)

    def lora_params(self, name):
        return [(n, p) for n, p in self.model.named_parameters() if f".{name}." in n and "lora_" in n]

    def old_fingerprint(self):
        return round(sum(p.detach().double().sum().item() for k in self.accepted for _, p in self.lora_params(f"e{k}")), 9)

    def snapshot(self, k):
        self.old_fp = self.old_fingerprint()

    def train(self, k, data, seed):
        name = f"e{k}"
        self.ensure(name, k)
        self.model.set_adapter(name)
        for n, p in self.model.named_parameters():
            p.requires_grad = f".{name}." in n and "lora_" in n
        params = [p for _, p in self.lora_params(name)]
        st = train(self.model, params, data, CFG["steps"], self.lr, seed)
        st["n_trainable"] = sum(p.numel() for p in params)
        st["older_experts_unchanged"] = self.old_fingerprint() == self.old_fp
        st["params_finite"] = all(bool(torch.isfinite(p).all()) for p in params)
        return st

    def system(self, extra=None):
        return ExpertSystem(self.model, self.accepted + ([extra] if extra else []), self.W.skills, self.W.replay_prompts())

    def commit(self, k):
        self.accepted.append(k)

    def rollback(self, k):
        pass   # the new expert is simply never routed to; older experts were never touched

    def save_state(self, k, accepted):
        sd = {n: p.detach().cpu() for n, p in self.lora_params(f"e{k}")}
        torch.save(sd, os.path.join(WORK, "experts", f"e{k}_{'accepted' if accepted else 'rejected'}.pt"))

    def load_state(self, k, accepted):
        if not accepted:
            return True
        p = find_saved(os.path.join("experts", f"e{k}_accepted.pt"))
        if not p:
            return False
        name = f"e{k}"
        self.ensure(name, k)
        sd = torch.load(p)
        with torch.no_grad():
            for n, prm in self.lora_params(name):
                prm.copy_(sd[n].to(prm.device, prm.dtype))
        self.accepted.append(k)
        return True


def make_mem_class():
    class _Mem(nn.Module):
        def __init__(s, d, n, k):
            super().__init__()
            s.keys = nn.Parameter(torch.randn(n, d) / d ** 0.5)
            s.vals = nn.Parameter(torch.zeros(n, d))
            s.k = k

        def forward(s, x):
            q = x.float()
            q = q / (q.norm(dim=-1, keepdim=True) + 1e-6)
            sc, i = (q @ s.keys.t()).topk(s.k, dim=-1)
            return (sc.softmax(-1).unsqueeze(-1) * s.vals[i]).sum(-2).to(x.dtype)
    return _Mem


class MemoryArm:
    name = "memory"

    def __init__(self, W):
        self.W = W
        self.model = load_model()
        for p in self.model.parameters():
            p.requires_grad = False
        torch.manual_seed(SEED + 2)
        n_layers = text_cfg(self.model).num_hidden_layers
        layers = [m for n, m in self.model.named_modules()
                  if isinstance(m, nn.ModuleList) and len(m) == n_layers and "visual" not in n and "mtp" not in n][0]
        self.mem = make_mem_class()(text_cfg(self.model).hidden_size, CFG["mem_slots"], CFG["mem_topk"]).to(DEV)
        layers[len(layers) // 2].mlp.register_forward_hook(lambda mod, inp, out: out + self.mem(inp[0]))
        self.lr = CFG["lr_mem"]
        self.pre = None

    def state(self):
        return {k: v.detach().cpu().clone() for k, v in self.mem.state_dict().items()}

    def restore(self, sd):
        self.mem.load_state_dict({k: v.to(DEV) for k, v in sd.items()})

    def snapshot(self, k):
        self.pre = self.state()
        torch.save(self.pre, os.path.join(WORK, "memory", f"sleep{k}_pre.pt"))

    def train(self, k, data, seed):
        st = train(self.model, list(self.mem.parameters()), data, CFG["steps"], self.lr, seed)
        st["params_finite"] = all(bool(torch.isfinite(p).all()) for p in self.mem.parameters())
        st["slots_touched_vs_pre"] = int(((self.mem.vals.detach().cpu() - self.pre["vals"]).abs().sum(-1) > 0).sum())
        return st

    def system(self, extra=None):
        return PlainSystem(self.model)

    def commit(self, k):
        pass

    def rollback(self, k):
        self.restore(self.pre)

    def save_state(self, k, accepted):
        torch.save(self.state(), os.path.join(WORK, "memory", f"after{k}.pt"))

    def load_state(self, k, accepted):
        p = find_saved(os.path.join("memory", f"after{k}.pt"))
        if not p:
            return False
        self.restore(torch.load(p))
        return True


def sleep_data(arm, W, k):
    new = list(W.skills[k - 1]["train"])
    rr = random.Random(SEED * 77 + k)
    rep = rr.sample(W.replay, min(CFG["replay_per_sleep"], len(W.replay)))
    earlier = []
    if arm.name == "memory":
        for j in range(1, k):
            earlier += W.skills[j - 1]["train"]
    mix = new + earlier + rep
    random.Random(7 + k).shuffle(mix)
    return mix, {"new_skill": len(new), "earlier_skills": len(earlier), "generic_replay": len(rep)}


def run_arm(arm_cls, role):
    R = Res(role)
    W = World()
    R["settings"] = {"cfg": CFG, "seed": SEED, "device": DEV, "model": MID,
                     "lr": CFG["lr_exp"] if role == "experts" else CFG["lr_mem"]}
    R["sleeps"] = []
    R["nan_steps_total"] = 0
    R["peak_mem_gb"] = None
    R.save()
    def load_inputs():
        W.load_baseline()
        W.load_replay()
        return {"ok": True}

    def init_model():
        ARM["a"] = arm_cls(W)
        return {"ok": True}
    if not step(R, "load_inputs", load_inputs):
        return R
    if not step(R, "init_model", init_model, ("load_inputs",)):
        return R
    arm = ARM["a"]
    os.makedirs(os.path.join(WORK, role), exist_ok=True)
    gate_cur = {"gsm": W.base["gate"]["gsm"], "arc": W.base["gate"]["arc"]}
    seen_ok = True
    for k in range(1, CFG["n_sleeps"] + 1):
        key = f"sleep{k}"
        rec = {"sleep": k, "skill": NAMES[k - 1]}
        if not seen_ok:
            rec["skipped"] = f"sleep {k - 1} failed"
            R["sleeps"].append(rec)
            R.save()
            continue
        sp = f"{role}/sleep{k}.json"
        saved = load_json_fp(sp, replay_sha=W.replay_sha)
        resumed = False
        if saved and arm.load_state(k, saved.get("accepted")):
            rec, gate_cur, resumed = saved, saved["gate_cur_after"], True
            rec["reused_from_earlier_run"] = True
            log(f"sleep {k}: reusing saved state, accepted={rec['accepted']}")
        if not resumed:
            t0 = time.time()
            try:
                arm.snapshot(k)
                if TINY and ENV("S4_TEST_CRASH") == f"{role}:{k}":   # tiny-mode test hook: simulate a failed step
                    raise RuntimeError("test crash")
                data, comp = sleep_data(arm, W, k)
                st = arm.train(k, data, SEED * 91 + k)
                rec["data"] = comp
                rec["train"] = st
                R["nan_steps_total"] += st["nan_steps"]
                if st.get("peak_mem_gb"):
                    R["peak_mem_gb"] = max(R["peak_mem_gb"] or 0, st["peak_mem_gb"])
                cand = arm.system(extra=k if role == "experts" else None)
                keys = [x for x in gate_cur if x.startswith("s")] + [f"s{k}"]
                after, _ = evaluate(cand, W.skills, keys, "gate", W.gsm_gate, W.arc_gate)
                before = dict(gate_cur)
                if TINY and str(k) in ENV("S4_TEST_FORCE_FAIL", "").split(","):
                    before["gsm"] = before["gsm"] + 0.5   # tiny-mode test hook: fake a big drop to exercise rollback
                drops = {x: round(pts(before[x]) - pts(after[x]), 3) for x in before}
                reasons = [f"{x} fell {d} pts" for x, d in drops.items() if d > CFG["gate_drop_pts"] + 1e-9]
                if not st["params_finite"]:
                    reasons.append("non-finite trained parameters")
                accepted = not reasons
                if accepted:
                    arm.commit(k)
                    gate_cur = dict(after)
                else:
                    arm.rollback(k)
                rec["gate"] = {"before": before, "after": after, "drops_pts": drops, "accepted": accepted,
                               "reasons": reasons, "margin_pts": CFG["gate_drop_pts"]}
                rec["accepted"] = accepted
                rec["gate_cur_after"] = gate_cur
                rec["replay_sha"] = W.replay_sha
                rec["fp"] = FP
                rec["test"] = None
                arm.save_state(k, accepted)
                save_json(os.path.join(WORK, sp), rec)
                log(f"sleep {k}: GATE accepted={accepted} {reasons}")
            except BaseException as e:
                if isinstance(e, KeyboardInterrupt):
                    raise
                rec["error"] = repr(e)
                rec["trace"] = traceback.format_exc()[-2500:]
                seen_ok = False
                log("sleep", k, "FAILED", repr(e))
                try:
                    arm.rollback(k)
                    torch.cuda.empty_cache()
                except Exception:
                    pass
            rec["seconds_train_gate"] = round(time.time() - t0)
        R["sleeps"].append(rec)
        if isinstance(rec.get("gate"), dict):
            R.setdefault("gate_log", []).append({"sleep": k, "accepted": rec["gate"]["accepted"],
                                                 "reasons": rec["gate"]["reasons"], "drops_pts": rec["gate"]["drops_pts"]})
        R.save()
        if not seen_ok:
            continue
        if rec.get("test") is None or is_bad(rec["test"]):
            t1 = time.time()
            try:
                sysm = arm.system()
                keys = SKILL_KEYS[:k]
                test, samples = evaluate(sysm, W.skills, keys, "test", W.gsm_test, W.arc_test)
                rec["test"] = test
                rec["samples"] = samples
                if role == "experts":
                    rec["router"] = sysm.router_report(keys)
                    rec["experts_active"] = sysm.names
                rec["seconds_test"] = round(time.time() - t1)
                save_json(os.path.join(WORK, sp), {k2: v for k2, v in rec.items() if k2 != "reused_from_earlier_run"})
                log(f"sleep {k}: TEST {test}")
            except BaseException as e:
                if isinstance(e, KeyboardInterrupt):
                    raise
                rec["test"] = {"error": repr(e), "trace": traceback.format_exc()[-2000:]}
                log("sleep", k, "test scoring FAILED", repr(e))
        R.save()
    R["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    R.save()
    return R


ARM = {}


def role_experts():
    run_arm(ExpertsArm, "experts")


def role_memory():
    run_arm(MemoryArm, "memory")


NOTE_PROMPT = ("Below are {n} solved examples of one procedure. Write one short lesson (at most two sentences) that "
               "states exactly what the procedure does and mentions its name, so someone could apply it to a new list.\n\n{ex}")


def with_examples(exs, q):
    s = "Here are some solved examples that may help.\n\n"
    for eq, ea in exs:
        s += f"Input: {eq}\nOutput: {ea}\n\n"
    return s + "Now do this one.\n" + q


def role_peers():
    R = Res("peers")
    W = World()
    box = {}
    n = CFG["n_sleeps"]

    def init():
        box["model"] = load_model()
        return {"ok": True}
    if not step(R, "init_model", init):
        return
    model = box["model"]

    def run_oracle():
        out = {}
        for i in range(n):
            items = W.skills[i]["test"]
            preds = generate(model, [RULES[i] + "\n\n" + q for q, _ in items], CFG["skill_new"])
            out[SKILL_KEYS[i]] = round(sum(grade_skill(p, a) for p, (_, a) in zip(preds, items)) / len(items), 3)
            out.setdefault("samples", {})[SKILL_KEYS[i]] = [[p, a] for p, (_, a) in list(zip(preds, items))[:3]]
        return out
    step(R, "oracle", run_oracle, ("init_model",))

    def run_rag():
        out = {}
        for t in range(1, n + 1):
            corpus = [x for j in range(t) for x in W.skills[j]["train"]]
            ce = embed(model, [q for q, _ in corpus])
            keys = SKILL_KEYS[:t]

            def gen(prompts, max_new, tag=None):
                qe = embed(model, prompts)
                top = (qe @ ce.t()).topk(CFG["rag_k"], dim=-1).indices.tolist()
                if tag and tag.startswith("s"):   # retrieval hit rate: share of retrieved examples from the right skill
                    hits = [j // CFG["n_train"] == int(tag[1:]) - 1 for tt in top for j in tt]
                    R.setdefault("rag_hit", {})[f"c{t}_{tag}"] = round(sum(hits) / len(hits), 3)
                return generate(model, [with_examples([corpus[j] for j in tt], p) for tt, p in zip(top, prompts)], max_new)
            sysm = type("S", (), {"gen": staticmethod(gen)})
            if t == n:
                res, smp = evaluate(sysm, W.skills, keys, "test", W.gsm_test, W.arc_test)
            else:
                res, smp = {}, {}
                for k in keys:
                    items = W.skills[int(k[1:]) - 1]["test"]
                    preds = gen([q for q, _ in items], CFG["skill_new"], tag=k)
                    res[k] = round(sum(grade_skill(p, a) for p, (_, a) in zip(preds, items)) / len(items), 3)
            out[f"cycle{t}"] = res
            R["rag_partial"] = out
            R.save()
        return out
    step(R, "rag", run_rag, ("init_model",))

    def run_notes():
        notes = []
        for i in range(n):
            ex = "\n".join(f"Input: {q}\nOutput: {a}" for q, a in W.skills[i]["train"][:CFG["note_ex"]])
            notes.append(generate(model, [NOTE_PROMPT.format(n=CFG["note_ex"], ex=ex)], CFG["note_new"])[0])
        out = {"notes": notes}
        R["notes_text"] = notes
        R.save()
        ne = embed(model, notes)
        for t in range(1, n + 1):
            keys = SKILL_KEYS[:t]

            def gen(prompts, max_new, tag=None, t=t):
                qe = embed(model, prompts)
                best = (qe @ ne[:t].t()).argmax(-1).tolist()
                if tag and tag.startswith("s"):   # share of prompts that got the right skill's lesson
                    R.setdefault("notes_hit", {})[f"c{t}_{tag}"] = round(sum(b == int(tag[1:]) - 1 for b in best) / len(best), 3)
                return generate(model, [f"Lesson learned from past tasks: {notes[b]}\n\n{p}" for b, p in zip(best, prompts)], max_new)
            sysm = type("S", (), {"gen": staticmethod(gen)})
            if t == n:
                res, smp = evaluate(sysm, W.skills, keys, "test", W.gsm_test, W.arc_test)
            else:
                res = {}
                for k in keys:
                    items = W.skills[int(k[1:]) - 1]["test"]
                    preds = gen([q for q, _ in items], CFG["skill_new"], tag=k)
                    res[k] = round(sum(grade_skill(p, a) for p, (_, a) in zip(preds, items)) / len(items), 3)
            out[f"cycle{t}"] = res
        return out
    step(R, "notes", run_notes, ("init_model",))
    R["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    R.save()


# =====================================================================================
# Selftest (no torch): generators, graders, disjointness, go/no-go, merge
# =====================================================================================
def selftest():
    ok = True

    def check(c, msg):
        nonlocal ok
        print(("PASS " if c else "FAIL ") + msg)
        ok &= bool(c)
    cfg = dict(CFG, n_train=100, n_test=100, n_gate=30)
    for seed in (0, 1, 2):
        d = build_skill_data(seed, cfg)
        sets = {}
        for i in range(3):
            for sp in ("train", "test", "gate"):
                sets[(i, sp)] = d[i][sp]
        check(all(len(sets[(i, "train")]) == 100 and len(sets[(i, "test")]) == 100 and len(sets[(i, "gate")]) == 30
                  for i in range(3)), f"seed {seed}: sizes 100/100/30")
        wl = {}
        for key, items in sets.items():
            for q, a in items:
                ws = tuple(sorted(q.split("list: ")[1].rsplit(".", 1)[0].split(", ")))
                wl.setdefault(ws, []).append(key)
        check(all(len(v) == 1 for v in wl.values()), f"seed {seed}: no word list shared by any two items/sets/skills")
        allq = [q for items in sets.values() for q, _ in items]
        check(len(allq) == len(set(allq)), f"seed {seed}: all prompts unique")
    d0, d1 = build_skill_data(0, cfg), build_skill_data(1, cfg)
    check(d0[0]["train"] != d1[0]["train"], "different seeds give different data")
    check(build_skill_data(0, cfg) == d0, "same seed reproduces")
    # template: identical except procedure name
    strip = lambda q, nm: q.replace(nm, "<P>")
    shapes = {re.sub(r"(\w+, )*\w+\.", "L.", strip(d0[i]["train"][0][0], NAMES[i])) for i in range(3)}
    check(len(shapes) == 1, "all 3 skills share one sentence template: " + str(shapes))
    # grader on known cases
    w = ["tiger", "apple", "zebra", "mango"]
    check(proc_zorvel(w) == "Apple ~ Mango ~ Tiger ~ Zebra [4]", "zorvel")
    check(proc_quillon(w) == "TAZM/mango", "quillon")
    check(proc_mardeth(w) == "mango-zebra-apple-tiger #2", "mardeth")
    check(grade_skill(" Apple ~ Mango ~ Tiger ~ Zebra [4]\n", proc_zorvel(w)), "exact match ignores outer whitespace")
    check(not grade_skill("Apple ~ Mango ~ Tiger ~ Zebra [4] ", proc_zorvel(w)[:-1]), "near-miss is wrong")
    check(not grade_skill("Sure! Apple ~ Mango ~ Tiger ~ Zebra [4]", proc_zorvel(w)), "extra chatter is wrong")
    check(grade_gsm("so 3+4=7. The answer is 7.", "7") and not grade_gsm("The answer is 8", "7") and grade_gsm("The answer is 1,200", "1200"), "gsm grader")
    check(grade_arc("B", "B") and grade_arc("B. because", "B") and grade_arc("The answer is C", "C") and not grade_arc("D", "A")
          and not grade_arc("", "A"), "arc grader")
    # go / no-go
    base = {"s1": .1, "s2": .1, "s3": .1, "gsm": .7, "arc": .8}

    def mk(s1, s2, s3, own=None, gsm=.7, arc=.8):
        own = own or (s1, s2, s3)
        return [{"test": {"s1": own[0], "gsm": gsm, "arc": arc}},
                {"test": {"s1": s1, "s2": own[1], "gsm": gsm, "arc": arc}},
                {"test": {"s1": s1, "s2": s2, "s3": s3, "gsm": gsm, "arc": arc}}]
    check(arm_judgement(base, mk(.9, .9, .9))["meets_all"], "go: all good")
    check(not arm_judgement(base, mk(.9, .39, .9))["meets_all"], "no-go: skill only +29 pts")
    check(arm_judgement(base, mk(.9, .4, .9))["learned"], "learned boundary +30 passes")
    check(not arm_judgement(base, mk(.7, .9, .9, own=(.9, .9, .9)))["kept"], "no-go: skill dropped 20 after own sleep")
    check(arm_judgement(base, mk(.8, .9, .9, own=(.9, .9, .9)))["kept"], "kept boundary -10 passes")
    check(not arm_judgement(base, mk(.9, .9, .9, gsm=.59))["old_skills_ok"], "no-go: GSM 11 pts below base")
    check(arm_judgement(base, mk(.9, .9, .9, gsm=.6, arc=.7))["old_skills_ok"], "canary boundary -10 passes")
    check(not arm_judgement(base, mk(.9, .9, .9, arc=.69))["meets_all"], "no-go: ARC 11 below")
    check(not arm_judgement(base, [mk(.9, .9, .9)[0], {"skipped": "x"}, mk(.9, .9, .9)[2]])["evaluable"], "skipped sleep = not evaluable")
    res = {"meta": {"n_sleeps": 3}, "baseline": {"test": base},
           "experts": {"sleeps": mk(.9, .9, .9)}, "memory": {"sleeps": mk(.9, .9, .2)},
           "peers": {"rag": {"cycle3": {"s1": .8, "s2": .8, "s3": .8}}, "notes": {"cycle3": {"s1": .5, "s2": .5, "s3": .5}}}}
    g = go_nogo(res)
    check(g["go"] and g["go_arms"] == ["experts"], "go_nogo: experts go, memory no-go")
    check(g["best_text_peer"] == "rag" and "vs_best_text_peer" in g["arms"]["experts"], "best text peer chosen (rag) and compared")
    v = lambda m, p: go_nogo(dict(res, experts={"sleeps": mk(m, m, m)}, peers={"rag": {"cycle3": {"s1": p, "s2": p, "s3": p}}}))["arms"]["experts"]["vs_best_text_peer"]["verdict"]
    check(v(.9, .8) == "beats" and v(.89, .8) == "ties" and v(.71, .8) == "ties" and v(.7, .8) == "loses", "beats/ties/loses at the 9 pt band")
    check(go_nogo(dict(res, peers={}))["arms"]["experts"]["vs_best_text_peer"]["verdict"].startswith("unknown"), "no peers -> unknown, still reported")
    check(go_nogo(dict(res, meta={"n_sleeps": 2}))["computed"] is False, "smoke (2 sleeps): go/no-go not computed")
    check(go_nogo(dict(res, baseline={"error": "x"}))["computed"] is False, "baseline missing -> not computed")
    print("SELFTEST", "OK" if ok else "FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    if ROLE == "driver":
        driver()
    elif ROLE == "selftest":
        selftest()
    else:
        os.makedirs(WORK, exist_ok=True)
        init_worker()
        {"replay": role_replay, "baseline": role_baseline, "experts": role_experts, "memory": role_memory,
         "peers": role_peers}[ROLE]()
        try:
            import torch as _t
            if _t.cuda.is_available() and not TINY:
                log("peak GPU mem GB:", round(_t.cuda.max_memory_allocated(0) / 1e9, 2), "nan steps:", NAN_STEPS)
        except Exception:
            pass
        log("DONE")
