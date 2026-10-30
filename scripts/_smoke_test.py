"""Smoke test on bionlp: load each cached model, generate, measure speed, and report versions."""
import json, time, platform, torch, transformers, os
from transformers import AutoModelForCausalLM, AutoTokenizer

MODELS = {
    "Qwen2.5-7B-Instruct": "Qwen/Qwen2.5-7B-Instruct",
    "Mistral-7B-Instruct-v0.3": "mistralai/Mistral-7B-Instruct-v0.3",
    "Phi-3.5-mini-instruct": "microsoft/Phi-3.5-mini-instruct",
    "Qwen2.5-1.5B-Instruct": "Qwen/Qwen2.5-1.5B-Instruct",
}
out = {"torch": torch.__version__, "transformers": transformers.__version__,
       "python": platform.python_version(), "gpu": torch.cuda.get_device_name(0),
       "models": {}}
for name, path in MODELS.items():
    try:
        t0 = time.time()
        tok = AutoTokenizer.from_pretrained(path)
        model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float16, device_map="cuda")
        model.eval()
        msgs = [{"role": "user", "content": "In two sentences, name two software companies in Bhopal."}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids = tok(text, return_tensors="pt").to("cuda")
        t1 = time.time()
        with torch.no_grad():
            gen = model.generate(**ids, max_new_tokens=80, do_sample=False)
        dt = time.time() - t1
        resp = tok.decode(gen[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
        ntok = gen.shape[1] - ids["input_ids"].shape[1]
        out["models"][name] = {"load_s": round(t1 - t0, 1), "tok_per_s": round(ntok / dt, 2),
                               "resp": resp[:300], "ok": True}
        print(name, out["models"][name], flush=True)
        del model; torch.cuda.empty_cache()
    except Exception as e:
        out["models"][name] = {"ok": False, "error": repr(e)[:400]}
        print(name, "FAILED", repr(e)[:400], flush=True)

with open(os.path.dirname(os.path.abspath(__file__)) + "/smoke_test.json", "w") as f:
    json.dump(out, f, indent=2)
print(json.dumps({k: (v.get("tok_per_s") if isinstance(v, dict) else v) for k, v in out["models"].items()}, indent=1))
