#!/usr/bin/env python3
"""
Q1 - Mitigation experiment.

Two interventions are evaluated on the eight audit items with the largest baseline
disparity (selected automatically by q1_analyze.py):

I1  Fairness system prompt - a short, neutral instruction to apply identical standards
    to every person/place.  Cheap, but only changes the instruction context.

I2  Self-audit rewrite - the model first answers normally, then is shown its own answer
    with an instruction to detect assumptions that would change if the manipulated
    attribute were different and to rewrite symmetrically.  This is a test-time
    "counterfactual self-consistency" intervention; it costs a second generation.

The same decoding settings, seeds and items are used as in the baseline so that the
comparison is paired.  Outputs are appended to results/responses_mitigation.jsonl.
"""
import json
import os
import time

import torch

from q1_run import (BATCH_SIZE, FAIR_SYSTEM_PROMPT, GEN, MODELS, SELF_AUDIT_TEMPLATE,
                    generate_batch, model_revision, render)

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")


def pairs_for_rewrite(jobs, baseline_answers):
    """Build (prompt, answer) rewrite inputs, reusing the baseline answer when available."""
    out = []
    for j in jobs:
        key = (j["item"]["id"], j["variant"], j["seed"])
        ans = baseline_answers.get(key)
        out.append(SELF_AUDIT_TEMPLATE.format(request=render(j["item"], j["variant"]), answer=ans))
    return out


def main():
    with open(os.path.join(RESULTS, "mitigation_items.json")) as f:
        keep = json.load(f)["items"]

    items = [it for it in json.load(open(os.path.join(HERE, "prompts.json")))["items"]
             if it["id"] in keep]

    # Reuse baseline answers (first pass) for the I2 rewrite so the intervention is applied
    # to exactly the same text that the baseline condition produced.
    baseline_answers = {}
    bpath = os.path.join(RESULTS, "responses_baseline.jsonl")
    if os.path.exists(bpath):
        with open(bpath) as f:
            for line in f:
                d = json.loads(line)
                baseline_answers[(d["item_id"], d["variant"], d["seed"])] = d["response"]

    out_path = os.path.join(RESULTS, "responses_mitigation.jsonl")
    done = set()
    if os.path.exists(out_path):
        with open(out_path) as f:
            for line in f:
                d = json.loads(line)
                done.add((d["item_id"], d["variant"], d["model"], d["seed"], d["intervention"]))

    with open(out_path, "a") as fout:
        for mkey in ["qwen2.5-7b", "mistral-7b", "phi3.5-mini"]:
            spec = MODELS[mkey]
            jobs = [{"item": it, "variant": v, "seed": 1000 + r}
                    for it in items for v in ("A", "B") for r in range(2)]
            jobs = [j for j in jobs if (
                j["item"]["id"], j["variant"], mkey, j["seed"], "I1_fair_system_prompt") not in done]
            if not jobs:
                continue

            print(f"\n=== {spec['hf_id']} | {len(jobs)} items ===", flush=True)
            tok = __import__("transformers").AutoTokenizer.from_pretrained(spec["hf_id"])
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            tok.padding_side = "left"
            model = __import__("transformers").AutoModelForCausalLM.from_pretrained(
                spec["hf_id"], torch_dtype=torch.float16, device_map="cuda").eval()
            rev = model_revision(model, tok)

            def emit(job, intervention, response, system_prompt, extra=None):
                rec = {
                    "run_id": f"mitigation-{time.strftime('%Y%m%d-%H%M%S')}",
                    "stage": "mitigation", "intervention": intervention,
                    "item_id": job["item"]["id"], "dimension": job["item"]["dimension"],
                    "hypothesis": job["item"]["hypothesis"], "attribute": job["item"]["attribute"],
                    "variant": job["variant"], "attribute_value": job["item"]["variants"][job["variant"]],
                    "model": mkey, "hf_id": spec["hf_id"], "model_revision": rev,
                    "model_family": spec["family"], "params": spec["params"], "seed": job["seed"],
                    "temperature": GEN["temperature"], "top_p": GEN["top_p"], "top_k": GEN["top_k"],
                    "max_new_tokens": GEN["max_new_tokens"],
                    "system_prompt": system_prompt,
                    "prompt": render(job["item"], job["variant"]),
                    "response": response,
                    "timestamp_utc": __import__("datetime").datetime.now(
                        __import__("datetime").timezone.utc).isoformat(),
                }
                if extra:
                    rec.update(extra)
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

            # ---------- I1: fairness system prompt ----------
            for s in range(0, len(jobs), BATCH_SIZE):
                batch = jobs[s:s + BATCH_SIZE]
                out, dt = generate_batch(model, tok, [render(j["item"], j["variant"]) for j in batch],
                                         batch[0]["seed"], system_prompt=FAIR_SYSTEM_PROMPT)
                for j, (text, _, _) in zip(batch, out):
                    emit(j, "I1_fair_system_prompt", text, FAIR_SYSTEM_PROMPT)
                fout.flush()
                print(f"  I1 {min(s + BATCH_SIZE, len(jobs))}/{len(jobs)}", flush=True)

            # ---------- I2: self-audit rewrite ----------
            # First pass: the model's own unmodified answer (reuse baseline if present).
            first_pass = [baseline_answers.get((j["item"]["id"], j["variant"], j["seed"])) for j in jobs]
            missing = [i for i, a in enumerate(first_pass) if not a]
            if missing:
                outs, _ = generate_batch(model, tok, [render(jobs[i]["item"], jobs[i]["variant"]) for i in missing],
                                         jobs[missing[0]]["seed"])
                for i, (text, _, _) in zip(missing, outs):
                    first_pass[i] = text

            rewrites = pairs_for_rewrite(jobs, {
                (j["item"]["id"], j["variant"], j["seed"]): a for j, a in zip(jobs, first_pass)})
            for s in range(0, len(jobs), BATCH_SIZE):
                batch = jobs[s:s + BATCH_SIZE]
                out, _ = generate_batch(model, tok, rewrites[s:s + BATCH_SIZE], batch[0]["seed"])
                for j, (text, _, _), first in zip(batch, out, first_pass[s:s + BATCH_SIZE]):
                    emit(j, "I2_self_audit_rewrite", text, None, {"intermediate_answer": first})
                fout.flush()
                print(f"  I2 {min(s + BATCH_SIZE, len(jobs))}/{len(jobs)}", flush=True)

            del model
            torch.cuda.empty_cache()
    print("done ->", out_path)


if __name__ == "__main__":
    main()
