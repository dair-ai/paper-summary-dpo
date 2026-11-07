"""SFT then DPO training of GPT-OSS-20B (LoRA) on Amazon SageMaker AI Training Sessions, then sampling of the trained model.

The defaults are the settings of our final model (run 4): 2 epochs of SFT on the chosen summaries, then 2 epochs of DPO.
    python training/train_dpo.py --run my-run

Built on the AWS Training Sessions cookbook (commit 7f2d3e0) for session setup, datums and log-prob parsing.
DPO math is in dpo.py. Every step, the reference pass, token counts and the cost estimate are written to runs/<run>/.
The session is always stopped on exit.

Prompt layout: GPT-OSS chat template (reasoning effort low) + "<|channel|>final<|message|>".
Completion: summary text + "<|return|>". Forcing the final channel keeps training and sampling identical.
"""
import argparse, json, os, random, time
from datetime import datetime, timezone
from pathlib import Path

from sagemaker.train.training_session import AdamParams, LossFunction, ModelInput, SamplingParams
from training_sessions_cookbook.service import make_datum
from training_sessions_cookbook.service.session import open_session
from training_sessions_cookbook.preference.train import _extract_logprobs

from common import (BASE_MODEL_ARN, COOKBOOK_COMMIT, RATE, REGION, ROOT, TOKENIZER_REVISION as REV, aws_settings,
                    load_split, prompt_tokens as _prompt_tokens, tokenizer)
from dpo import dpo_term

# The cost guard stops the run if the projected cost, times GUARD_FACTOR, would pass --cap_usd.
GUARD_FACTOR = 3.0

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--beta", type=float, default=0.2)
ap.add_argument("--lr", type=float, default=1e-5)
ap.add_argument("--sft_alpha", type=float, default=1.0, help="RPO-style NLL weight on the chosen side, per token")
ap.add_argument("--epochs", type=int, default=2)
ap.add_argument("--batch_pairs", type=int, default=8)
ap.add_argument("--lora_rank", type=int, default=64)
ap.add_argument("--lora_alpha", type=int, default=128)
ap.add_argument("--seed", type=int, default=20261102)
ap.add_argument("--cap_usd", type=float, default=5.0)
ap.add_argument("--spent_usd", type=float, default=0.0, help="guard estimate already used by earlier runs")
ap.add_argument("--sample_splits", default="test,dev")
ap.add_argument("--sft_epochs", type=int, default=2, help="supervised epochs on the chosen edits before DPO")
ap.add_argument("--sft_lr", type=float, default=5e-5)
ap.add_argument("--limit", type=int, default=0, help="smoke test: use only this many train and dev pairs")
ap.add_argument("--max_steps", type=int, default=0)
ap.add_argument("--sample_limit", type=int, default=0)
args = ap.parse_args()

aws = aws_settings()
run_dir = ROOT / "runs" / args.run
run_dir.mkdir(parents=True, exist_ok=False)
(run_dir / "config.json").write_text(json.dumps({**vars(args), "base_model_arn": BASE_MODEL_ARN, "tokenizer_revision": REV,
    "cookbook_commit": COOKBOOK_COMMIT, "rates_per_million": RATE, "guard_factor": GUARD_FACTOR,
    "started_utc": datetime.now(timezone.utc).isoformat()}, indent=2) + "\n")
metrics_f = open(run_dir / "metrics.jsonl", "a")
def log(rec):
    rec["time_utc"] = datetime.now(timezone.utc).isoformat()
    metrics_f.write(json.dumps(rec) + "\n"); metrics_f.flush()
    print(json.dumps(rec), flush=True)

tok = tokenizer()
RETURN = tok.encode("<|return|>", add_special_tokens=False)
def prompt_tokens(prompt):
    return _prompt_tokens(tok, prompt)
def completion_tokens(text):
    return tok.encode(text.strip(), add_special_tokens=False) + RETURN
def slice_(p, c):
    full = p + c
    return full[:-1], full[1:], [0.0] * (len(p) - 1) + [1.0] * len(c)

def load(split):
    rows = load_split(split)
    out = []
    for r in rows:
        p = prompt_tokens(r["prompt"])
        out.append({"id": r["arxiv_id"], "p": p, "c": completion_tokens(r["chosen"]), "r": completion_tokens(r["rejected"])})
        assert len(p) + max(len(out[-1]["c"]), len(out[-1]["r"])) <= 4096, r["arxiv_id"]
    return out
train, dev = load("train"), load("dev")
if args.limit:
    train, dev = train[:args.limit], dev[:args.limit]

usage = {"prefill": 0, "train": 0, "sample": 0}
def est():
    return sum(usage[k] / 1e6 * RATE[k] for k in RATE)
def guard(extra_prefill=0, extra_train=0, extra_sample=0):
    worst = args.spent_usd + GUARD_FACTOR * (est() + extra_prefill / 1e6 * RATE["prefill"] + extra_train / 1e6 * RATE["train"]
                                             + extra_sample / 1e6 * RATE["sample"])
    if worst > args.cap_usd:
        raise SystemExit(f"cost guard: projected {worst:.2f} USD with factor {GUARD_FACTOR} passes cap {args.cap_usd}")

def seq_logps(client, pairs):
    """Summed completion log-probs for chosen and rejected of each pair, plus per-token logprobs for training."""
    datums, slices = [], []
    for x in pairs:
        for side in ("c", "r"):
            inp, tgt, mask = slice_(x["p"], x[side])
            datums.append(make_datum(inp, tgt, mask)); slices.append((inp, tgt, mask))
    n_tok = sum(len(s[0]) for s in slices)
    guard(extra_prefill=n_tok)
    res = client.forward(datums, loss_fn=LossFunction.CROSS_ENTROPY).result()
    usage["prefill"] += n_tok
    lps = _extract_logprobs(res, datums)
    sums = [sum(lp for lp, m in zip(l, s[2]) if m > 0) for l, s in zip(lps, slices)]
    return sums, lps, slices

svc = tc = None
session_name = f"dpo-{args.run}-{int(time.time())}"
try:
    svc, tc, svc_tok = open_session(session_name=session_name, base_model_arn=BASE_MODEL_ARN,
        model_package_group_arn=aws["MODEL_PACKAGE_GROUP_ARN"], role_arn=aws["ROLE_ARN"],
        output_s3_path=f"{aws['OUTPUT_S3_PATH']}paper-summary-dpo/{args.run}/",
        lora_rank=args.lora_rank, lora_alpha=args.lora_alpha, region=REGION)
    probe = "Curator's take <|channel|>final<|message|> 15.79% et al."
    assert svc_tok.encode(probe, add_special_tokens=False) == tok.encode(probe, add_special_tokens=False), "tokenizer mismatch"
    log({"event": "session_open", "session": tc.training_session_name, "train_pairs": len(train), "dev_pairs": len(dev)})

    # Optional SFT on the preferred edits first. DPO's reference is then the SFT model, computed after this phase.
    sft_rng, sft_step = random.Random(args.seed + 1), 0
    for epoch in range(args.sft_epochs):
        order = list(train); sft_rng.shuffle(order)
        for k in range(0, len(order), args.batch_pairs):
            batch = order[k:k + args.batch_pairs]
            datums, n_tok = [], 0
            for x in batch:
                inp, tgt, mask = slice_(x["p"], x["c"])
                datums.append(make_datum(inp, tgt, mask)); n_tok += len(inp)
            guard(extra_train=n_tok)
            fb = tc.forward_backward(datums, loss_fn=LossFunction.CROSS_ENTROPY)
            opt = tc.optim_step(AdamParams(learning_rate=args.sft_lr, beta1=0.9, beta2=0.95, eps=1e-8, weight_decay=0.0))
            fb_r, opt_r = fb.result(), opt.result()
            usage["train"] += n_tok; sft_step += 1
            m = {**(getattr(fb_r, "metrics", None) or {}), **(getattr(opt_r, "metrics", None) or {})}
            log({"event": "sft_step", "epoch": epoch, "step": sft_step, "service": {k2: m[k2] for k2 in m if "loss" in k2 or "grad_norm" in k2},
                 "usage": dict(usage), "est_usd": round(est(), 4)})

    # Reference log-probs from the current adapter (zero delta, or the SFT model when --sft_epochs > 0), computed once.
    ref = {}
    for split, rows in (("train", train), ("dev", dev)):
        for k in range(0, len(rows), args.batch_pairs):
            chunk = rows[k:k + args.batch_pairs]
            sums, _, _ = seq_logps(tc, chunk)
            for j, x in enumerate(chunk):
                ref[x["id"]] = (sums[2 * j], sums[2 * j + 1])
    (run_dir / "reference_logps.json").write_text(json.dumps(ref, indent=1) + "\n")
    log({"event": "reference_done", "pairs": len(ref), "usage": dict(usage), "est_usd": round(est(), 4)})

    def evaluate(rows, tag, step):
        losses, acc, margins = [], 0, []
        for k in range(0, len(rows), args.batch_pairs):
            chunk = rows[k:k + args.batch_pairs]
            sums, _, _ = seq_logps(tc, chunk)
            for j, x in enumerate(chunk):
                loss, _, _, rc, rr = dpo_term(sums[2 * j], sums[2 * j + 1], *ref[x["id"]], args.beta)
                losses.append(loss); acc += rc > rr; margins.append(rc - rr)
        log({"event": "eval", "split": tag, "step": step, "loss": sum(losses) / len(losses), "accuracy": acc / len(rows),
             "margin": sum(margins) / len(margins), "usage": dict(usage), "est_usd": round(est(), 4)})

    evaluate(dev, "dev", 0)
    rng, step = random.Random(args.seed), 0
    for epoch in range(args.epochs):
        order = list(train); rng.shuffle(order)
        for k in range(0, len(order), args.batch_pairs):
            batch = order[k:k + args.batch_pairs]
            sums, lps, slices = seq_logps(tc, batch)
            datums, losses, acc, margins, lc, lr_ = [], [], 0, [], [], []
            for j, x in enumerate(batch):
                loss, adv_c, adv_r, rc, rr = dpo_term(sums[2 * j], sums[2 * j + 1], *ref[x["id"]], args.beta)
                # RPO term: alpha * mean token NLL of the chosen completion adds alpha / len(chosen) per chosen token.
                adv_c += args.sft_alpha / len(x["c"])
                loss += args.sft_alpha * (-sums[2 * j] / len(x["c"]))
                for side, adv in ((0, adv_c), (1, adv_r)):
                    inp, tgt, mask = slices[2 * j + side]
                    datums.append(make_datum(inp, tgt, mask, logprobs=lps[2 * j + side], advantages=[adv * m for m in mask]))
                losses.append(loss); acc += rc > rr; margins.append(rc - rr); lc.append(sums[2 * j]); lr_.append(sums[2 * j + 1])
            n_tok = sum(len(s[0]) for s in slices)
            guard(extra_train=n_tok)
            fb = tc.forward_backward(datums, loss_fn=LossFunction.IMPORTANCE_SAMPLING)
            opt = tc.optim_step(AdamParams(learning_rate=args.lr, beta1=0.9, beta2=0.95, eps=1e-8, weight_decay=0.0))
            fb_r, opt_r = fb.result(), opt.result()
            usage["train"] += n_tok
            step += 1
            log({"event": "step", "epoch": epoch, "step": step, "loss": sum(losses) / len(losses), "accuracy": acc / len(batch),
                 "margin": sum(margins) / len(margins), "logp_chosen": sum(lc) / len(lc), "logp_rejected": sum(lr_) / len(lr_),
                 "service": {**(getattr(fb_r, "metrics", None) or {}), **(getattr(opt_r, "metrics", None) or {})},
                 "usage": dict(usage), "est_usd": round(est(), 4)})
            if args.max_steps and step >= args.max_steps:
                break
        evaluate(dev, "dev", step)
        if args.max_steps and step >= args.max_steps:
            break

    state = tc.save_state(f"paper-summary DPO {args.run} final").result()
    log({"event": "saved_state", "model_package_arn": getattr(state, "model_package_arn", None)})
    sc = tc.save_weights_and_get_sampling_client(f"paper-summary DPO {args.run} sampler")
    out_dir = run_dir / "samples"; out_dir.mkdir()
    for split in [s for s in args.sample_splits.split(",") if s]:
        rows = load_split(split)
        for r in rows[:args.sample_limit or None]:
            p = prompt_tokens(r["prompt"])
            guard(extra_prefill=len(p), extra_sample=1024)
            seq = sc.sample(ModelInput.from_ints(p), num_samples=1, sampling_params=SamplingParams(max_tokens=1024, temperature=0.0)).result().sequences[0]
            usage["prefill"] += len(p); usage["sample"] += len(seq.tokens)
            text = tok.decode(seq.tokens).removesuffix("<|return|>").strip()
            (out_dir / f"{r['arxiv_id']}.md").write_text(text + "\n")
            log({"event": "sample", "split": split, "id": r["arxiv_id"], "stop": str(seq.stop_reason), "tokens": len(seq.tokens)})
    log({"event": "done", "usage": dict(usage), "est_usd": round(est(), 4)})
finally:
    if tc is not None:
        try:
            tc.stop(); print("session stopped", flush=True)
        except Exception as e:
            print("stop failed", repr(e), flush=True)
    if svc is not None:
        print("running sessions:", svc.list_sessions(status_equals="Running"), flush=True)
