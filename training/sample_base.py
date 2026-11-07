"""Sample the untrained base model, with no training and no style rules, on the test and dev prompts.

Same chat template, forced final channel, temperature 0 and 1,024-token cap as the trained model.
    python training/sample_base.py                  # writes runs/base/<arxiv_id>.md
    python training/sample_base.py --prompted       # adds the style rules to the prompt, writes runs/prompted/

--prompted answers "why not just put the rules in the prompt?". The only change is that the Content and Style
sections of style-spec.md are inserted before the paper text. Our results used it on the test split only.
The session stops on exit.
"""
import argparse
import json
from datetime import datetime, timezone

from sagemaker.train.training_session import ModelInput, SamplingParams, ServiceClient

from common import BASE_MODEL_ARN, REGION, ROOT, aws_settings, load_split, prompt_tokens, tokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--prompted", action="store_true", help="insert the style rules into the prompt")
ap.add_argument("--splits", default=None, help="comma-separated, default test,dev (test only with --prompted)")
ap.add_argument("--limit", type=int, default=0, help="only sample this many papers per split")
args = ap.parse_args()
splits = (args.splits or ("test" if args.prompted else "test,dev")).split(",")

MARK = "Paper text:"
spec = (ROOT / "style-spec.md").read_text()
RULES = "Also follow these house style rules exactly:\n\n" + spec[spec.index("## Content"):].strip() + "\n\n"

aws = aws_settings()
tok = tokenizer()
out = ROOT / "runs" / ("prompted" if args.prompted else "base")
out.mkdir(parents=True, exist_ok=True)
svc = ServiceClient(region=REGION, role_arn=aws["ROLE_ARN"], model_package_group_arn=aws["MODEL_PACKAGE_GROUP_ARN"],
                    s3_output_path=f"{aws['OUTPUT_S3_PATH']}paper-summary-dpo/{out.name}/")
recs = []
with svc.create_sampling_client(base_model=BASE_MODEL_ARN) as sc:
    for split in splits:
        for r in load_split(split)[:args.limit or None]:
            if (out / f"{r['arxiv_id']}.md").exists():
                continue
            prompt = r["prompt"]
            if args.prompted:
                assert prompt.count(MARK) == 1, r["arxiv_id"]
                prompt = prompt.replace(MARK, RULES + MARK)
            p = prompt_tokens(tok, prompt)
            seq = sc.sample(ModelInput.from_ints(p), num_samples=1,
                            sampling_params=SamplingParams(max_tokens=1024, temperature=0.0)).result().sequences[0]
            (out / f"{r['arxiv_id']}.md").write_text(tok.decode(seq.tokens).removesuffix("<|return|>").strip() + "\n")
            recs.append({"split": split, "id": r["arxiv_id"], "input_tokens": len(p), "output_tokens": len(seq.tokens),
                         "stop": str(seq.stop_reason), "time_utc": datetime.now(timezone.utc).isoformat()})
            (out / "records.json").write_text(json.dumps(recs, indent=1) + "\n")
            print(split, r["arxiv_id"], len(seq.tokens), "tokens", flush=True)
print(len(recs), "samples; input", sum(x["input_tokens"] for x in recs), "output", sum(x["output_tokens"] for x in recs))
