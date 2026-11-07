"""Shared settings for every script that talks to Amazon SageMaker AI Training Sessions.

Account-specific values come from environment variables or a .env file in the repo root (see .env.example).
"""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv():
    f = ROOT / ".env"
    if not f.exists():
        return
    for line in f.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.removeprefix("export ").split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv()

REGION = os.environ.get("AWS_REGION", "us-east-1")
BASE_MODEL_ARN = os.environ.get(
    "BASE_MODEL_ARN",
    f"arn:aws:sagemaker:{REGION}:aws:hub-content/SageMakerPublicHub/Model/openai-reasoning-gpt-oss-20b/3.43.2")
TOKENIZER = "openai/gpt-oss-20b"
TOKENIZER_REVISION = "6cee5e81ee83917806bbde320786a8fb61efebee"
COOKBOOK_COMMIT = "7f2d3e004d5c3e7a96ae59eab0ae2b070abba94a"
# List prices per million tokens for GPT-OSS-20B on Training Sessions in us-east-1, used by the cost guard.
RATE = {"prefill": 0.12, "train": 0.36, "sample": 0.30}


def aws_settings():
    """Return the role, model package group and S3 output path. Fails early with a clear message if one is missing."""
    names = ("ROLE_ARN", "MODEL_PACKAGE_GROUP_ARN", "OUTPUT_S3_PATH")
    missing = [n for n in names if not os.environ.get(n)]
    if missing:
        raise SystemExit(f"Missing {', '.join(missing)}. Copy .env.example to .env and fill it in.")
    return {n: os.environ[n].rstrip("/") + ("/" if n == "OUTPUT_S3_PATH" else "") for n in names}


def tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(TOKENIZER, revision=TOKENIZER_REVISION)


def prompt_tokens(tok, prompt):
    """GPT-OSS chat template with low reasoning effort, then the final channel is forced so the model answers directly."""
    final = tok.encode("<|channel|>final<|message|>", add_special_tokens=False)
    return list(tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=True, add_generation_prompt=True,
                                        reasoning_effort="low", return_dict=False)) + final


def load_split(split):
    """Rows from data/dpo/<split>.jsonl, which scripts/build_inputs.py writes."""
    f = ROOT / "data/dpo" / f"{split}.jsonl"
    if not f.exists():
        raise SystemExit(f"{f} not found. Run python scripts/build_inputs.py first.")
    return [json.loads(l) for l in f.read_text().split("\n") if l.strip()]
