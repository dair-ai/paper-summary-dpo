"""Automatic style and support checks. Counts how many summaries in each folder have each problem.

By default it scores our published outputs in results/ on the 20 test papers, which reproduces the results table:
    python training/evaluate.py
To score your own run against ours:
    python training/evaluate.py --trained runs/my-run/samples
Needs data/text/, which scripts/build_inputs.py writes. "numbers not in input" compares against that text.
"""
import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NUM = re.compile(r"(?<![\w.])\d[\d,]*\.?\d*")
nums = lambda t: {m.replace(",", "").rstrip(".") for m in NUM.findall(t) if len(m.replace(",", "").rstrip(".")) >= 2}
take = lambda o: o.split("Key points", 1)[0]
points = lambda o: o.split("Key points", 1)[-1]
CHECKS = {
    "em or en dash": lambda o, s: bool(re.search("[—–]", o)),
    "Section/Figure/Table ref": lambda o, s: bool(re.search(r"\b(Section|Figure|Fig\.|Table|Definition|Appendix)\s*\d", o)),
    "et al outside credit": lambda o, s: o.count("et al") > 1 or "et al" in points(o),
    "missing Why it matters": lambda o, s: "Why it matters" not in o,
    "bullet not '**Label.** text'": lambda o, s: any(not re.match(r"- \*\*[^*]+\.\*\* \S", l) for l in points(o).splitlines()
                                                      if l.startswith(("- ", "* ", "• "))),
    "no 'First author et al.' credit": lambda o, s: not re.search(r"\bet al\.|\bThe authors\b", take(o)),
    "numbers not in input": lambda o, s: bool(nums(o) - nums(s)),
    "results in curator's take": lambda o, s: bool(re.search(r"\d+(\.\d+)?\s?%|\d+(\.\d+)?x\b", take(o))),
    "opens 'The paper tackles/addresses'": lambda o, s: bool(re.search(r"(?i)the paper (tackles|addresses|investigates)", o)),
}

ap = argparse.ArgumentParser()
ap.add_argument("--split", default="test")
ap.add_argument("--base", default="results/base")
ap.add_argument("--prompted", default="results/prompted")
ap.add_argument("--trained", default="results/sft-dpo-v4/samples")
ap.add_argument("--out", default=None, help="optional path for the JSON report")
args = ap.parse_args()

ids = [json.loads(l)["arxiv_id"] for l in (ROOT / "data/pairs" / f"{args.split}.jsonl").read_text().split("\n") if l.strip()]
models = {"base": args.base, "prompted": args.prompted, "trained": args.trained}
report = {}
for name, folder in models.items():
    d = ROOT / folder
    have = [i for i in ids if (d / f"{i}.md").exists()]
    if not have:
        continue
    r = {c: 0 for c in CHECKS} | {"words": 0.0, "papers": len(have)}
    for i in have:
        o = (d / f"{i}.md").read_text()
        src_f = ROOT / "data/text" / f"{i}.txt"
        if not src_f.exists():
            raise SystemExit("data/text/ is missing. Run python scripts/build_inputs.py first.")
        s = src_f.read_text()
        for c, f in CHECKS.items():
            r[c] += bool(f(o, s))
        r["words"] += len(o.split()) / len(have)
    report[name] = r

names = list(report)
print(f"{args.split} split: number of summaries with each problem")
print(f"{'check':38}" + "".join(f"{n:>10}" for n in names))
for c in list(CHECKS) + ["words", "papers"]:
    print(f"{c:38}" + "".join(f"{report[n][c]:>10.0f}" for n in names))
if args.out:
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
