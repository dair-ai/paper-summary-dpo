"""Turn a finished training run into a one-page HTML results view.

    python training/show_results.py --run my-run
    open runs/my-run/results.html

The page scores your run's test summaries with the same checks as evaluate.py and compares them with our published
base and prompted outputs. It also shows how training went and one paper before and after.
Needs data/text/ from scripts/build_inputs.py.
"""
import argparse
import html
import json
import re
from datetime import datetime
from pathlib import Path

from checks import CHECKS

ROOT = Path(__file__).resolve().parent.parent
ROWS = [("Em or en dashes", "em or en dash"),
        ("Wrong bullet label format", "bullet not '**Label.** text'"),
        ('Opens with "The paper tackles"', "opens 'The paper tackles/addresses'"),
        ('Missing "First author et al." credit', "no 'First author et al.' credit"),
        ('Missing "Why it matters"', "missing Why it matters"),
        ("Results in the curator's take", "results in curator's take"),
        ("Numbers not in the input", "numbers not in input")]
HEADLINE = ["em or en dash", "bullet not '**Label.** text'", "no 'First author et al.' credit"]

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True, help="run name under runs/")
ap.add_argument("--example", default="2608.25593", help="test paper to show before and after")
ap.add_argument("--base", default="results/base")
ap.add_argument("--prompted", default="results/prompted")
args = ap.parse_args()

run_dir = ROOT / "runs" / args.run
samples = run_dir / "samples"
if not samples.exists():
    raise SystemExit(f"{samples} not found. The run has to finish so the model writes its summaries.")
ids = [json.loads(l)["arxiv_id"] for l in (ROOT / "data/pairs/test.jsonl").read_text().split("\n") if l.strip()]
ids = [i for i in ids if (samples / f"{i}.md").exists()]
if not ids:
    raise SystemExit("The run has no test summaries. Run it with the default --sample_splits.")


def score(folder):
    r = {c: 0 for c in CHECKS}
    for i in ids:
        src = ROOT / "data/text" / f"{i}.txt"
        if not src.exists():
            raise SystemExit("data/text/ is missing. Run python scripts/build_inputs.py first.")
        o, s = (folder / f"{i}.md").read_text(), src.read_text()
        for c, f in CHECKS.items():
            r[c] += bool(f(o, s))
    return r


models = {"Base": score(ROOT / args.base), "Prompted": score(ROOT / args.prompted), "Trained": score(samples)}

# Training curve and timing from metrics.jsonl
events = [json.loads(l) for l in (run_dir / "metrics.jsonl").read_text().split("\n") if l.strip()]
sft = [e["service"].get("actor/loss") for e in events if e["event"] == "sft_step" and "actor/loss" in e.get("service", {})]
dpo = [e["accuracy"] for e in events if e["event"] == "step"]
t0 = next(e["time_utc"] for e in events if e["event"] == "session_open")
t1 = next((e["time_utc"] for e in events if e["event"] == "done"), events[-1]["time_utc"])
minutes = (datetime.fromisoformat(t1) - datetime.fromisoformat(t0)).total_seconds() / 60
cost = next((e["est_usd"] for e in reversed(events) if "est_usd" in e), None)
config = json.loads((run_dir / "config.json").read_text())


def spark(values, lo, hi, label_lo, label_hi):
    if len(values) < 2:
        return '<div class="nochart">Not enough steps to plot</div>'
    w, h, p = 420, 130, 6
    pts = " ".join(f"{p + k * (w - 2 * p) / (len(values) - 1):.1f},{h - p - (min(max(v, lo), hi) - lo) / (hi - lo) * (h - 2 * p):.1f}"
                   for k, v in enumerate(values))
    return (f'<svg viewBox="0 0 {w} {h}"><line x1="{p}" y1="{h - p}" x2="{w - p}" y2="{h - p}" stroke="#ddd"/>'
            f'<line x1="{p}" y1="{p}" x2="{w - p}" y2="{p}" stroke="#eee"/>'
            f'<polyline points="{pts}" fill="none" stroke="#111" stroke-width="2.5" stroke-linejoin="round"/>'
            f'<text x="{w - p}" y="{p + 12}" text-anchor="end" font-size="11" fill="#888">{label_hi}</text>'
            f'<text x="{w - p}" y="{h - p - 4}" text-anchor="end" font-size="11" fill="#888">{label_lo}</text></svg>')


def render(md):
    out, in_list = [], False
    for line in md.strip().split("\n"):
        t = html.escape(line.strip())
        t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
        t = re.sub(r"[—–]", '<span class="dash">—</span>', t)
        if re.match(r"^[-*•] ", line.strip()):
            if not in_list:
                out.append("<ul>"); in_list = True
            out.append(f"<li>{t[2:]}</li>")
            continue
        if in_list:
            out.append("</ul>"); in_list = False
        if not t:
            continue
        out.append(f'<h3>{t}</h3>' if re.fullmatch(r"<b>[^<]+</b>", t) else f"<p>{t}</p>")
    if in_list:
        out.append("</ul>")
    return "\n".join(out)


ex = args.example if args.example in ids else ids[0]
base_ex, trained_ex = (ROOT / args.base / f"{ex}.md").read_text(), (samples / f"{ex}.md").read_text()
n = len(ids)
cards = "".join(
    f'<div class="card"><div class="tag">{label.upper()}</div><div class="big">{models["Base"][key]} <span class="to">→</span> '
    f'{models["Trained"][key]}</div><div class="small">of {n} summaries, base vs trained</div></div>'
    for label, key in ROWS if key in HEADLINE)
rows = "".join(f'<tr><td>{label}</td>' + "".join(f'<td class="n{" t" if m == "Trained" else ""}">{models[m][key]}</td>' for m in models)
               + "</tr>" for label, key in ROWS)

page = f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>Results · {html.escape(args.run)}</title><style>
html,body{{margin:0;background:#fff;color:#111;font-family:-apple-system,BlinkMacSystemFont,"Helvetica Neue",Arial,sans-serif}}
.wrap{{max-width:1180px;margin:0 auto;padding:32px 24px 48px}}
h1{{font-size:26px;margin:0 0 6px}} .sub{{color:#555;font-size:15px;margin:0 0 26px}}
h2{{font-size:13px;letter-spacing:2px;color:#666;font-weight:600;margin:36px 0 14px}}
.cards{{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}}
.card{{border:1.5px solid #111;border-radius:10px;padding:18px 20px}}
.tag{{font-size:11px;letter-spacing:2px;color:#666;margin-bottom:10px}}
.big{{font-size:40px;font-weight:700}} .to{{color:#aaa;font-weight:400}} .small{{font-size:13px;color:#666;margin-top:4px}}
table{{width:100%;border-collapse:collapse;font-size:15px}}
th,td{{text-align:left;padding:10px 12px;border-bottom:1px solid #e5e5e5}} th{{font-size:12px;letter-spacing:1.5px;color:#666;font-weight:600}}
td.n,th.n{{text-align:right;width:110px}} td.t{{font-weight:700;background:#f2f2f2}}
.charts{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}
.chart{{border:1.5px solid #111;border-radius:10px;padding:16px 18px}} .chart svg{{width:100%;height:auto;display:block;margin-top:8px}}
.chart b{{font-size:15px}} .chart p{{font-size:13px;color:#666;margin:4px 0 0}}
.ex{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}
.sum{{border:1.5px solid #111;border-radius:10px;padding:18px 20px}} .sum.good{{border-width:2.5px}}
.sum h3{{font-size:14px;margin:12px 0 6px}} .sum p,.sum li{{font-size:14px;line-height:1.55;margin:0}} .sum ul{{padding-left:18px;margin:0}} .sum li{{margin-bottom:6px}}
.dash{{background:#111;color:#fff;border-radius:3px;padding:0 3px;font-weight:700}}
.foot{{margin-top:30px;border:1.5px dashed #111;border-radius:10px;padding:14px 18px;font-size:14px;line-height:1.6}}
.nochart{{color:#888;font-size:13px;padding:30px 0}}
</style></head><body><div class="wrap">
<h1>Results</h1>
<p class="sub">GPT-OSS-20B after SFT and DPO on Training Sessions in Amazon SageMaker AI. Scored on {n} test papers the model never saw.</p>
<div class="cards">{cards}</div>
<h2>EVERY CHECK · NUMBER OF SUMMARIES WITH THE PROBLEM</h2>
<table><tr><th>Check</th><th class="n">BASE</th><th class="n">PROMPTED</th><th class="n">TRAINED</th></tr>{rows}</table>
<p class="sub" style="margin-top:10px;font-size:13px">Base is the model with no training. Prompted is the base model with our style rules in the prompt. Trained is this run.</p>
<h2>HOW TRAINING WENT</h2>
<div class="charts">
<div class="chart"><b>Stage 1 · SFT loss</b><p>Goes down as the model learns to write the corrected summaries.</p>{spark(sft, 0, max(sft) if sft else 1, "0", f"{max(sft):.2f}" if sft else "")}</div>
<div class="chart"><b>Stage 2 · DPO preference accuracy</b><p>How often the model prefers the corrected summary over the base draft.</p>{spark(dpo, 0, 1, "0%", "100%")}</div>
</div>
<h2>ONE PAPER BEFORE AND AFTER · arXiv {ex}</h2>
<div class="ex"><div class="sum"><div class="tag">BASE MODEL</div>{render(base_ex)}</div>
<div class="sum good"><div class="tag" style="color:#111;font-weight:700">TRAINED MODEL</div>{render(trained_ex)}</div></div>
<div class="foot"><b>Run</b> {html.escape(args.run)} · {len(sft)} SFT steps and {len(dpo)} DPO steps · LoRA rank {config["lora_rank"]} ·
about {minutes:.0f} minutes{f" · about ${cost:.2f} at list prices" if cost is not None else ""}</div>
</div></body></html>"""
out = run_dir / "results.html"
out.write_text(page)
print(f"Wrote {out.relative_to(ROOT)}")
