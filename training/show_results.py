"""Turn a finished training run into an HTML page with the base and trained summaries side by side.

    python training/show_results.py --run my-run
    open runs/my-run/results.html

The page shows each test paper's summary from the base model next to the one from your trained model, with every
em or en dash highlighted. Use the arrows or the menu to move between papers. For the counts behind the results table,
run python training/evaluate.py --trained runs/my-run/samples.
"""
import argparse
import html
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True, help="run name under runs/")
ap.add_argument("--first", default="2608.25593", help="test paper to show first")
ap.add_argument("--base", default="results/base", help="folder with the base model's summaries")
args = ap.parse_args()

samples = ROOT / "runs" / args.run / "samples"
if not samples.exists():
    raise SystemExit(f"{samples} not found. The run has to finish so the model writes its summaries.")
base_dir = ROOT / args.base
ids = [json.loads(l)["arxiv_id"] for l in (ROOT / "data/pairs/test.jsonl").read_text().split("\n") if l.strip()]
ids = [i for i in ids if (samples / f"{i}.md").exists() and (base_dir / f"{i}.md").exists()]
if not ids:
    raise SystemExit("No test summaries found for this run. Run training with the default --sample_splits.")
if args.first in ids:
    ids.remove(args.first)
    ids.insert(0, args.first)


def render(md):
    out, in_list = [], False
    for line in md.strip().split("\n"):
        raw = line.strip()
        t = html.escape(raw)
        t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
        t = re.sub(r"[—–]", '<span class="dash">—</span>', t)
        if re.match(r"^[-*•] ", raw):
            if not in_list:
                out.append("<ul>"); in_list = True
            out.append(f"<li>{t[2:]}</li>")
            continue
        if in_list:
            out.append("</ul>"); in_list = False
        if t:
            out.append(f"<h3>{t}</h3>" if re.fullmatch(r"<b>[^<]+</b>", t) else f"<p>{t}</p>")
    if in_list:
        out.append("</ul>")
    return "\n".join(out)


papers = [{"id": i, "base": render((base_dir / f"{i}.md").read_text()), "trained": render((samples / f"{i}.md").read_text())}
          for i in ids]
options = "".join(f'<option value="{k}">arXiv {p["id"]}</option>' for k, p in enumerate(papers))

page = f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>Base vs trained · {html.escape(args.run)}</title><style>
html,body{{margin:0;background:#fff;color:#111;font-family:-apple-system,BlinkMacSystemFont,"Helvetica Neue",Arial,sans-serif}}
.wrap{{max-width:1180px;margin:0 auto;padding:32px 24px 48px}}
.top{{display:flex;align-items:flex-end;justify-content:space-between;gap:16px;margin-bottom:22px}}
h1{{font-size:26px;margin:0 0 6px}} .sub{{color:#555;font-size:15px;margin:0}}
.nav{{display:flex;gap:8px;align-items:center}}
.nav button,.nav select{{font:inherit;font-size:14px;border:1.5px solid #111;background:#fff;border-radius:8px;padding:6px 12px;cursor:pointer}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:20px}}
.sum{{border:1.5px solid #111;border-radius:10px;padding:20px 22px}} .sum.good{{border-width:2.5px}}
.tag{{font-size:11px;letter-spacing:2px;color:#666;margin-bottom:12px}} .good .tag{{color:#111;font-weight:700}}
.sum h3{{font-size:14px;margin:14px 0 6px}} .sum h3:first-of-type{{margin-top:0}}
.sum p,.sum li{{font-size:14.5px;line-height:1.55;margin:0}} .sum ul{{padding-left:18px;margin:0}} .sum li{{margin-bottom:7px}}
.dash{{background:#111;color:#fff;border-radius:3px;padding:0 3px;font-weight:700}}
</style></head><body><div class="wrap">
<div class="top"><div><h1>Base model vs trained model</h1>
<p class="sub">Same paper, same prompt. Test papers the model never saw. Em dashes are highlighted.</p></div>
<div class="nav"><button onclick="go(-1)">←</button><select id="pick" onchange="show(+this.value)">{options}</select><button onclick="go(1)">→</button></div></div>
<div class="grid"><div class="sum"><div class="tag">BASE MODEL</div><div id="base"></div></div>
<div class="sum good"><div class="tag">TRAINED MODEL</div><div id="trained"></div></div></div>
</div><script>
const P={json.dumps(papers)};let k=0;
function show(i){{k=(i+P.length)%P.length;document.getElementById('base').innerHTML=P[k].base;
document.getElementById('trained').innerHTML=P[k].trained;document.getElementById('pick').value=k;}}
function go(d){{show(k+d)}}
document.addEventListener('keydown',e=>{{if(e.key==='ArrowRight')go(1);if(e.key==='ArrowLeft')go(-1);}});
show(0);
</script></body></html>"""
out = ROOT / "runs" / args.run / "results.html"
out.write_text(page)
print(f"Wrote {out.relative_to(ROOT)} with {len(papers)} papers")
