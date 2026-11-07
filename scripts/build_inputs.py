"""Rebuild the model inputs for all 122 papers and write the full DPO files to data/dpo/.

We do not redistribute paper text. For each paper, data/papers.json records the exact arXiv PDF version, its SHA-256,
the pdftotext line where the input stops, and the SHA-256 of the resulting text. This script downloads each PDF,
extracts title, abstract and introduction with pdftotext, and checks every hash. A mismatch means your input differs
from ours, and the script says which paper and why.

Requires pdftotext from poppler. We used version 25.12.0.
"""
import hashlib
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PDF_DIR, TEXT_DIR, DPO_DIR = ROOT / "data/pdf", ROOT / "data/text", ROOT / "data/dpo"
sha = lambda b: hashlib.sha256(b).hexdigest()


def pdftotext_version():
    try:
        out = subprocess.run(["pdftotext", "-v"], capture_output=True, text=True)
    except FileNotFoundError:
        raise SystemExit("pdftotext not found. Install poppler (macOS: brew install poppler, Ubuntu: apt install poppler-utils).")
    return (out.stderr or out.stdout).splitlines()[0].strip()


def download(arxiv_id, version, expected):
    f = PDF_DIR / f"{arxiv_id}v{version}.pdf"
    if f.exists() and sha(f.read_bytes()) == expected:
        return f
    url = f"https://arxiv.org/pdf/{arxiv_id}v{version}"
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "paper-summary-dpo reproduction"})
            f.write_bytes(urllib.request.urlopen(req, timeout=90).read())
            time.sleep(1)  # be polite to arXiv
            return f
        except Exception as e:
            print(f"  retry {arxiv_id}: {e}", flush=True)
            time.sleep(5 * (attempt + 1))
    raise SystemExit(f"could not download {url}")


def main():
    version = pdftotext_version()
    print(version)
    if "25.12.0" not in version:
        print("  note: we used pdftotext 25.12.0. Another version may extract slightly different text.")
    for d in (PDF_DIR, TEXT_DIR, DPO_DIR):
        d.mkdir(parents=True, exist_ok=True)
    papers = json.loads((ROOT / "data/papers.json").read_text())
    prompt = (ROOT / "prompt.md").read_text()
    texts, problems = {}, []
    for n, p in enumerate(papers, 1):
        i = p["arxiv_id"]
        f = download(i, p["arxiv_version"], p["pdf_sha256"])
        if sha(f.read_bytes()) != p["pdf_sha256"]:
            problems.append(f"{i}: PDF hash differs from ours")
            continue
        lines = subprocess.run(["pdftotext", "-enc", "UTF-8", str(f), "-"], capture_output=True, text=True).stdout.split("\n")
        k = p["end_line"]
        if (lines[k] if k < len(lines) else "") != p["end_line_text"]:
            problems.append(f"{i}: line {k} is {lines[k]!r} in your extraction, expected {p['end_line_text']!r}")
            continue
        text = "\n".join(lines[:k]).strip()
        if sha(text.encode()) != p["text_sha256"]:
            problems.append(f"{i}: extracted text hash differs from ours")
            continue
        (TEXT_DIR / f"{i}.txt").write_text(text + "\n")
        texts[i] = text
        print(f"[{n}/{len(papers)}] {i} ok", flush=True)

    for split in ("train", "dev", "test"):
        rows = [json.loads(l) for l in (ROOT / "data/pairs" / f"{split}.jsonl").read_text().split("\n") if l.strip()]
        out = [{"arxiv_id": r["arxiv_id"], "source": r["source"], "prompt": prompt.replace("{paper_text}", texts[r["arxiv_id"]]),
                "chosen": r["chosen"], "rejected": r["rejected"]} for r in rows if r["arxiv_id"] in texts]
        (DPO_DIR / f"{split}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out))
        print(f"data/dpo/{split}.jsonl: {len(out)} of {len(rows)} pairs")

    if problems:
        print(f"\n{len(problems)} papers did not match:")
        print("\n".join("  " + x for x in problems))
        sys.exit(1)
    print("\nAll 122 inputs match ours exactly.")


if __name__ == "__main__":
    main()
