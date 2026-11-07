# Post-Training a Model to Summarize AI Papers

We post-trained [GPT-OSS-20B](https://huggingface.co/openai/gpt-oss-20b) to write AI paper summaries in the DAIR.AI Academy house style. Training used SFT and then DPO on [Training Sessions in Amazon SageMaker AI](https://aws.amazon.com/about-aws/whats-new/2026/10/training-session-apis-on-sagemaker-ai/). This repo has everything you need to rebuild the data, train the model and score it the same way we did.

The model reads a paper's title, abstract and introduction. It writes a short curator's take and three or four key points.

## Results

These counts are for the 20 test papers. Each number is how many of the 20 summaries had the problem. "Base" is GPT-OSS-20B with no training. "Prompted" is the base model with our full style spec added to the prompt. "Trained" is our final model.

| Check | Base | Prompted | Trained |
| --- | ---: | ---: | ---: |
| Em or en dashes | 20 | 12 | 1 |
| Wrong bullet label format | 20 | 0 | 0 |
| Opens with "The paper tackles" | 12 | 6 | 0 |
| Missing "First author et al." credit | 13 | 13 | 3 |
| Missing "Why it matters" | 7 | 0 | 1 |
| Results in the curator's take | 1 | 0 | 0 |
| Numbers not in the input | 0 | 0 | 0 |

In a review on the same 20 papers, we picked the trained model's summary over the base model's on all 20 (`results/human-review.json`). The two summaries were shown as A and B in random order. The review was not fully blind, because the base model's em dashes were easy to spot.

An AI reviewer also checked every claim against the paper text. The trained model made about as many factual mistakes as the base model (`results/faithfulness-audit.md`). Training taught the style. It did not make the summaries more accurate.

## How it works

Each training example is a pair of summaries for one paper. The rejected summary is the base model's draft. The chosen summary is a corrected version of that draft that follows `style-spec.md`. There are 82 training pairs, 20 dev pairs and 20 test pairs, split by paper.

Training runs in two stages on one LoRA adapter (rank 64, alpha 128).

1. SFT for 2 epochs on the chosen summaries.
2. DPO for 2 epochs on the pairs, with beta 0.2 and an extra RPO term on the chosen side.

Training Sessions has no built-in DPO loss. We compute the DPO gradient for each summary ourselves and pass it to the built-in importance-sampling loss as a per-sequence weight. The math is in `training/dpo.py`, and it includes a gradient check you can run with `python training/dpo.py`.

## Reproduce it

You need Python 3 (we used 3.14), `pdftotext` from poppler (we used 25.12.0), and an AWS account with access to Training Sessions in Amazon SageMaker AI. Training our final model is estimated at about $0.61 at list prices. A cost guard in the script stops any run that is projected to pass $5.

**1. Install.**

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install --no-deps "git+https://github.com/aws/sagemaker-training-sessions-cookbook@7f2d3e004d5c3e7a96ae59eab0ae2b070abba94a"
brew install poppler   # or: sudo apt install poppler-utils
```

The cookbook is installed with `--no-deps` because it pins a newer `transformers` than the one we used. Our versions are in `requirements.txt`.

**2. Rebuild the inputs.** We do not redistribute paper text. This script downloads the exact arXiv PDF version of all 122 papers, extracts the title, abstract and introduction, and checks every result against our SHA-256 hashes.

```bash
python scripts/build_inputs.py
```

It writes `data/text/` and the full DPO files in `data/dpo/`. It should end with "All 122 inputs match ours exactly."

**3. Score our published outputs.** This needs no AWS account. It reproduces the results table from the summaries in `results/`.

```bash
python training/evaluate.py
```

**4. Configure AWS.** Copy `.env.example` to `.env` and fill in your IAM role, model package group and S3 path. All three must be in the same region.

```bash
cp .env.example .env
```

**5. Check the setup with a short run.** This trains for 2 steps on 4 pairs and writes one summary. It costs a few cents.

```bash
python training/train_dpo.py --run smoke --limit 4 --sft_epochs 1 --max_steps 2 --sample_limit 1 --sample_splits test
```

**6. Train the final model.** The defaults are our final settings. It took about 16 minutes for us.

```bash
python training/train_dpo.py --run my-run
python training/evaluate.py --trained runs/my-run/samples
```

To read the base and trained summaries side by side for every test paper:

```bash
python training/show_results.py --run my-run
open runs/my-run/results.html
```

**7. Optional. Rebuild the baselines.**

```bash
python training/sample_base.py              # base model, writes runs/base/
python training/sample_base.py --prompted   # base model with the style rules, writes runs/prompted/
python training/evaluate.py --base runs/base --prompted runs/prompted --trained runs/my-run/samples
```

## What is in the repo

| Path | Contents |
| --- | --- |
| `prompt.md` | The prompt. It has no style rules, so any style change comes from training |
| `style-spec.md` | The 11 style rules the corrected summaries follow |
| `data/papers.json` | For each paper, the arXiv version, PDF hash, where the input text ends, and the text hash |
| `data/pairs/` | Train, dev and test pairs with the chosen and rejected summaries |
| `scripts/build_inputs.py` | Downloads the papers and rebuilds the model inputs |
| `training/dpo.py` | DPO loss and gradient |
| `training/train_dpo.py` | SFT then DPO on Training Sessions, then sampling |
| `training/sample_base.py` | Base and prompted baselines |
| `training/checks.py`, `training/evaluate.py` | The automatic checks behind the results table |
| `training/show_results.py` | Turns a finished run into an HTML page with base and trained summaries side by side |
| `results/` | Our final model's config, training metrics and summaries, the base and prompted summaries, the human review, the accuracy audit and the Nov 7 reproduction |

## Reproduction check

On Nov 7, 2026 we cloned this repo into an empty folder and followed the steps above with a fresh Python environment. All 122 inputs rebuilt from arXiv matched ours byte for byte. We retrained the final model from scratch, resampled both baselines and scored all three on the 20 test papers. The outputs are in `results/reproduction-2026-11-07/`.

| Check | Base | Prompted | Trained |
| --- | ---: | ---: | ---: |
| Em or en dashes | 11 | 10 | 1 |
| Wrong bullet label format | 2 | 0 | 0 |
| Opens with "The paper tackles" | 1 | 0 | 0 |
| Missing "First author et al." credit | 7 | 4 | 3 |
| Missing "Why it matters" | 9 | 0 | 1 |
| Results in the curator's take | 3 | 0 | 0 |
| Numbers not in the input | 1 | 0 | 0 |

The retrained model reproduced every count in the trained column of the main table exactly.

The base model did not. On Nov 7 the hosted GPT-OSS-20B wrote noticeably better summaries than it did on Nov 2 for the same prompts. The model ARN, tokenizer, prompt tokens and sampling settings were identical. We confirmed the change was on the base model by sampling it through a new, empty model package group. Our main table uses the base outputs from Nov 2 and the prompted outputs from Nov 4, which are in `results/base/` and `results/prompted/`. If you resample the baselines, expect numbers closer to the Nov 7 table.

## Notes

Sampling uses temperature 0, but outputs from the hosted service still vary between sessions. Your summaries will differ in wording from ours even when the counts match.

The papers come from the DAIR.AI Top AI Papers of the Week newsletter. The 10 papers marked `calibration` were the first ones we corrected by hand. AI agents wrote the corrections for the other 112, following the style spec, and we spot-checked 20 of them.

## License

The code is MIT licensed. The summaries in `data/pairs/` and `results/` are licensed under CC BY 4.0. Paper text is not included. The rejected summaries and model outputs were generated by GPT-OSS-20B, which is Apache 2.0 licensed.
