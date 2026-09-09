# Multilingual & Crosslingual Fact-Checked Claim Retrieval
### SemEval-2025 Task 7 — M.Tech Thesis Project

## Overview

This repository contains the full experimental pipeline for **Multilingual and
Crosslingual Fact-Checked Claim Retrieval**, addressing all four subtasks of
SemEval-2025 Task 7:

| Subtask | Query Language | Claim Pool Language |
|---|---|---|
| English Monolingual | English | English |
| Hindi Monolingual | Hindi | Hindi |
| English→Hindi Crosslingual | English | Hindi |
| Hindi→English Crosslingual | Hindi | English (mixed pool) |

Given a social-media post, the task is to retrieve the correct fact-checked
claim from a large candidate pool (up to ~272K claims) using dense retrieval —
not generation. All four pipelines share a common backbone
(`intfloat/multilingual-e5-large`) fine-tuned with QLoRA, but diverge in what
actually worked, which is the more interesting part of this project.

## Core Contributions

1. **A reproducible, leakage-free retrieve pipeline** — dataset counts
   verified against the official paper via hard assertions, dev/test
   separation strictly enforced for every hyperparameter (LoRA config, ensemble
   weight α, rerank weight β), and OCR-text fusion to recover signal from
   image-only posts.

2. **A diagnosed failure mode in synthetic crosslingual training data.**
   The English→Hindi crosslingual model, when fine-tuned naively on
   machine-translated pairs, collapsed by ~5x relative to the frozen baseline
   (Success@1: 0.181 → 0.037). Root-caused to the model learning
   *translationese* artifacts instead of genuine semantic alignment. Fixed by
   mixing real (non-translated) monolingual pairs into the training set —
   the only configuration across all experiments that beat the frozen
   baseline on every metric.

3. **Evidence that ensembling frozen + fine-tuned models is not just a
   safety net but often the single largest gain in the whole pipeline** —
   observed consistently across all four subtasks. Because the fine-tuned
   model and the frozen model make *decorrelated* errors, score-level fusion
   (`α · sim_finetuned + (1−α) · sim_frozen`, α tuned only on dev) recovers
   Success@1 losses from fine-tuning while keeping Success@5/10 gains.

4. **A recurring precision/recall trade-off**, documented across every
   subtask: contrastive fine-tuning and cross-encoder reranking reliably
   improve Success@5/10 but frequently *regress* Success@1, because
   hard-negative-based objectives optimize for "correct answer somewhere in
   top-k," not "correct answer specifically in rank 1." Ensembling is shown
   to be the effective countermeasure for this trade-off, not the fine-tuning
   itself.

5. **Retrieve-then-rerank architecture** (bi-encoder → top-50 shortlist →
   cross-encoder rerank), with a documented headroom check before adding the
   reranker (confirming meaningful room to gain), and dev-tuned fusion weight
   β between the reranker's score and the retriever's rank prior.

## Results Summary (Test Set, Success@1 / @5 / @10)

| Subtask | Frozen Baseline | Best Pipeline | Key Lever |
|---|---|---|---|
| English Monolingual | 0.434 / 0.716 / 0.788 | 0.440 / 0.782 / **0.860** | Ensemble (α=0.20) + CE rerank |
| Hindi Monolingual | 0.364 / 0.606 / 0.692 | **0.398** / 0.670 / 0.738 | Ensemble (α=0.40) + CE rerank |
| English→Hindi Crosslingual | 0.181 / 0.513 / 0.638 | **0.263** / 0.581 / 0.650 | Real+synthetic data mixing |
| Hindi→English Crosslingual | 0.192 / 0.511 / 0.614 | **0.226** / 0.584 / 0.689 | Hard-negative LoRA fine-tuning |

The English Monolingual pipeline's Success@10 (0.860) beats the shared task
paper's own reported E5-Large baseline (0.818) by 4.2 points, closing roughly
40% of the remaining gap to the top leaderboard system (PINGAN AI, 0.916) —
using only bi-encoder fine-tuning, score-level ensembling, and a single
cross-encoder rerank stage.

## Method Stack (shared across subtasks)

- **Base encoder:** `intfloat/multilingual-e5-large`, mean-pooled,
  L2-normalized, with E5's required `query:` / `passage:` prefixes
- **Fine-tuning:** QLoRA (4-bit NF4 quantization, LoRA rank 8–16 on
  attention Q/K/V projections), trained on InfoNCE contrastive loss with
  mined hard negatives + in-batch negatives
- **Ensembling:** linear fusion of frozen and fine-tuned similarity scores,
  weight tuned by dev-only grid search
- **Reranking:** `BAAI/bge-reranker-v2-m3` cross-encoder (and
  `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` in an earlier experiment),
  fused with the retriever's rank via a tunable weight, optionally further
  fine-tuned on task-specific positive/negative pairs
  - **Evaluation:** Success@k (k=1, 5, 10), matching the official SemEval task
  metric

## Limitations & Honest Caveats

- Dev/test sets are small (478–627 posts per subtask), so most stage-to-stage
  deltas sit close to the sampling-noise floor (~±2–4 percentage points on
  Success@1); consistent *direction* of improvement across stages is more
  trustworthy than exact magnitudes.
- Training-loop randomness was not fully seeded in places, so small
  differences between similar configurations may partly reflect run-to-run
  variance.
- The English monolingual cross-encoder was fine-tuned on random (not hard)
  negatives — the one place the hard-negative lesson from bi-encoder training
  wasn't carried forward, and a likely source of left-on-the-table
  performance.

## Citation / Related Work

This project builds on the official SemEval-2025 Task 7 shared task
("Multilingual and Crosslingual Fact-Checked Claim Retrieval") baseline and
dataset.
- **Evaluation:** Success@k (k=1, 5, 10), matching the official SemEval task
  metric
