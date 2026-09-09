
*(Adjust the tree above to match your actual repo layout.)*

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
