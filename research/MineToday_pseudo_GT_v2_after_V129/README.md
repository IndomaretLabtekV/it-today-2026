# MineToday Pseudo-GT v2

Updated after V129.

## What changed
- V129 selects 50 Top1 promotions.
- 5 of those already agree with pseudo-GT v1.
- 45 are new Top1 changes versus pseudo-GT v1.
- The rest of each user's module order is preserved from pseudo-GT v1, so V49 lower-order evidence is not discarded.

## Why update
V129 is materially stronger evidence than V49 for Top1:
- primary offline gain: +0.001130
- random folds positive: 5/5
- seen gain: +0.001193
- unseen gain: +0.001072
- latest / near / contiguous worlds are all positive at 5% coverage
- user-reported public score: 0.66075

The exact-template world is weaker, so this is not treated as exact hidden GT.

## Files
- pseudo_ground_truth_v2.csv: recommended posterior pseudo-GT
- pseudo_ground_truth_v2_conservative_top30.csv: only the 30 strongest V129 Top1 actions
- pseudo_ground_truth_rank_v2.csv: rank + evidence ledger
- UPDATE_NOTES.json: machine-readable methodology

K and relevance jitter are intentionally unchanged because V129 provides new Top1 evidence, not new information about K or the random score jitter.
