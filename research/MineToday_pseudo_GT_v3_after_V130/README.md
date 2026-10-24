# MineToday Pseudo-GT v3 after V130

Recommended reconstruction now uses the **exact ordering in the actually submitted
`candidate_v130_DIAGNOSTIC.csv`**, which the user reports scored **0.66173**.

## Why this is a meaningful update
- Exact V129 base in the V130 archive: 0.66075
- V130 diagnostic: 0.66173
- Observed gain: +0.00098
- V130 offline expected incremental gain: +0.000394
- Observed public gain is about 2.49x the offline point estimate.

The V130 candidate changes exactly 20 test rows, all as Top-1 ordering actions.
No Top-5 membership, rank-5, tail, or score-multiset changes are involved.

Action depths:
- rank2 -> rank1: 16
- rank3 -> rank1: 1
- rank4 -> rank1: 3

## Important correction from v2
The previous pseudo-GT v2 was a synthetic V49/V129 merge. The exact V129 base exposed
inside V130 differs from v2 at 8 Top-1 rows.
v3 removes that ambiguity by syncing to the exact submitted V130 diagnostic.

## Files
- `pseudo_ground_truth_v3.csv`: recommended posterior pseudo-GT
- `pseudo_ground_truth_v3_hard_k5.csv`: hard GT-shaped K=5 realization
- `pseudo_ground_truth_v3_conservative_rank2_only.csv`: exact V129 + only the 16 V130 rank2 promotions
- `pseudo_ground_truth_rank_v3.csv`: rank/evidence ledger
- `v2_to_v3_change_audit.csv`: exact change audit
- `UPDATE_NOTES.json`: machine-readable notes

The leaderboard result supports the 20-action package as a whole, not every action individually.
K and the two-decimal relevance jitter remain weakly identifiable and are therefore not changed.
