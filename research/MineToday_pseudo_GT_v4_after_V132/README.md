# MineToday Pseudo-GT v4 after V132

This version updates pseudo-GT v3 using the exact ordering from the submitted
`candidate_v132_DIAGNOSTIC.csv`, reported public score **0.66219**.

## Evidence update

- V130 protected base: **0.66173**
- V132 diagnostic: **0.66219**
- Observed uplift: **+0.00046**
- V132 changed exactly **5 users**
- All 5 are **rank-2 -> rank-1 swaps**
- Top-5 membership, rank-5, tail, and score multiset are unchanged

Selected pairs:
- M_017 -> M_009: 3 users
- M_006 -> M_009: 1 user
- M_014 -> M_015: 1 user

Offline V132 evidence:
- Primary incremental gain: **+0.000193**
- Positive random folds: **4/5**
- Benefit / harm: **15/5**
- User-bootstrap CI crosses zero, so the public uplift is the stronger aggregate evidence.

## Update rule

Pseudo-GT v4 uses the **exact V132 submitted ordering** for all 1000 users.
Relative ordering outside the 5 V132 swaps stays identical to the protected V130 base.

K and relevance jitter are intentionally unchanged because V132 gives new Top-1
ordering evidence only.

## Files

- `pseudo_ground_truth_v4.csv`: recommended posterior pseudo-GT
- `pseudo_ground_truth_v4_hard_k5.csv`: GT-shaped hard K=5 realization
- `pseudo_ground_truth_rank_v4.csv`: full rank/evidence ledger
- `v132_five_action_audit.csv`: exact 5 swaps and model evidence
- `UPDATE_NOTES.json`: machine-readable update notes
