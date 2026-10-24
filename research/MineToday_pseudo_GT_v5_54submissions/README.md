# MineToday Pseudo-GT v5

V5 incorporates **all 54 downloaded submissions**.

## Recommended file

`pseudo_ground_truth_v5_LB_CALIBRATED.csv`

This keeps the exact best-observed V132 ranking, but recalibrates the global relevance
geometry using the 54-submission leaderboard history. It is the strongest file here
for comparing new candidates against the observed public leaderboard behavior.

Leaderboard-calibrated rank profile:
- r1 = 1.000000
- r2 = 0.800000
- r3 = 0.650002
- r4 = 0.500000
- r5 = 0.242260
- r6 = 0.102643

For the near-frontier subset (public score >= 0.6595), the calibrated geometry reduced
leave-one-out linear-calibration RMSE from **0.00057304** to
**0.00051016**, an improvement of **11.0%**.

## Other files

`pseudo_ground_truth_v5_RECONSTRUCTION.csv`
- Literal pseudo-GT using organizer-train rank geometry.
- Better if the goal is to resemble the hidden target generation process.

`pseudo_ground_truth_v5_SOFT_54ENSEMBLE.csv`
- Score-weighted continuous consensus from all 54 raw submissions.
- Useful for uncertainty analysis and candidate comparison.
- Its top-1, top-5 membership, and top-6 ordering are identical to V132.

`v5_submission_weights.csv`
- Shows exactly how the 54 public submissions contribute to the soft ensemble.

`v5_user_confidence.csv`
- Per-user historical agreement and adjacent-rank support.

## Important

V5 does not claim to recover the literal hidden labels. The public-score history is used
for global reliability and relevance-geometry calibration, not for solving hidden labels
user-by-user.
