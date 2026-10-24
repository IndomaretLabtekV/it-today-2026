# MineToday Research Pseudo-Ground-Truth

This package is a **pseudo-GT for forensic analysis**, not recovered hidden ground truth.

## Evidence used
- Organizer-provided `train_relevance.csv`, `test.csv`, chats, assessments, and module catalog.
- The user's own V49 diagnostic/anchor outputs.
- No external hidden labels, leaked test answers, or leaderboard-score inversion are used.

## Latent target geometry recovered from train
- K=4: 1362 users (0.34050)
- K=5: 1285 users (0.32125)
- K=6: 1353 users (0.33825)
- Conditional rank means:
  - r1=1.000000
  - r2=0.849197
  - r3=0.700015
  - r4=0.549072
  - r5|K>=5=0.399905
  - r6|K>=6=0.250399

`pseudo_ground_truth.csv` uses posterior-mean relevance for uncertain ranks 5/6:
- r5 = E[value | rank5 exists] * P(K>=5) = 0.263737
- r6 = E[value | rank6 exists] * P(K>=6) = 0.084698

`pseudo_ground_truth_hard_k5.csv` is a GT-shaped K=5 realization using the same ranking.

## Important limitation
The organizer train data shows K and the small numeric jitter are weakly/non-identifiable from the visible features.
Therefore an honest reconstruction cannot be claimed to be exact or near 1.0000 without hidden-label evidence.
The most defensible part of this reconstruction is the module ordering, especially ranks 1-4.
