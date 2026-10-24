# V134 cross-encoder falsifier (CLOSED)

Candidate-only BAAI/bge-reranker-v2-m3 (rev 953dc6f6, fp16) scored all
12000 V132 OOF action rows (incumbent + challenger) and ranked top-20 by
ce_diff. Result: err_corr 0.055 (diverse, pass) but top-20 gain -0.000334,
5/15 benefit (need >= +0.00015). Two calibrated variants (per-user z,
utility-flavored) also negative. Branch closed as action selector.
Margin-gate observation (9/10 on selected) is post-hoc on n=20 realized
outcomes (24% likely under V132's own 75% base rate): hypothesis only.
