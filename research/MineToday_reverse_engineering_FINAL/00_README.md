
# MineToday Reverse Engineering — Canonical Final Pack

Start here:

1. `01_FINAL_REPORT.md`
   - Main forensic conclusion and recovered generator architecture.

2. `02_reconstructed_generator_final.py`
   - Executable minimal-equivalent generator for the recovered stages.
   - Deliberately refuses to fabricate an exact module ranker.

Evidence:
- `03_target_geometry.csv`
- `04_chat_generator_evidence.csv`
- `05_chat_template_counts.csv`
- `06_assessment_forensics.json`
- `07_assessment_archetypes.csv`
- `08_duplicate_nonidentifiability.csv`
- `09_rng_bruteforce_log.csv`
- `10_user_state_forensics.csv`
- `11_hypothesis_ledger.csv`

Reproducibility:
- `12_reproduce_forensics.py`
- `13_rng_probe.py`

The package distinguishes observed facts, statistical reconstruction, and
non-identifiable components. That separation is intentional: it makes the analysis
stronger than a script that merely guesses unknown internals.
