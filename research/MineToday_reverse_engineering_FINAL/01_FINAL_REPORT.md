
# MineToday 2026 Synthetic Dataset Reverse Engineering — Final Forensic Report

## Executive conclusion

The released dataset is demonstrably synthetic and exposes enough regularity to reconstruct
a large fraction of the generator architecture. The strongest recovered stages are:

1. **chat population / engagement generation**
2. **timestamp-window generation**
3. **special pivot trajectory generation**
4. **orphan chat-only dirty-data injection**
5. **assessment archetypes plus a later corruption layer**
6. **K selection**
7. **rank-band relevance materialization**

The one stage that cannot be uniquely recovered from the released columns is the
**exact module-ranking function**. This is not merely an optimization failure:
there are repeated exact observable states with conflicting labels, proving that
released observables alone do not uniquely determine the target.

Therefore the final reconstruction is intentionally split into:
- mechanisms that are essentially recovered;
- mechanisms that are statistically strong but not literal-source-identifiable;
- mechanisms that are provably non-identifiable without additional information.

---

## 1. Dataset/export architecture

Released tables:

- `user_assessments.csv`: 5,000 users
- `train_relevance.csv`: U_0001..U_4000
- `test.csv`: U_4001..U_5000
- `chat_history.csv`: chat records extend through **U_5090**

There are exactly **90 chat-only orphan IDs U_5001..U_5090**.
Across U_0001..U_5090, exactly **150 IDs have no chat**.
Those 150 all belong to the valid U_0001..U_5000 population, so the valid
population has exactly **150/5000 = 3.00% no-chat users**.

This is highly suggestive of a deliberate exact-size 3% sample rather than
an accidental export omission.

---

## 2. Chat engagement generator

Observed valid-user counts:

| n_chat | users |
|---:|---:|
| 0 | 150 |
| 1 | 1435 |
| 2 | 1434 |
| 3 | 482 |
| 4 | 502 |
| 5 | 464 |
| 6 | 90 |
| 7 | 111 |
| 8 | 107 |
| 9 | 116 |
| 10 | 109 |

A compact generator consistent with the marginal is:

```python
select exactly 150 / 5000 no-chat users

for every remaining user:
    tier ~ Categorical([LOW=.60, MEDIUM=.30, HIGH=.10])

    LOW    -> n_chat ~ UniformInteger(1,2)
    MEDIUM -> n_chat ~ UniformInteger(3,5)
    HIGH   -> n_chat ~ UniformInteger(6,10)
```

For the 5,090-ID marginal model including the 90 appended orphan users,
the chi-square goodness-of-fit p-value is **0.395624**.

The within-tier count splits are also consistent with uniform integer choice.

---

## 3. Regular timestamp generator

The timestamps start at **2026-01-05** and exhibit tier-specific finite windows.

### LOW

For non-pivot users with 1–2 chats:

```python
times = sorted(Uniform(BASE, BASE + 5 days) for _ in range(n_chat))
```

Pooled KS p-value against Uniform(0,5d): **0.173955**.

### MEDIUM

For regular non-pivot 3–5-chat users:

```python
times = sorted(Uniform(BASE, BASE + 45 days) for _ in range(n_chat))
```

Pooled KS p-value: **0.322207**.

Order-statistic window estimates by n are all near 45 days.

### HIGH

For 6–10-chat users, order-statistic estimates cluster near 90 days,
and seconds-of-day are uniform. A 90-day sorted-uniform window is the
strongest simple reconstruction, but the pooled 90-day KS test is slightly
too structured to call literal-source exact. Confidence is therefore HIGH,
not VERY_HIGH.

---

## 4. Special pivot trajectory

Exactly **240 users** enter a special trajectory, all within the MEDIUM count regime.
For every one of those users, the pivot message is **chat #2**.

Two pivot text families:

- Data Analyst / Excel+SQL → switch to GenAI: **129**
- Machine Learning → switch to dashboard: **111**

Recovered timing:

```python
t1 ~ Uniform(BASE, BASE + 21 days)
gap_days ~ UniformInteger(25,60)
t2 = t1 + gap_days days

t3..tn = sorted(
    Uniform(t2 + 3 days, t2 + 48 days)
)
```

Evidence:

- t1 Uniform(0,21d) KS p = **0.449613**
- integer gap 25..60 chi-square p = **0.652183**
- post-pivot Uniform(3,48d) KS p = **0.729168**
- pivot t2 preserves the exact clock time of t1 because the gap is an integer number of days

This is one of the strongest recovered script branches.

---

## 5. Orphan chat-only branch

U_5001..U_5090 form a separate 90-user branch:

```python
for uid in 5001..5090:
    n_chat ~ UniformInteger(1,3)
    times = sorted(Uniform(BASE, BASE+60d) for _ in range(n_chat))
```

Evidence:

- count split 25 / 31 / 34, uniform GOF p = **0.496585**
- timestamp Uniform(0,60d) KS p = **0.930018**

These IDs have no assessment rows and are outside the released train/test population,
so this is best interpreted as deliberate dirty-data / orphan-record injection.

---

## 6. Chat text templating

There is an explicit transactional/support-noise family.

Known exact support messages account for **682 / 14,573 = 4.6799%** of chat rows.

First-message support family: 5 templates, counts 47/35/47/42/51.
Uniform-family chi-square p ≈ 0.492.

Follow-up support family: 4 templates, counts 122/121/112/105.
Uniform-family chi-square p ≈ 0.640.

Normal follow-ups are exhaustively covered by 10 leading prefix families such as
`Sori nanya mulu`, `Btw sekalian`, `Ok makasih`, `Oh gitu`, etc.
Their usage is close to uniform (chi-square p ≈ 0.055).

The data therefore strongly support a literal template-list plus `choice(...)` design.

---

## 7. Assessment generator and corruption

The assessment table contains 50,000 cells.

Observed corruption/type injection:

- string-valued cells: **748**
- rows with strings: **246**
- Indonesian number-word cells: **383**
- numeric-string cells: **365**
- out-of-range cells: **251**
- rows with out-of-range cells: **73**

Sentinel examples include `-3`, `-1`, `12`, and `99`.

This cannot plausibly be natural survey noise; it is a deliberate post-generation
data-quality layer.

### Expert override

There are 149 rows where every observed valid value is exactly 5.
There is exactly one additional near-perfect row, U_4876, with nine observed-valid
answers, only one of them non-5 (`Tiga`), plus a `99` sentinel.

That gives a uniquely plausible **150-user = 3% expert override** before corruption.

### Archetypes

Clustering clean non-expert assessments reveals crisp integer-centered prototypes.
The strongest is the advanced profile:

```text
[4,3,3,3,4,3,2,2,2,3]
```

with residual standard deviation around **0.83**, consistent with
rounded/clipped independent Gaussian perturbation around a fixed archetype.

The lower-skill population is multi-archetype, so a single novice/intermediate
prototype would be false precision. Those candidate profiles are preserved in
`07_assessment_archetypes.csv`.

---

## 8. K generator

Train positive-count distribution:

- K=4: 1362
- K=5: 1285
- K=6: 1353

Uniform{4,5,6} GOF p = **0.264676**.

The strongest reconstruction is:

```python
K = rng.choice([4,5,6])
# equivalently rng.integers(4,7)
```

No meaningful dependence on visible user state is needed to explain the K marginal.

---

## 9. Relevance score materialization

This component is effectively cracked.

```python
rank1 = 1.00
rank2 = round(Uniform(.80,.90), 2)
rank3 = round(Uniform(.65,.75), 2)
rank4 = round(Uniform(.50,.60), 2)
rank5 = round(Uniform(.35,.45), 2)  # if K >= 5
rank6 = round(Uniform(.20,.30), 2)  # if K == 6
```

Rounded-uniform GOF p-values:

- rank2: **0.624441**
- rank3: **0.210428**
- rank4: **0.258865**
- rank5: **0.766203**
- rank6: **0.866473**

The half-frequency endpoint bins are the expected signature of a continuous uniform
draw rounded to 2 decimals. The numeric target magnitude is therefore generated
**after the ranking** and should not be interpreted as a calibrated semantic score.

---

## 10. Why the exact module ranker cannot be uniquely reconstructed

This is the key limitation.

Repeated released observable states disagree in their targets:

| observable equality | repeated groups | users | conflicting full-rank groups |
|---|---:|---:|---:|
| same normalized assessment | 316 | 1408 | 316 |
| same exact chat sequence | 232 | 884 | 232 |
| same normalized assessment + same exact chat sequence | 11 | 23 | 11 |

Even in the strongest combined duplicate key, a deterministic mode recovers the exact
rank only about 47.8% of duplicated users.

Therefore the literal generator must contain at least one unreleased degree of freedom:

- latent persona/state not present in the CSV;
- per-user/module RNG;
- random noise added before rank selection;
- or another hidden feature.

This proves that an exact deterministic formula
`released_assessment + released_chat -> exact target rank`
does **not** exist.

---

## 11. RNG brute-force status

Simple small-seed hypotheses have been eliminated.

Tested examples:

- NumPy `default_rng`, direct K stream, seed 0..999,999: no match
- Python `random.Random`, direct K stream, seed 0..999,999: no match
- legacy `RandomState`, direct K stream, seed 0..99,999: no match
- recovered chat-count hierarchical stream, NumPy/Python, seed 0..999,999: no match
- exact 150-user no-chat set as NumPy `choice(..., replace=False)`, seed 0..999,999: no match
- Python `random.sample` no-chat set, seed 0..199,999: no match
- simple 3% all-5 expert mask across multiple RNGs: no match in tested ranges
- direct relevance-jitter streams and common-seed offset scans: no credible exact match

This eliminates a generator of the form:

```python
seed = small_integer
rng = RNG(seed)
# immediately emit one of the observed streams
```

It does **not** eliminate a seeded script where assessment/chat/template generation
consumes the stream before labels, separate RNGs are used per stage, a per-user seed is
derived by hashing, or the seed has high entropy.

Without call order, brute-forcing a generic 64/128-bit state is not mathematically
meaningful.

---

## 12. Final reconstructed architecture

```text
5000 valid synthetic users
│
├── latent archetype
│   └── assessment vector
│       ├── likely exact 3% all-5 expert override
│       ├── fixed integer-centered profiles + noise
│       └── later string/sentinel corruption
│
├── exact-size ~3% no-chat selection
│
├── engagement tier
│   ├── LOW    -> 1..2 chats -> sorted uniform 5-day window
│   ├── MEDIUM -> 3..5 chats -> sorted uniform 45-day window
│   │   └── special 240-user pivot branch with recovered timing
│   └── HIGH   -> 6..10 chats -> approximately sorted uniform 90-day window
│
├── message generation
│   ├── topic/template family
│   ├── ~5% transactional/support noise
│   └── 10 near-uniform follow-up prefixes
│
├── hidden user/module state + assessment + chat intent
│   └── stochastic module utility / ranker    [NOT uniquely identifiable]
│
├── K ~ Uniform{4,5,6}
│
└── rank-band continuous uniform jitter -> round(...,2)

then append 90 chat-only orphan users U_5001..U_5090
    ├── n_chat ~ Uniform{1,2,3}
    └── timestamps ~ Uniform(0,60 days)
```

---

## 13. Confidence boundary

### Effectively recovered / VERY_HIGH
- 5,000 valid-user assessment population
- 90 appended chat-only orphan IDs
- exact 150 no-chat valid users
- low/medium chat-count ranges
- low 5-day and regular-medium 45-day timestamp windows
- 240-user position-2 pivot branch and its timing
- orphan 1..3 count / 60-day timestamp branch
- assessment corruption occurring as a separate synthetic layer
- K ∈ {4,5,6} with uniform behavior
- post-ranking relevance jitter bands

### HIGH
- 60/30/10 engagement probabilities
- 90-day HIGH timestamp window
- exact 150-user pre-corruption expert override
- integer-centered assessment archetypes + independent perturbation
- support-noise ~5% branch
- approximately uniform follow-up-prefix choice

### Unresolved / non-identifiable from released data
- exact global/per-stage seed
- exact RNG library for every stage
- exact RNG call order
- exact lower-skill archetype probabilities
- exact topic-template tables and weights
- exact hidden module-ranking formula

This is the defensible endpoint of reverse engineering from the released artifacts alone.
Claiming a unique literal source file beyond this point would require inventing information
that the outputs do not contain.
