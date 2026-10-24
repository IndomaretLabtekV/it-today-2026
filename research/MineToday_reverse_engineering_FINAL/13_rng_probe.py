
"""
Focused RNG seed probes.

These probes intentionally test only hypotheses that are falsifiable from
the released output. They do NOT pretend that a 64/128-bit seed can be
exhaustively brute-forced without knowing RNG call order.
"""
import random
import numpy as np

def probe_default_rng_k(target, max_seed=1_000_000):
    target=np.asarray(target,dtype=int)
    for seed in range(max_seed):
        r=np.random.default_rng(seed)
        if np.array_equal(r.integers(4,7,size=len(target)),target):
            return seed
    return None

def probe_python_k(target, max_seed=1_000_000):
    target=list(target)
    for seed in range(max_seed):
        r=random.Random(seed)
        x=[r.choice([4,5,6]) for _ in target]
        if x==target:
            return seed
    return None

def probe_default_rng_nochat(target_indices_zero_based, max_seed=1_000_000):
    target=np.sort(np.asarray(target_indices_zero_based,dtype=int))
    for seed in range(max_seed):
        r=np.random.default_rng(seed)
        x=np.sort(r.choice(5000,size=150,replace=False))
        if np.array_equal(x,target):
            return seed
    return None

if __name__=="__main__":
    print("Import this file and pass observed fingerprints.")
