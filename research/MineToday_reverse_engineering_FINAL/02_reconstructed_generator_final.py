
"""
MineToday 2026 synthetic generator: final minimal-equivalent reconstruction.

IMPORTANT
---------
This file deliberately separates:
  A) mechanisms strongly recovered from the released data;
  B) mechanisms that remain non-identifiable.

It is NOT claimed to be the organizer's literal source code.
"""

from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np

N_VALID_USERS = 5000
N_ORPHAN_CHAT_USERS = 90
BASE_DATE = datetime(2026, 1, 5)

MODULES = [f"M_{i:03d}" for i in range(1,18)]
RANK_BANDS = {
    2:(0.80,0.90), 3:(0.65,0.75), 4:(0.50,0.60),
    5:(0.35,0.45), 6:(0.20,0.30),
}

EXPERT_PROFILE = np.array([5,5,5,5,5,5,5,5,5,5])
ADVANCED_PROFILE = np.array([4,3,3,3,4,3,2,2,2,3])

FIRST_SUPPORT = [
    "Halo admin, saya mau reset password akun tapi email verifikasi ga masuk-masuk.",
    "Min mau tanya, pembayaran cicilan kelas saya kok belum keupdate di sistem ya?",
    "Min, kok sertifikat kelulusan saya bulan lalu belum masuk ke email ya?",
    "Website LMS-nya lagi down kah? Saya ga bisa login dari tadi.",
    "Min invoice buat reimbursement kantor saya bisa diterbitkan ga ya?",
]

FOLLOW_SUPPORT = [
    "Oiya btw min, sertifikat kelulusan saya bulan lalu kok belum masuk email ya?",
    "Btw min, pembayaran cicilan kelas saya kok belum keupdate di sistem ya?",
    "Oiya satu lagi, email verifikasi reset password saya kok ga masuk-masuk ya min?",
    "Eh ngomong-ngomong, website LMS-nya lagi down kah? Saya ga bisa login dari tadi.",
]

FOLLOW_PREFIXES = [
    "Sori nanya mulu",
    "Btw sekalian",
    "Ok makasih",
    "Oh gitu",
    "Terus min",
    "Habis itu",
    "Nah kalo udah itu",
    "Oiya lanjut",
    "Eh satu lagi",
    "Eh iya",
]

def uniform_datetime(rng, start: datetime, end: datetime) -> datetime:
    seconds = int((end-start).total_seconds())
    return start + timedelta(seconds=int(rng.integers(0, seconds+1)))

def sorted_uniform_times(rng, n: int, start: datetime, end: datetime):
    return sorted(uniform_datetime(rng, start, end) for _ in range(n))

def choose_exact_no_chat_users(rng):
    # Strongest reconstruction: exact 150 = 3% of the 5000 valid users.
    return set(rng.choice(np.arange(1,5001), size=150, replace=False).tolist())

def choose_engagement(rng):
    # Strongly supported marginal; exact source API unknown.
    return rng.choice(["LOW","MEDIUM","HIGH"], p=[0.60,0.30,0.10])

def choose_chat_count(rng, tier):
    if tier == "LOW":
        return int(rng.integers(1,3))      # 1..2
    if tier == "MEDIUM":
        return int(rng.integers(3,6))      # 3..5
    return int(rng.integers(6,11))         # 6..10

def regular_timestamps(rng, n, tier):
    window = {"LOW":5, "MEDIUM":45, "HIGH":90}[tier]
    return sorted_uniform_times(
        rng, n, BASE_DATE, BASE_DATE + timedelta(days=window)
    )

def pivot_timestamps(rng, n):
    """
    Very strongly recovered medium-user special trajectory:
      t1 ~ U(BASE, BASE+21d)
      t2 = t1 + integer day gap U{25,...,60}
      t3..tn ~ sorted U(t2+3d, t2+48d)
    """
    assert 3 <= n <= 5
    t1 = uniform_datetime(rng, BASE_DATE, BASE_DATE + timedelta(days=21))
    gap = int(rng.integers(25,61))
    t2 = t1 + timedelta(days=gap)
    later = sorted_uniform_times(
        rng, n-2, t2 + timedelta(days=3), t2 + timedelta(days=48)
    )
    return [t1,t2,*later]

def generate_orphan_chat_timestamps(rng):
    out = {}
    for uid in range(5001,5091):
        n = int(rng.integers(1,4))
        out[uid] = sorted_uniform_times(
            rng, n, BASE_DATE, BASE_DATE + timedelta(days=60)
        )
    return out

def sample_expert_mask(rng):
    """
    Likely exact-size 3% expert override before corruption.
    This is HIGH confidence, not proven literal source.
    """
    return set(rng.choice(np.arange(1,5001), size=150, replace=False).tolist())

def generate_advanced_assessment(rng):
    """
    Strong advanced archetype approximation:
      round(Normal([4,3,3,3,4,3,2,2,2,3], sigma≈0.83)), clipped to 0..5
    """
    return np.clip(np.rint(rng.normal(ADVANCED_PROFILE, 0.83)),0,5).astype(int)

def materialize_relevance(ordered_modules, rng):
    """
    Near-certain recovered target materialization.
    """
    k = int(rng.integers(4,7))
    y = {m:0.0 for m in MODULES}
    for rank,m in enumerate(ordered_modules[:k], start=1):
        if rank == 1:
            y[m] = 1.0
        else:
            lo,hi = RANK_BANDS[rank]
            y[m] = round(float(rng.uniform(lo,hi)),2)
    return y

class ModuleRankerNotIdentifiable(RuntimeError):
    pass

def rank_modules_exactly_from_released_columns(*args, **kwargs):
    """
    Intentionally not implemented.

    The train set contains repeated exact observable states
    (same normalized assessment and same exact chat sequence)
    with conflicting module sets/ranks. Therefore the literal ranking
    cannot be a deterministic function of the released columns alone.

    Recovering the organizer's exact ranker would require at least one of:
      - hidden latent state,
      - source code,
      - RNG/call-order leakage,
      - additional unreleased columns.
    """
    raise ModuleRankerNotIdentifiable(
        "Exact module ranker is non-identifiable from released columns."
    )
