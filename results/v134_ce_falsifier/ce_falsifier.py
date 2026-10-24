"""V134 falsifier: candidate-only cross-encoder (BGE-reranker-v2-m3, pinned).

For each V132 OOF action row, score (user chat, incumbent module) and
(user chat, challenger module) with the cross-encoder. Checks:
  1. diversity: corr(ce_err, v132_err) < 0.95
  2. gain: top-20 OOF actions by ce_diff realize >= +0.00015 mean NDCG
Exploratory only: reuses saved V132 OOF geometry, not promotion-grade.
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path("/home/athilla/Documents/IF_ITB/Lomba/IT-TODAY/intelligo-course-recommender")
REV = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
MODEL = "BAAI/bge-reranker-v2-m3"
OUT = Path("/tmp/ce_falsifier")
OUT.mkdir(exist_ok=True)

t0 = time.time()
led = pd.read_csv(ROOT / "results/v132/v132/v132_action_opportunity_ledger.csv")
print("ledger rows:", len(led), "cols:", list(led.columns))
assert len(led) == 12000

chats = pd.read_csv(ROOT / "data/chat_history.csv")
by_user = chats.groupby("user_id")["user_chat_text"].apply(list)
cat = pd.read_csv(ROOT / "data/modules_catalog.csv").set_index("module_id")
mod_text = {m: str(cat.loc[m, "module_name"]) + " [SEP] " + str(cat.loc[m, "description_and_syllabus"])
            + " [SEP] " + str(cat.loc[m, "prerequisite_level"]) for m in cat.index}

users = led["user_id"].tolist()
queries = [" ".join(str(t) for t in by_user.get(u, [])) for u in users]
inc_docs = [mod_text[m] for m in led["incumbent"]]
ch_docs = [mod_text[m] for m in led["challenger"]]
print(f"empty queries: {sum(1 for q in queries if not q.strip())}")
print(f"prep {time.time()-t0:.1f}s")

tok = AutoTokenizer.from_pretrained(MODEL, revision=REV, trust_remote_code=True)
net = AutoModelForSequenceClassification.from_pretrained(
    MODEL, revision=REV, trust_remote_code=True, torch_dtype=torch.float16).cuda().eval()
print("model loaded")

@torch.no_grad()
def score_pairs(qs, ds, bs=32):
    out = np.zeros(len(qs), dtype=np.float64)
    for i in range(0, len(qs), bs):
        e = tok(list(qs[i:i+bs]), list(ds[i:i+bs]), padding=True, truncation=True,
                max_length=512, return_tensors="pt")
        e = {k: v.cuda() for k, v in e.items()}
        out[i:i+bs] = net(**e).logits.float().cpu().numpy().ravel()
        if i % (bs*20) == 0:
            print(f"  {i}/{len(qs)}", flush=True)
    return out

t0 = time.time()
s_inc = score_pairs(queries, inc_docs)
s_ch = score_pairs(queries, ch_docs)
print(f"scored {time.time()-t0:.1f}s")
np.save(OUT / "ce_inc.npy", s_inc)
np.save(OUT / "ce_ch.npy", s_ch)

ce_diff = s_ch - s_inc
realized = led["realized_delta"].to_numpy(float)
pred = led["pred_utility"].to_numpy(float)
err_ce = ce_diff - realized
err_v = pred - realized
corr = float(np.corrcoef(err_ce, err_v)[0, 1])
print(f"err_corr(ce, v132) = {corr:.4f}  (need < 0.95)")

order = np.argsort(-ce_diff, kind="stable")[:20]
gain20 = float(realized[order].sum() / 4000)
folds_pos = sorted(set(int(f) for f in led["fold"].iloc[order]))
fold_gain = {int(f): float(realized[order[led["fold"].iloc[order].to_numpy() == f]].sum()/4000)
             for f in folds_pos}
print(f"top20 realized gain = {gain20:+.6f} (need >= +0.00015)")
print("folds touched:", folds_pos)
print("per-fold:", {k: round(v, 6) for k, v in fold_gain.items()})
n_benefit = int((realized[order] > 0).sum())
print(f"benefit/harm: {n_benefit}/{20-n_benefit}")

report = {"err_corr": corr, "top20_gain": gain20, "folds": folds_pos,
          "fold_gain": fold_gain, "benefit": n_benefit,
          "pass_diversity": bool(corr < 0.95), "pass_gain": bool(gain20 >= 0.00015)}
(OUT / "falsifier_report.json").write_text(json.dumps(report, indent=1))
print(json.dumps(report, indent=1))
