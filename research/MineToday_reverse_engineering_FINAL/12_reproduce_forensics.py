
"""
Recompute the principal forensic statistics from the organizer CSVs.

Usage:
    python 12_reproduce_forensics.py /path/to/inteligo_unzipped
"""
from pathlib import Path
import sys, json, re
import numpy as np
import pandas as pd
from scipy.stats import chisquare, kstest

base = Path(sys.argv[1] if len(sys.argv)>1 else ".")
train = pd.read_csv(base/"train_relevance.csv")
chat = pd.read_csv(base/"chat_history.csv", parse_dates=["timestamp"])
mods = [c for c in train if c.startswith("M_")]

Y=train[mods].to_numpy(float)
sy=np.sort(Y,axis=1)[:,::-1]
K=(Y>0).sum(1)

print("K counts:", {k:int((K==k).sum()) for k in [4,5,6]})
print("K uniform GOF:", chisquare([(K==4).sum(),(K==5).sum(),(K==6).sum()]))

bands={2:(.8,.9),3:(.65,.75),4:(.5,.6),5:(.35,.45),6:(.2,.3)}
for r,(lo,hi) in bands.items():
    vals=sy[K>=r,r-1]
    levels=np.round(np.arange(lo,hi+.001,.01),2)
    obs=np.array([(vals==v).sum() for v in levels])
    exp=np.ones(len(levels)); exp[[0,-1]]=0.5
    exp=exp/exp.sum()*len(vals)
    print(f"rank{r}: n={len(vals)} mean={vals.mean():.6f} GOF p={chisquare(obs,exp).pvalue:.6f}")

chat["uid_num"]=chat.user_id.str.extract(r"(\d+)")[0].astype(int)
counts=chat.groupby("uid_num").size().reindex(range(1,5091),fill_value=0)
print("chat counts:", counts.value_counts().sort_index().to_dict())
print("valid no-chat:", int((counts.reindex(range(1,5001),fill_value=0)==0).sum()))
print("orphan users:", int((chat.uid_num>5000).groupby(chat.uid_num).any().sum()))
