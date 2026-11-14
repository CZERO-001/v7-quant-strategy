# -*- coding: utf-8 -*-
"""12层顺序优化（层数不变）"""
import os, json, itertools
import numpy as np, pandas as pd
OUT=r"D:/NEW2/v4/out"; BASE=r"D:/NEW/cyb_ma_monitor"
def log(*a): print(*a, flush=True)
d=pd.read_parquet(os.path.join(OUT,"features_v2.parquet"))
z=np.load(os.path.join(OUT,"cond_matrix_v2.npz"),allow_pickle=True); M=z["M"]; names=[str(x) for x in z["names"]]
idx={n:i for i,n in enumerate(names)}
T1=d["T1"].values.astype(float); chg=d["突破时涨幅"].values.astype(float)
day_code,uniq=pd.factorize(d["日期"].astype(str).values); nd=len(uniq)
st=np.zeros(nd,int); en=np.zeros(nd,int)
for i in range(nd):
    rr=np.where(day_code==i)[0]; st[i]=rr[0]; en[i]=rr[-1]+1
def mask(k):
    m=np.ones(len(d),bool)
    for c in k: m&=M[:,idx[c]]
    return m
plans=json.load(open(os.path.join(OUT,"final_plans_refined.json"),encoding="utf-8"))
C=[tuple(int(x) for x in l) for l in plans["C_覆盖"]["layers"]]
def cmask(idxs):
    m=np.ones(len(d),bool)
    for k in idxs: m&=M[:,k]
    return m
ALL={"主1":cmask(C[0]),"主2":cmask(C[1]),"主3":cmask(C[2]),"主4":cmask(C[3]),"主5":cmask(C[4]),"主6":cmask(C[5]),
 "N1":mask(["涨5-7","前20>40","量>=4","ret10>20"]),
 "R0":mask(["涨5-6","量>=3","额>=2亿","前20>50","10点前"]),
 "R2":mask(["涨5-7","量>=4","额>=1亿","前20>40"]),
 "R4":mask(["涨4-6","量>=4","额>=2亿","前20>30","高开"]),
 "R5":mask(["涨5-8","前20<-10","沉寂>=20","10点前"]),
 "X1":mask(["涨8-10","高开>2","ret5>5","量>=4"])}
def evals(seq):
    score=np.full(len(d),-1e9)
    for step,nm in enumerate(seq):
        score=np.maximum(score,np.where(ALL[nm],-step*1e6+chg,-1e9))
    rows=[]
    for i in range(nd):
        s,e=st[i],en[i]; j=int(np.argmax(score[s:e]))
        if score[s+j]>-1e8: rows.append(s+j)
    if not rows: return 0,0,0
    t=T1[rows]; nav=10000.0
    for x in t: nav*=(1+x/100)
    return nav,float(t.mean()),len(t)
NAMES=list(ALL.keys())
log(f"层集合: {NAMES}")
# 当前 V7 顺序
V7=["主1","主2","N1","主3","主4","主5","主6","R0","R2","R4","R5","X1"]
f,m,nn=evals(V7)
log(f"当前V7顺序: {nn}条 均{m:.2f}% 1万->{f:,.0f}")
# 贪心
cur=[]; rest=NAMES[:]
log("\n=== 贪心（12层）===")
for step in range(len(NAMES)):
    best=None
    for nm in rest:
        for pos in range(len(cur)+1):
            cand=cur[:pos]+[nm]+cur[pos:]
            f2,m2,n2=evals(cand)
            if best is None or f2>best[0]: best=(f2,m2,n2,nm,pos)
    f2,m2,n2,nm,pos=best
    cur=cur[:pos]+[nm]+cur[pos:]
    rest=[x for x in rest if x!=nm]
    log(f"第{step+1}层 {nm} @{pos}: {n2}条 均{m2:.2f}% 1万{f2:,.0f}")
# 局部搜索
def local(seq,rounds=30):
    cur=seq[:]; f0,m0,n0=evals(cur)
    log(f"\n=== 局部搜索 起点 {f0:,.0f} ===")
    for rd in range(rounds):
        imp=False
        for i,j in itertools.combinations(range(len(cur)),2):
            cand=cur[:]; cand[i],cand[j]=cand[j],cand[i]
            f2,m2,n2=evals(cand)
            if f2>f0*1.0001:
                cur=cand; f0=f2; imp=True
                log(f"  交换{i}<->{j}: {f2:,.0f}")
        for i in range(len(cur)-2):
            cand=cur[:]; cand[i],cand[i+1],cand[i+2]=cand[i+2],cand[i],cand[i+1]
            f2,m2,n2=evals(cand)
            if f2>f0*1.0001:
                cur=cand; f0=f2; imp=True
                log(f"  旋转{i},{i+1},{i+2}: {f2:,.0f}")
        if not imp:
            log(f"  第{rd+1}轮无提升")
            break
    return cur,f0
cur2,f2=local(cur)
f,m,nn=evals(cur2)
log(f"\n最优12层顺序: {cur2}")
log(f"{nn}条 均{m:.2f}% 1万->{f:,.0f}")
