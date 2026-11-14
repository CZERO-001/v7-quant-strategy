# -*- coding: utf-8 -*-
"""12层最优顺序 2022/2023验证"""
import os, json
import numpy as np, pandas as pd
OUT=r"D:/NEW2/v4/out"
def log(*a): print(*a, flush=True)
D=pd.read_parquet(os.path.join(OUT,"daily_tdx.parquet")).sort_values(["code","d"]).reset_index(drop=True)
g=D.groupby("code",group_keys=False)
D["prev_close"]=g["close"].shift(1)
D["prev_chg"]=g["close"].pct_change(1)*100
D["ret5"]=g["close"].pct_change(5)*100
D["ret10"]=g["close"].pct_change(10)*100
D["hi60"]=g["close"].transform(lambda s:s.rolling(60).max())
D["dist60hi"]=(D["hi60"]/D["close"]-1)*100
D["hi20"]=g["high"].transform(lambda s:s.rolling(20).max())
D["lo20"]=g["low"].transform(lambda s:s.rolling(20).min())
D["amp20"]=(D["hi20"]-D["lo20"])/D["close"]*100
for p in (5,10,20,60):
    D[f"ma{p}"]=g["close"].transform(lambda s:s.rolling(p).mean())
D["ma_bull"]=(D["ma5"]>D["ma10"])&(D["ma10"]>D["ma20"])&(D["ma20"]>D["ma60"])
D["volratio"]=g["vol"].transform(lambda s:s.rolling(5).mean().shift(1))/g["vol"].transform(lambda s:s.rolling(20).mean().shift(1))
dmap=D.set_index(["d","code"])
def prep(y):
    R=pd.read_parquet(os.path.join(OUT,f"detail6_{y}.parquet"))
    R=R.merge(D[["d","code","prev_chg","ret5","ret10","dist60hi","amp20","ma_bull","close","open","volratio"]],left_on=["日期","股票代码"],right_on=["d","code"],how="left")
    return R.dropna(subset=["T1"])
SEQS={
 "当前V7":["主1","主2","N1","主3","主4","主5","主6","R0","R2","R4","R5","X1"],
 "最优12层":["主2","R0","主3","R5","X1","主4","主6","N1","R2","主5","主1","R4"],
}
def build(R,seq):
    A={"主1":(R["ret5"]>10)&(R["dist60hi"]<2)&(R["开盘跳空"]>0)&(R["amp20"]>4),
       "主2":(R["突破时涨幅"]>=5)&(R["突破时涨幅"]<7)&(R["前20日涨幅"]>30)&(R["买入价"]<20)&(R["开盘跳空"]>2),
       "主3":(R["突破时涨幅"]>=5)&(R["突破时涨幅"]<7)&(R["前20日涨幅"]>30)&(R["量能倍数"]>=4)&(R["ma_bull"]),
       "主4":(R["突破时涨幅"]>=8)&(R["突破时涨幅"]<10)&(R["开盘跳空"]>2)&(R["前20日涨幅"]>10)&(R["ret5"]<-5),
       "主5":(R["突破时涨幅"]>=7)&(R["突破时涨幅"]<10)&(R["前20日涨幅"]<-10)&(R["prev_chg"]<-2)&(R["突破分钟"]<=690),
       "主6":(R["突破时涨幅"]>=5)&(R["突破时涨幅"]<7)&(R["前20日涨幅"]>30)&(R["量能倍数"]>=4)&(R["volratio"]>1.2),
       "N1":(R["突破时涨幅"]>=5)&(R["突破时涨幅"]<7)&(R["前20日涨幅"]>40)&(R["量能倍数"]>=4)&(R["ret10"]>20),
       "R0":(R["突破时涨幅"]>=5)&(R["突破时涨幅"]<6)&(R["量能倍数"]>=3)&(R["突破时累计成交额"]>=2)&(R["前20日涨幅"]>50)&(R["突破分钟"]<=600),
       "R2":(R["突破时涨幅"]>=5)&(R["突破时涨幅"]<7)&(R["量能倍数"]>=4)&(R["突破时累计成交额"]>=1)&(R["前20日涨幅"]>40),
       "R4":(R["突破时涨幅"]>=4)&(R["突破时涨幅"]<6)&(R["量能倍数"]>=4)&(R["突破时累计成交额"]>=2)&(R["前20日涨幅"]>30)&(R["开盘跳空"]>0),
       "R5":(R["突破时涨幅"]>=5)&(R["突破时涨幅"]<8)&(R["前20日涨幅"]<-10)&(R["沉寂天数"]>=20)&(R["突破分钟"]<=600),
       "X1":(R["突破时涨幅"]>=8)&(R["突破时涨幅"]<10)&(R["开盘跳空"]>2)&(R["ret5"]>5)&(R["量能倍数"]>=4)}
    used=set(); picks=[]
    for nm in seq:
        sub=R[A[nm]&(~R["日期"].isin(used))]
        if len(sub)==0: continue
        sel=sub.sort_values(["日期","突破时涨幅"],ascending=[True,False]).groupby("日期").head(1).copy()
        sel["层"]=nm; picks.append(sel); used|=set(sel["日期"])
    return pd.concat(picks,ignore_index=True)
for y in ("2022","2023"):
    R=prep(y)
    log(f"\n===== {y} =====")
    for nm,seq in SEQS.items():
        sig=build(R,seq)
        nav=10000.0
        for r in sig.itertuples(): nav*=(1+r.T1/100)
        log(f"{nm}: {len(sig)}条 均{sig['T1'].mean():.2f}% 胜{(sig['T1']>0).mean()*100:.1f}% 全仓1万->{nav:,.0f}")
        log(f"  层分布: {sig['层'].value_counts().to_dict()}")
