# -*- coding: utf-8 -*-
"""v4 阶段28：v6基础上加入v2新规则，测试不同优先级"""
import os, json
import numpy as np, pandas as pd

OUT=r"D:\NEW2\v4\out"; BASE=r"D:\NEW\cyb_ma_monitor"
def log(*a): print(*a, flush=True)
dia=pd.read_parquet(os.path.join(BASE,"cache","ohlcv_cyb.parquet")); dia["d"]=dia["d"].astype(str)
dates=sorted([x for x in dia["d"].unique() if x>="20260105" and x<="20261113"]); date_pos={x:i for i,x in enumerate(dates)}
d=pd.read_parquet(os.path.join(OUT,"features_v2.parquet"))
z=np.load(os.path.join(OUT,"cond_matrix_v2.npz"),allow_pickle=True); M=z["M"]; names=[str(x) for x in z["names"]]
idx={n:i for i,n in enumerate(names)}
T1=d["T1"].values.astype(float); chg=d["突破时涨幅"].values.astype(float)
day_code,uniq_days=pd.factorize(d["日期"].astype(str).values); n_days=len(uniq_days)
starts=np.zeros(n_days,int); ends=np.zeros(n_days,int)
for i in range(n_days):
    idxr=np.where(day_code==i)[0]; starts[i]=idxr[0]; ends[i]=idxr[-1]+1
def mask(conds):
    m=np.ones(len(d),bool)
    for c in conds: m&=M[:,idx[c]]
    return m

plans=json.load(open(os.path.join(OUT,"final_plans_refined.json"),encoding="utf-8"))
C_layers=[tuple(int(x) for x in l) for l in plans["C_覆盖"]["layers"]]
def cmask(idxs):
    m=np.ones(len(d),bool)
    for k in idxs: m&=M[:,k]
    return m
L=[]
for i,idxs in enumerate(C_layers): L.append((f"主{i+1}",cmask(idxs)))
L.append(("R0",mask(["涨5-6","量>=3","额>=2亿","前20>50","10点前"])))
L.append(("R2",mask(["涨5-7","量>=4","额>=1亿","前20>40"])))
L.append(("R4",mask(["涨4-6","量>=4","额>=2亿","前20>30","高开"])))
L.append(("R5",mask(["涨5-8","前20<-10","沉寂>=20","10点前"])))
L.append(("X1",mask(["涨8-10","高开>2","ret5>5","量>=4"])))
NEW={
 "N1":mask(["涨5-7","前20>40","量>=4","ret10>20"]),
 "N2":mask(["涨5-7","前20>30","量>=4","ret10>20"]),
 "N3":mask(["涨5-6","ret60>30","量>=5","昨弱势(占比<30)"]),
}
for k,m in NEW.items(): log(f"{k}: 独立 {int(m.sum())} 条, 均值 {T1[m].mean():.2f}%")

def build(layers):
    score=np.full(len(d),-1e9)
    for step,(nm,m) in enumerate(layers):
        score=np.maximum(score,np.where(m,-step*1e6+chg,-1e9))
    rows=[]
    for i in range(n_days):
        s,e=starts[i],ends[i]; j=int(np.argmax(score[s:e]))
        if score[s+j]>-1e8: rows.append(dict(日期=uniq_days[i],row=s+j,层=int(round(-score[s+j]/1e6))+1))
    out=pd.DataFrame(rows); out["T1"]=T1[out["row"]]
    out["买入价"]=d["买入价"].values[out["row"]]; out["当日收盘涨幅"]=d["当日收盘涨幅"].values[out["row"]]
    return out
def met(df):
    daily=df.groupby("日期")["T1"].mean(); r=daily/100; nav=np.cumprod(1+r)
    return dict(n=len(df),mean=df["T1"].mean(),th=(nav.iloc[-1]-1)*100,win=(df["T1"]>0).mean()*100)
def sim(sig,start=10000.0):
    sig=sig.dropna(subset=["T1"]).sort_values("日期"); sm={str(r["日期"]):r for _,r in sig.iterrows()}
    cash=start; pos=None; eq=[]; tr=[]
    for dt in dates:
        can=pos is None
        if pos is not None and pos["sell_date"]==dt:
            a=pos["shares"]*pos["sell_px"]; f=max(5,a*0.00025)+a*0.0005+a*0.00001; cash+=a-f; tr.append(1); pos=None; can=False
        if can and dt in sm and pos is None:
            r=sm[dt]; px=float(r["买入价"]); sh=int(cash//(px*100))*100
            while sh>=100:
                a=sh*px; f=max(5,a*0.00025)+a*0.00001
                if a+f<=cash+1e-9: break
                sh-=100
            if sh>=100:
                a=sh*px; f=max(5,a*0.00025)+a*0.00001; cash-=a+f
                nxt=dates[date_pos[dt]+1] if date_pos[dt]+1<len(dates) else None
                cp=float(r["当日收盘涨幅"]) if np.isfinite(r.get("当日收盘涨幅",np.nan)) else np.nan
                pos=dict(shares=sh,sell_date=nxt,sell_px=px*(1+float(r["T1"])/100),close_px=px*(1+cp/100) if np.isfinite(cp) else px)
        eq.append((dt, cash+ (pos["shares"]*(pos["close_px"] if (pos["sell_date"] and dt<pos["sell_date"]) else pos["sell_px"]) if pos else 0)))
    e=pd.DataFrame(eq,columns=["日期","equity"]); final=e["equity"].iloc[-1]; peak=e["equity"].cummax()
    return final,(e["equity"]/peak-1).min()*100,len(tr)
def rep(df,label):
    m=met(df); f,dd,n=sim(df)
    log(f"{label}: {len(df)}条 {m['mean']:.2f}% 胜{m['win']:.1f}% 理论{m['th']:,.0f}% | 1万->{f:,.0f} 回撤{dd:.1f}% 做{n}笔")
    return f

rep(build(L),"v6(v2特征) 基准")
# 插入位置：主1后、主2后、主6后、X1后
for pos_name,pos in [("主1后",1),("主2后",2),("主6后",6),("X1后",len(L))]:
    layers=L[:pos]+[("N1",NEW["N1"])]+L[pos:]
    rep(build(layers),f"N1插入{pos_name}")
# 同时插入 N1+N2
for pos in (2,6):
    layers=L[:pos]+[("N1",NEW["N1"]),("N2",NEW["N2"])]+L[pos:]
    rep(build(layers),f"N1+N2插入主{pos}后")
# N1 替换主3（因为都涉及涨5-7+量>=4）
layers=[x for x in L if x[0]!="主3"]
layers=L[:2]+[("N1",NEW["N1"])]+layers[2:]
rep(build(layers),"N1替换主3")
