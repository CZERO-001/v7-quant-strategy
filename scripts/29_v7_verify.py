# -*- coding: utf-8 -*-
"""v4 阶段29：N1 规则稳健性检查 + v7 方案验证"""
import os, json
import numpy as np, pandas as pd

OUT=r"D:\NEW2\v4\out"; BASE=r"D:\NEW\cyb_ma_monitor"
def log(*a): print(*a, flush=True)
d=pd.read_parquet(os.path.join(OUT,"features_v2.parquet"))
z=np.load(os.path.join(OUT,"cond_matrix_v2.npz"),allow_pickle=True); M=z["M"]; names=[str(x) for x in z["names"]]
idx={n:i for i,n in enumerate(names)}
T1=d["T1"].values.astype(float)
def mask(conds):
    m=np.ones(len(d),bool)
    for c in conds: m&=M[:,idx[c]]
    return m
N1=mask(["涨5-7","前20>40","量>=4","ret10>20"])
log(f"N1: {int(N1.sum())} 条 均值 {T1[N1].mean():.2f}% 胜率 {(T1[N1]>0).mean()*100:.1f}%")
sub=d[N1][["日期","股票代码","股票名称","突破时涨幅","买入价","量能倍数","前20日涨幅","ret10","T1"]].sort_values("日期")
log(sub.to_string(index=False))
v=np.sort(T1[N1]); log(f"去最大1: {v[:-1].mean():.2f}% | 去最大2: {v[:-2].mean():.2f}% | 最小: {v[0]:.2f}%")
log(f"分月: {sub.assign(月=sub['日期'].str[:6]).groupby('月')['T1'].mean().round(2).to_dict()}")
log("参数邻域:")
for c in [["涨5-7","前20>40","量>=4","ret10>20"],["涨5-7","前20>40","量>=4","ret10>10"],
          ["涨5-7","前20>30","量>=4","ret10>20"],["涨5-7","前20>40","量>=3","ret10>20"],
          ["涨5-6","前20>40","量>=4","ret10>20"],["涨5-8","前20>40","量>=4","ret10>20"],
          ["涨5-7","前20>50","量>=4","ret10>20"],["涨5-7","前20>40","量>=5","ret10>20"]]:
    m=mask(c); log(f"  {' & '.join(c)}: {int(m.sum())}条 {T1[m].mean():.2f}%")

# 构建 v7 并模拟
dia=pd.read_parquet(os.path.join(BASE,"cache","ohlcv_cyb.parquet")); dia["d"]=dia["d"].astype(str)
dates=sorted([x for x in dia["d"].unique() if x>="20260105" and x<="20261113"]); date_pos={x:i for i,x in enumerate(dates)}
chg=d["突破时涨幅"].values.astype(float)
day_code,uniq_days=pd.factorize(d["日期"].astype(str).values); n_days=len(uniq_days)
starts=np.zeros(n_days,int); ends=np.zeros(n_days,int)
for i in range(n_days):
    idxr=np.where(day_code==i)[0]; starts[i]=idxr[0]; ends[i]=idxr[-1]+1
plans=json.load(open(os.path.join(OUT,"final_plans_refined.json"),encoding="utf-8"))
C_layers=[tuple(int(x) for x in l) for l in plans["C_覆盖"]["layers"]]
def cmask(idxs):
    m=np.ones(len(d),bool)
    for k in idxs: m&=M[:,k]
    return m
L=[]
for i,idxs in enumerate(C_layers[:2]): L.append((f"主{i+1}",cmask(idxs)))
L.append(("N1",N1))
for i,idxs in enumerate(C_layers[2:],start=2): L.append((f"主{i+1}",cmask(idxs)))
L.append(("R0",mask(["涨5-6","量>=3","额>=2亿","前20>50","10点前"])))
L.append(("R2",mask(["涨5-7","量>=4","额>=1亿","前20>40"])))
L.append(("R4",mask(["涨4-6","量>=4","额>=2亿","前20>30","高开"])))
L.append(("R5",mask(["涨5-8","前20<-10","沉寂>=20","10点前"])))
L.append(("X1",mask(["涨8-10","高开>2","ret5>5","量>=4"])))
score=np.full(len(d),-1e9)
for step,(nm,m) in enumerate(L):
    score=np.maximum(score,np.where(m,-step*1e6+chg,-1e9))
rows=[]
for i in range(n_days):
    s,e=starts[i],ends[i]; j=int(np.argmax(score[s:e]))
    if score[s+j]>-1e8: rows.append(dict(日期=uniq_days[i],row=s+j,层=int(round(-score[s+j]/1e6))+1))
v7=pd.DataFrame(rows); v7["T1"]=T1[v7["row"]]
v7["买入价"]=d["买入价"].values[v7["row"]]; v7["当日收盘涨幅"]=d["当日收盘涨幅"].values[v7["row"]]
v7["股票代码"]=d["股票代码"].values[v7["row"]]; v7["股票名称"]=d["股票名称"].values[v7["row"]]
v7["层名"]=v7["层"].map({i+1:nm for i,(nm,m) in enumerate(L)})
v7.to_csv(os.path.join(OUT,"signals_v7.csv"),index=False,encoding="utf-8-sig")
log(f"\nv7: {len(v7)}条 平均{v7['T1'].mean():.2f}% 胜率{(v7['T1']>0).mean()*100:.1f}%")
for i,(nm,m) in enumerate(L):
    g=v7[v7["层"]==i+1]
    if len(g): log(f"  {nm}: 独立{int(m.sum())}({T1[m].mean():.2f}%) 方案内{len(g)}({g['T1'].mean():.2f}%)")
daily=v7.groupby("日期")["T1"].mean(); nav=np.cumprod(1+daily/100); log(f"理论复利 {(nav.iloc[-1]-1)*100:,.0f}%")
def sim(sig,start=10000.0,start_date=None):
    sig=sig.dropna(subset=["T1"]).sort_values("日期")
    if start_date: sig=sig[sig["日期"]>=start_date]
    sm={str(r["日期"]):r for _,r in sig.iterrows()}
    ds=[x for x in dates if x>=(start_date or dates[0])]
    cash=start; pos=None; eq=[]; tr=[]
    for dt in ds:
        can=pos is None
        if pos is not None and pos["sell_date"]==dt:
            a=pos["shares"]*pos["sell_px"]; f=max(5,a*0.00025)+a*0.0005+a*0.00001; cash+=a-f
            tr.append({**pos,"net":a-f,"sell_fee":f}); pos=None; can=False
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
                pos=dict(buy_date=dt,code=r["股票代码"],name=r["股票名称"],layer=r["层名"],shares=sh,buy_px=px,buy_amt=a,
                         buy_fee=f,sell_date=nxt,sell_px=px*(1+float(r["T1"])/100),close_px=px*(1+cp/100) if np.isfinite(cp) else px)
        eq.append((dt, cash+ (pos["shares"]*(pos["close_px"] if (pos["sell_date"] and dt<pos["sell_date"]) else pos["sell_px"]) if pos else 0)))
    e=pd.DataFrame(eq,columns=["日期","equity"]); final=e["equity"].iloc[-1]; peak=e["equity"].cummax()
    t=pd.DataFrame(tr)
    if len(t): t["cost"]=t["buy_amt"]+t["buy_fee"]; t["pnl"]=t["net"]-t["cost"]; t["ret"]=t["pnl"]/t["cost"]*100
    return final,(e["equity"]/peak-1).min()*100,t,e
f,dd,t,e=sim(v7); log(f"1万元: {f:,.0f}元 ({(f/10000-1)*100:+,.0f}%) {len(t)}笔 回撤{dd:.1f}% 实际单笔{t['ret'].mean():.2f}%")
t.to_csv(os.path.join(OUT,"v7_trades.csv"),index=False,encoding="utf-8-sig"); e.to_csv(os.path.join(OUT,"v7_equity.csv"),index=False,encoding="utf-8-sig")
log("\n多起点:")
rows2=[]
for mth in ["202601","202602","202603","202604","202605","202606","202607","202608","202609","202610","202611"]:
    ds=[x for x in dates if x>=mth+"01"]
    if not ds: continue
    f2,dd2,t2,_=sim(v7,10000,ds[0]); rows2.append(dict(起点=ds[0],期末=round(f2,0),收益=f"{(f2/10000-1)*100:,.0f}%",笔数=len(t2),回撤=f"{dd2:.1f}%"))
    log(f"  {ds[0]}: {f2:,.0f}元 ({(f2/10000-1)*100:+,.0f}%)")
pd.DataFrame(rows2).to_csv(os.path.join(OUT,"v7_multistart.csv"),index=False,encoding="utf-8-sig")
