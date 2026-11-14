# -*- coding: utf-8 -*-
"""8版本 × 3年 每年1万元A股规则模拟"""
import os, json
import numpy as np, pandas as pd
OUT=r"D:/NEW2/v4/out"; BASE=r"D:/NEW/cyb_ma_monitor"
def log(*a): print(*a, flush=True)
# ---------- 2022/2023 数据 ----------
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
DAILY={y:D[D["d"].str.startswith(y)]["d"].unique().tolist() for y in ("2022","2023")}
def prep(y):
    R=pd.read_parquet(os.path.join(OUT,f"detail6_{y}.parquet"))
    R=R.merge(D[["d","code","prev_chg","ret5","ret10","dist60hi","amp20","ma_bull","close","open","volratio"]],left_on=["日期","股票代码"],right_on=["d","code"],how="left")
    return R.dropna(subset=["T1"])
def build22(R,ver):
    C1=(R["ret5"]>10)&(R["dist60hi"]<2)
    MAIN=[("主1",C1&(R["突破分钟"]<=600)&(R["突破时涨幅"]>=4)&(R["突破时涨幅"]<10) if ver=="V4A" else C1&(R["开盘跳空"]>0)&(R["amp20"]>4)),
          ("主2",(R["突破时涨幅"]>=5)&(R["突破时涨幅"]<7)&(R["前20日涨幅"]>30)&(R["买入价"]<20)&(R["开盘跳空"]>2)),
          ("主3",(R["突破时涨幅"]>=5)&(R["突破时涨幅"]<7)&(R["前20日涨幅"]>30)&(R["量能倍数"]>=4)),
          ("主4",(R["突破时涨幅"]>=8)&(R["突破时涨幅"]<10)&(R["开盘跳空"]>2)&(R["前20日涨幅"]>10)&(R["ret5"]<-5)),
          ("主5",(R["突破时涨幅"]>=7)&(R["突破时涨幅"]<10)&(R["前20日涨幅"]<-10)&(R["prev_chg"]<-2)&(R["突破分钟"]<=690)),
          ("主6",(R["突破时涨幅"]>=5)&(R["突破时涨幅"]<7)&(R["前20日涨幅"]>30)&(R["量能倍数"]>=4)&(R["volratio"]>1.2))]
    RB=[("R0",(R["突破时涨幅"]>=5)&(R["突破时涨幅"]<6)&(R["量能倍数"]>=3)&(R["突破时累计成交额"]>=2)&(R["前20日涨幅"]>50)&(R["突破分钟"]<=600)),
        ("R2",(R["突破时涨幅"]>=5)&(R["突破时涨幅"]<7)&(R["量能倍数"]>=4)&(R["突破时累计成交额"]>=1)&(R["前20日涨幅"]>40)),
        ("R4",(R["突破时涨幅"]>=4)&(R["突破时涨幅"]<6)&(R["量能倍数"]>=4)&(R["突破时累计成交额"]>=2)&(R["前20日涨幅"]>30)&(R["开盘跳空"]>0)),
        ("R5",(R["突破时涨幅"]>=5)&(R["突破时涨幅"]<8)&(R["前20日涨幅"]<-10)&(R["沉寂天数"]>=20)&(R["突破分钟"]<=600))]
    N1=("N1",(R["突破时涨幅"]>=5)&(R["突破时涨幅"]<7)&(R["前20日涨幅"]>40)&(R["量能倍数"]>=4)&(R["ret10"]>20))
    X1=("X1",(R["突破时涨幅"]>=8)&(R["突破时涨幅"]<10)&(R["开盘跳空"]>2)&(R["ret5"]>5)&(R["量能倍数"]>=4))
    if ver=="V3":
        return [("R0",(R["突破时涨幅"]>=5)&(R["突破时涨幅"]<6)&(R["量能倍数"]>=3)&(R["突破时累计成交额"]>=2)&(R["前20日涨幅"]>50)&(R["突破分钟"]<=600)),
                ("R1",(R["突破时涨幅"]>=5)&(R["突破时涨幅"]<7)&(R["量能倍数"]>=3)&(R["突破时累计成交额"]>=2)&(R["前20日涨幅"]>50)&(R["突破分钟"]<=600)),
                ("R2",(R["突破时涨幅"]>=5)&(R["突破时涨幅"]<7)&(R["量能倍数"]>=4)&(R["突破时累计成交额"]>=1)&(R["前20日涨幅"]>40)),
                ("R3",(R["突破时涨幅"]>=5)&(R["突破时涨幅"]<7)&(R["量能倍数"]>=4)&(R["突破时累计成交额"]>=2)&(R["前20日涨幅"]>30)&(R["突破分钟"]<=600)),
                ("R4",(R["突破时涨幅"]>=4)&(R["突破时涨幅"]<6)&(R["量能倍数"]>=4)&(R["突破时累计成交额"]>=2)&(R["前20日涨幅"]>30)&(R["开盘跳空"]>0)),
                ("R5",(R["突破时涨幅"]>=5)&(R["突破时涨幅"]<8)&(R["前20日涨幅"]<-10)&(R["沉寂天数"]>=20)&(R["突破分钟"]<=600)),
                ("R6",(R["突破时涨幅"]>=5)&(R["突破时涨幅"]<10)&(R["前20日涨幅"]>=0)&(R["突破分钟"]<=570))]
    if ver=="V4A": return MAIN[:5]
    if ver=="V4B": return MAIN[:5]+[("B1",(R["开盘跳空"]>2)&(R["dist60hi"]<2)&(R["ret5"]>5)&(R["买入价"]<20)),("B2",(R["开盘跳空"]>2)&(R["dist60hi"]<2)&(R["prev_chg"]>3)&(R["volratio"]>1.2))]
    if ver=="V4C": return MAIN
    if ver in ("V5","V6稳健"): return MAIN+RB
    if ver=="V6进取": return MAIN+RB+[X1]
    return MAIN[:2]+[N1]+MAIN[2:]+RB+[X1]
def pick(R,layers):
    used=set(); picks=[]
    for nm,m in layers:
        sub=R[m & (~R["日期"].isin(used))]
        if len(sub)==0: continue
        sel=sub.sort_values(["日期","突破时涨幅"],ascending=[True,False]).groupby("日期").head(1).copy()
        sel["层"]=nm; picks.append(sel); used|=set(sel["日期"])
    return pd.concat(picks,ignore_index=True) if picks else pd.DataFrame()
# ---------- 2026 数据 ----------
f26=pd.read_parquet(os.path.join(OUT,"features_v2.parquet"))
z=np.load(os.path.join(OUT,"cond_matrix_v2.npz"),allow_pickle=True); M=z["M"]; names=[str(x) for x in z["names"]]
idx={n:i for i,n in enumerate(names)}
T1_26=f26["T1"].values.astype(float); chg26=f26["突破时涨幅"].values.astype(float)
day_code26,uniq26=pd.factorize(f26["日期"].astype(str).values); n26=len(uniq26)
st26=np.zeros(n26,int); en26=np.zeros(n26,int)
for i in range(n26):
    rr=np.where(day_code26==i)[0]; st26[i]=rr[0]; en26[i]=rr[-1]+1
def mask26(k):
    m=np.ones(len(f26),bool)
    for c in k: m&=M[:,idx[c]]
    return m
plans=json.load(open(os.path.join(OUT,"final_plans_refined.json"),encoding="utf-8"))
C=[tuple(int(x) for x in l) for l in plans["C_覆盖"]["layers"]]
def cmask(idxs):
    m=np.ones(len(f26),bool)
    for k in idxs: m&=M[:,k]
    return m
def build26(ver):
    MAIN=[(f"主{i+1}",cmask(idxs)) for i,idxs in enumerate(C)]
    RB=[("R0",mask26(["涨5-6","量>=3","额>=2亿","前20>50","10点前"])),("R2",mask26(["涨5-7","量>=4","额>=1亿","前20>40"])),
        ("R4",mask26(["涨4-6","量>=4","额>=2亿","前20>30","高开"])),("R5",mask26(["涨5-8","前20<-10","沉寂>=20","10点前"]))]
    N1=("N1",mask26(["涨5-7","前20>40","量>=4","ret10>20"]))
    X1=("X1",mask26(["涨8-10","高开>2","ret5>5","量>=4"]))
    if ver=="V3":
        return [("R0",mask26(["涨5-6","量>=3","额>=2亿","前20>50","10点前"])),("R1",mask26(["涨5-7","量>=3","额>=2亿","前20>50","10点前"])),
                ("R2",mask26(["涨5-7","量>=4","额>=1亿","前20>40"])),("R3",mask26(["涨5-7","量>=4","额>=2亿","前20>30","10点前"])),
                ("R4",mask26(["涨4-6","量>=4","额>=2亿","前20>30","高开"])),("R5",mask26(["涨5-8","前20<-10","沉寂>=20","10点前"])),
                ("R6",mask26(["涨5-10","前20>0","开盘首分"]))]
    if ver=="V4A":
        A=[(f"主{i+1}",cmask(idxs)) for i,idxs in enumerate(plans["A_精选"]["layers"])]
        return A
    if ver=="V4B": return [(f"层{i+1}",cmask(idxs)) for i,idxs in enumerate(plans["B_平衡"]["layers"])]
    if ver=="V4C": return MAIN
    if ver in ("V5","V6稳健"): return MAIN+RB
    if ver=="V6进取": return MAIN+RB+[X1]
    return MAIN[:2]+[N1]+MAIN[2:]+RB+[X1]
def build26_sig(ver):
    layers=build26(ver)
    score=np.full(len(f26),-1e9)
    for step,(nm,m) in enumerate(layers):
        score=np.maximum(score,np.where(m,-step*1e6+chg26,-1e9))
    rows=[]
    for i in range(n26):
        s,e=st26[i],en26[i]; j=int(np.argmax(score[s:e]))
        if score[s+j]>-1e8: rows.append(dict(日期=uniq26[i],row=s+j))
    out=pd.DataFrame(rows); out["T1"]=T1_26[out["row"]]; out["买入价"]=f26["买入价"].values[out["row"]]
    return out.dropna(subset=["T1"])
# ---------- A股规则模拟 ----------
def sim(sig,dates,frac=1.0,start=10000.0):
    if len(sig)==0: return start,0.0,0,0
    sm={str(r["日期"]):r for _,r in sig.iterrows()}
    cash=start; pos=None; eq=[]; bought=0; missed=0
    for k,dt in enumerate(dates):
        can=pos is None
        if pos is not None and pos["sell_date"]==dt:
            a=pos["shares"]*pos["sell_px"]; f=max(5,a*0.00025)+a*0.0005+a*0.00001; cash+=a-f; pos=None; can=False
        if can and dt in sm and pos is None:
            r=sm[dt]; px=float(r["买入价"]); use=cash*frac
            sh=int(use//(px*100))*100
            while sh>=100:
                a=sh*px; f=max(5,a*0.00025)+a*0.00001
                if a+f<=cash+1e-9: break
                sh-=100
            if sh>=100:
                a=sh*px; f=max(5,a*0.00025)+a*0.00001; cash-=a+f; bought+=1
                nxt=dates[k+1] if k+1<len(dates) else None
                pos=dict(shares=sh,sell_date=nxt,sell_px=px*(1+float(r["T1"])/100))
        elif dt in sm and not can: missed+=1
        eq.append((dt, cash+(pos["shares"]*pos["sell_px"] if pos else 0)))
    e=pd.DataFrame(eq,columns=["日期","equity"]); peak=e["equity"].cummax()
    return e["equity"].iloc[-1],(e["equity"]/peak-1).min()*100,bought,missed
vers=["V3","V4A","V4B","V4C","V5","V6稳健","V6进取","V7"]
rows=[]
for y in ("2022","2023"):
    R=prep(y); dates=DAILY[y]
    for ver in vers:
        sig=pick(R,build22(R,ver))
        for frac,lab in [(1.0,"全仓"),(0.5,"半仓")]:
            f,dd,b,mi=sim(sig,dates,frac)
            rows.append(dict(年=y,版本=ver,仓位=lab,信号=len(sig),实买=b,错过=mi,终值=round(f),收益率=round((f/10000-1)*100,1),回撤=round(dd,1)))
            log(f"{y} {ver} {lab}: {len(sig)}信号 实买{b} 错过{mi} -> {f:,.0f} ({dd:.1f}%)")
# 2026
d26=pd.read_parquet(os.path.join(BASE,"cache","ohlcv_cyb.parquet")); d26["d"]=d26["d"].astype(str)
dates26=sorted([x for x in d26["d"].unique() if x>="20260105" and x<="20261113"])
for ver in vers:
    sig=build26_sig(ver)
    for frac,lab in [(1.0,"全仓"),(0.5,"半仓")]:
        f,dd,b,mi=sim(sig,dates26,frac)
        rows.append(dict(年="2026",版本=ver,仓位=lab,信号=len(sig),实买=b,错过=mi,终值=round(f),收益率=round((f/10000-1)*100,1),回撤=round(dd,1)))
        log(f"2026 {ver} {lab}: {len(sig)}信号 实买{b} 错过{mi} -> {f:,.0f} ({dd:.1f}%)")
pd.DataFrame(rows).to_csv(os.path.join(OUT,"all_versions_nav.csv"),index=False,encoding="utf-8-sig")
log("\n已保存 all_versions_nav.csv")
