# -*- coding: utf-8 -*-
"""v4 阶段13：V3 方案在严格 A 股规则下的 1 万元模拟（同口径对比）"""
import os
import numpy as np, pandas as pd

OUT = r"D:\NEW2\v4\out"
BASE = r"D:\NEW\cyb_ma_monitor"
def log(*a): print(*a, flush=True)

dia = pd.read_parquet(os.path.join(BASE,"cache","ohlcv_cyb.parquet"))
dia["d"] = dia["d"].astype(str)
dates = sorted([d for d in dia["d"].unique() if d >= "20260105" and d <= "20261113"])
date_pos = {d:i for i,d in enumerate(dates)}
d = pd.read_parquet(os.path.join(OUT,"features.parquet"))

def topn(p, nn, cols, asc):
    if len(p)==0: return p.copy()
    return p.sort_values(["日期"]+cols, ascending=[True]+asc).groupby("日期").head(nn).copy()

rules = [
 ("R0",(d["突破时涨幅"]>=5)&(d["突破时涨幅"]<6)&(d["量能倍数"]>=3)&(d["突破时累计成交额(亿元)"]>=2)&(d["前20日涨幅"]>50)&(d["突破分钟"]<=600)),
 ("R1",(d["突破时涨幅"]>=5)&(d["突破时涨幅"]<7)&(d["量能倍数"]>=3)&(d["突破时累计成交额(亿元)"]>=2)&(d["前20日涨幅"]>50)&(d["突破分钟"]<=600)),
 ("R2",(d["突破时涨幅"]>=5)&(d["突破时涨幅"]<7)&(d["量能倍数"]>=4)&(d["突破时累计成交额(亿元)"]>=1)&(d["前20日涨幅"]>40)),
 ("R3",(d["突破时涨幅"]>=5)&(d["突破时涨幅"]<7)&(d["量能倍数"]>=4)&(d["突破时累计成交额(亿元)"]>=2)&(d["前20日涨幅"]>30)&(d["突破分钟"]<=600)),
 ("R4",(d["突破时涨幅"]>=4)&(d["突破时涨幅"]<6)&(d["量能倍数"]>=4)&(d["突破时累计成交额(亿元)"]>=2)&(d["前20日涨幅"]>30)&(d["开盘跳空"]>0)),
 ("R5",(d["突破时涨幅"]>=5)&(d["突破时涨幅"]<8)&(d["前20日涨幅"]<-10)&(d["沉寂天数"]>=20)&(d["突破分钟"]<=600)),
 ("R6",(d["突破时涨幅"]>=5)&(d["突破时涨幅"]<10)&(d["前20日涨幅"]>=0)&(d["突破分钟"]<=570)),
]
out=[]; used=set()
for k,c in rules:
    s=d[c & (~d["日期"].isin(used))]
    if len(s)==0: continue
    sel=topn(s,1,["突破时涨幅"],[False]); sel=sel.copy(); sel["层号"]=k
    out.append(sel); used |= set(sel["日期"].tolist())
v3=pd.concat(out, ignore_index=True)
v3n = v3[v3["层号"]!="R6"]
log(f"v3 全量 {len(v3)} 条, 去R6 {len(v3n)} 条")

def simulate(sig, start=10000.0, commission=0.00025, min_comm=5.0, stamp=0.0005, transfer=0.00001, name=""):
    sig = sig.copy()
    sig["T1"] = pd.to_numeric(sig["T1"], errors="coerce")
    sig["日期"] = sig["日期"].astype(str)
    sig = sig.dropna(subset=["T1"]).sort_values("日期")
    sig_map = {str(r["日期"]): r for _, r in sig.iterrows()}
    cash=float(start); pos=None; eq=[]; trades=[]
    skipped_busy=0; skipped_unaff=0
    for dt in dates:
        can_buy = pos is None
        if pos is not None and pos["sell_date"]==dt:
            amt=pos["shares"]*pos["sell_px"]; fee=max(min_comm,amt*commission)+amt*stamp+amt*transfer
            cash += amt-fee
            trades.append({**pos,"sell_fee":fee,"net":amt-fee,"sell_day":dt})
            pos=None; can_buy=False
        if can_buy and dt in sig_map and pos is None:
            r=sig_map[dt]; px=float(r["买入价"])
            shares=int(cash//(px*100))*100
            while shares>=100:
                amt=shares*px; fee=max(min_comm,amt*commission)+amt*transfer
                if amt+fee<=cash+1e-9: break
                shares-=100
            if shares<100:
                skipped_unaff+=1
            else:
                amt=shares*px; fee=max(min_comm,amt*commission)+amt*transfer
                cash-=amt+fee
                nxt=dates[date_pos[dt]+1] if date_pos[dt]+1<len(dates) else None
                cp=float(r["当日收盘涨幅"]) if np.isfinite(r.get("当日收盘涨幅",np.nan)) else np.nan
                pos=dict(buy_date=dt,code=r["股票代码"],name=r["股票名称"],layer=r["层号"],shares=shares,
                         buy_px=px,buy_amt=amt,buy_fee=fee,sell_date=nxt,sell_px=px*(1+float(r["T1"])/100),
                         close_px=px*(1+cp/100) if np.isfinite(cp) else px)
        elif dt in sig_map and not can_buy:
            skipped_busy+=1
        if pos is not None:
            if dt==pos["buy_date"] or (pos["sell_date"] is not None and dt<pos["sell_date"]):
                eqv=cash+pos["shares"]*pos["close_px"]
            else:
                eqv=cash+pos["shares"]*pos["sell_px"]
        else:
            eqv=cash
        eq.append((dt,eqv))
    eqdf=pd.DataFrame(eq,columns=["日期","equity"])
    final=float(eqdf["equity"].iloc[-1]); peak=eqdf["equity"].cummax(); dd=float((eqdf["equity"]/peak-1).min()*100)
    t=pd.DataFrame(trades)
    if len(t):
        t["cost"]=t["buy_amt"]+t["buy_fee"]; t["pnl"]=t["net"]-t["cost"]; t["ret"]=t["pnl"]/t["cost"]*100
    th=np.prod(1+sig["T1"].values/100)-1
    log(f"{name}: 信号{len(sig)} 做{len(trades)}笔 跳过(占用{skipped_busy}/买不起{skipped_unaff}) "
        f"期末{final:,.0f}元 ({(final/start-1)*100:,.0f}%) 回撤{dd:.1f}% 理论{th*100:,.0f}% "
        f"实际单笔均{t['ret'].mean():.2f}% 费用{t['buy_fee'].sum()+t['sell_fee'].sum():,.0f}")
    return dict(name=name, n=len(sig), trades=len(trades), final=final, ret=(final/start-1)*100,
                dd=dd, theory=th*100, mean=t["ret"].mean() if len(t) else np.nan, cs=skipped_busy, cu=skipped_unaff)

res=[]
res.append(simulate(v3, name="v3 全量(117)"))
res.append(simulate(v3n, name="v3 去R6(65)"))
for plan in ["C_覆盖","B_平衡","A_精选"]:
    s = pd.read_csv(os.path.join(OUT,f"signals_{plan}.csv"), dtype={"日期":str,"股票代码":str})
    s["股票代码"]=s["股票代码"].str.zfill(6)
    res.append(simulate(s, name=f"v4 {plan}"))
log("\n"+"="*80)
for r in res:
    log(f"{r['name']}: {r['final']:,.0f} 元 ({r['ret']:+.0f}%), 做{r['trades']}笔, 回撤{r['dd']:.1f}%, 实际单笔{r['mean']:.2f}%")
