# -*- coding: utf-8 -*-
"""v7 交付：v6 + N1 新规则（稳健版去X1 / 进取版含X1）"""
import os, json, ast
import numpy as np, pandas as pd
from openpyxl.utils import get_column_letter

OUT=r"D:\NEW2\v4\out"; FINAL=r"D:\NEW2\out"; ORIG=r"D:\NEW\cyb_ma_monitor"
os.makedirs(FINAL,exist_ok=True)
def log(*a): print(*a, flush=True)
dia=pd.read_parquet(os.path.join("D:\\NEW\\cyb_ma_monitor","cache","ohlcv_cyb.parquet")); dia["d"]=dia["d"].astype(str)
dates=sorted([x for x in dia["d"].unique() if x>="20260105" and x<="20261113"]); date_pos={x:i for i,x in enumerate(dates)}
d=pd.read_parquet(os.path.join(OUT,"features_v2.parquet"))
z=np.load(os.path.join(OUT,"cond_matrix_v2.npz"),allow_pickle=True); M=z["M"]; names=[str(x) for x in z["names"]]
idx={n:i for i,n in enumerate(names)}
v7=pd.read_csv(os.path.join(OUT,"signals_v7.csv"),dtype={"日期":str,"股票代码":str})
v7["股票代码"]=v7["股票代码"].str.zfill(6); v7["T1"]=pd.to_numeric(v7["T1"],errors="coerce")
v7["月"]=v7["日期"].str[:6]
adv=v7.copy()
steady=v7[v7["层名"]!="X1"].copy()
log(f"v7 稳健版 {len(steady)} 条; 进取版 {len(adv)} 条")

def metrics(df):
    daily=df.groupby("日期")["T1"].mean(); r=daily/100; nav=np.cumprod(1+r); peak=np.maximum.accumulate(nav)
    h1=df[df["日期"]<"20260701"]; h2=df[df["日期"]>="20260701"]; mo=df.groupby("月")["T1"].mean()
    return dict(n=len(df),days=df["日期"].nunique(),mean=df["T1"].mean(),th=(nav.iloc[-1]-1)*100,
                win=(df["T1"]>0).mean()*100,dd=(nav/peak-1).min()*100,
                h1=h1["T1"].mean(),h2=h2["T1"].mean(),npos=int((mo>0).sum()),nmon=len(mo))
def sim(sig,start=10000.0,start_date=None):
    sig=sig.dropna(subset=["T1"]).sort_values("日期")
    if start_date: sig=sig[sig["日期"]>=start_date]
    sm={str(r["日期"]):r for _,r in sig.iterrows()}
    ds=[x for x in dates if x>=(start_date or dates[0])]
    cash=start; pos=None; eq=[]; tr=[]
    for dt in ds:
        can=pos is None
        if pos is not None and pos["sell_date"]==dt:
            a=pos["shares"]*pos["sell_px"]; f=max(5,a*0.00025)+a*0.0005+a*0.00001
            cash+=a-f; tr.append({**pos,"net":a-f,"sell_fee":f}); pos=None; can=False
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
                pos=dict(buy_date=dt,code=r["股票代码"],name=r["股票名称"],layer=r["层名"],shares=sh,buy_px=px,
                         buy_amt=a,buy_fee=f,sell_date=nxt,sell_px=px*(1+float(r["T1"])/100),
                         close_px=px*(1+cp/100) if np.isfinite(cp) else px)
        eq.append((dt, cash+ (pos["shares"]*(pos["close_px"] if (pos["sell_date"] and dt<pos["sell_date"]) else pos["sell_px"]) if pos else 0)))
    e=pd.DataFrame(eq,columns=["日期","equity"]); final=e["equity"].iloc[-1]; peak=e["equity"].cummax()
    t=pd.DataFrame(tr)
    if len(t):
        t["cost"]=t["buy_amt"]+t["buy_fee"]; t["pnl"]=t["net"]-t["cost"]; t["ret"]=t["pnl"]/t["cost"]*100
    return final,(e["equity"]/peak-1).min()*100,t,e
fs,dds,ts,es=sim(steady); fa,dda,ta,ea=sim(adv)
log(f"稳健版 1万->{fs:,.0f} ({len(ts)}笔, 回撤{dds:.1f}%); 进取版 1万->{fa:,.0f} ({len(ta)}笔, 回撤{dda:.1f}%)")

OV=[]
for label,df,f,dd,n in [("v7 稳健版（推荐）",steady,fs,dds,len(ts)),("v7 进取版（含机会信号）",adv,fa,dda,len(ta))]:
    m=metrics(df); OV.append({"方案":label,"信号数":m["n"],"平均T+1%":round(m["mean"],2),"胜率%":round(m["win"],1),
        "理论上限复利%":round(m["th"],0),"上半年均%":round(m["h1"],2),"下半年均%":round(m["h2"],2),
        "月正数":f"{m['npos']}/{m['nmon']}","1万元严格模拟":round(f,0),"模拟笔数":n,"模拟回撤%":round(dd,1)})
OV.append({"方案":"v6 稳健版","信号数":114,"平均T+1%":6.35,"胜率%":77.2,"理论上限复利%":70334,"上半年均%":5.99,
           "下半年均%":6.88,"月正数":"11/11","1万元严格模拟":726155,"模拟笔数":72,"模拟回撤%":-23.0})
OV.append({"方案":"v6 进取版","信号数":121,"平均T+1%":6.41,"胜率%":76.9,"理论上限复利%":112199,"上半年均%":6.45,
           "下半年均%":6.36,"月正数":"11/11","1万元严格模拟":1105649,"模拟笔数":75,"模拟回撤%":-20.8})
OV.append({"方案":"v5","信号数":115,"平均T+1%":6.26,"胜率%":76.5,"理论上限复利%":67326,"上半年均%":5.84,
           "下半年均%":6.88,"月正数":"11/11","1万元严格模拟":654586,"模拟笔数":72,"模拟回撤%":-23.0})
OV.append({"方案":"v4C 单用","信号数":93,"平均T+1%":6.65,"胜率%":77.4,"理论上限复利%":26295,"上半年均%":6.49,
           "下半年均%":6.85,"月正数":"11/11","1万元严格模拟":431507,"模拟笔数":60,"模拟回撤%":-23.0})
OV=pd.DataFrame(OV)

explain={"涨3-5":"突破时涨幅3-5%","涨4-6":"突破时涨幅4-6%","涨5-6":"突破时涨幅5-6%","涨5-7":"突破时涨幅5-7%","涨5-8":"突破时涨幅5-8%",
"涨6-8":"突破时涨幅6-8%","涨7-10":"突破时涨幅7-10%","涨8-10":"突破时涨幅8-10%","涨5-10":"突破时涨幅5-10%","涨4-10":"突破时涨幅4-10%",
"量>=1.5":"量能倍数≥1.5","量>=2":"量能倍数≥2","量>=3":"量能倍数≥3","量>=4":"量能倍数≥4","量>=5":"量能倍数≥5","量>=6":"量能倍数≥6",
"额>=0.5亿":"成交额≥0.5亿","额>=1亿":"成交额≥1亿","额>=2亿":"成交额≥2亿","额>=3亿":"成交额≥3亿","额>=5亿":"成交额≥5亿",
"前20>50":"前20日涨幅>50%","前20>40":"前20日涨幅>40%","前20>30":"前20日涨幅>30%","前20>20":"前20日涨幅>20%","前20>10":"前20日涨幅>10%",
"前20>0":"前20日涨幅>0","前20<-10":"前20日跌幅>10%","前20<-20":"前20日跌幅>20%","沉寂>=10":"沉寂≥10","沉寂>=20":"沉寂≥20",
"沉寂>=30":"沉寂≥30","沉寂>=50":"沉寂≥50","开盘首分":"开盘首分钟突破","10点前":"10:00前突破","10点半前":"10:30前","上午":"上午突破",
"高开":"高开","高开>1":"高开>1%","高开>2":"高开>2%","低开":"低开","低开<-1":"低开<-1%","昨涨>0":"昨日上涨","昨涨>3":"昨日涨>3%",
"昨跌<-2":"昨日跌>2%","ret5>0":"前5日涨幅>0","ret5>5":"前5日涨幅>5%","ret5>10":"前5日涨幅>10%","ret5<-5":"前5日跌幅>5%",
"ret10>10":"前10日涨幅>10%","ret10>20":"前10日涨幅>20%","ret20>20":"前20日涨幅>20%","ret60>30":"前60日涨幅>30%",
"多头排列":"MA5>MA10>MA20>MA60","MA20上翘":"MA20上翘","MA60上翘":"MA60上翘","近60日高":"距60日最高价<3%","远60日高":"距60日最高价>10%",
"远250日高":"距250日最高价>30%","量比>1.2":"5日均量/20日均量>1.2","量比>1.5":">1.5","振幅20>4":"20日平均振幅>4%","振幅20>6":">6%",
"昨日涨停":"昨日涨停","昨日连板":"昨日连板","低价<20":"买入价<20元","高价>50":"买入价>50元","昨涨停>=50":"昨日涨停≥50家",
"昨涨停>=80":"≥80家","昨上涨占比>50":"昨日上涨占比>50%","昨上涨占比>60":">60%","昨创业均涨>0":"昨日创业板均涨>0",
"昨创业均涨>1":">1%","昨弱势(占比<30)":"昨日上涨占比<30%"}
def rule_cn(idxs): return " + ".join(explain.get(names[k],names[k]) for k in idxs)
plans=json.load(open(os.path.join(OUT,"final_plans_refined.json"),encoding="utf-8"))
C_layers=[tuple(int(x) for x in l) for l in plans["C_覆盖"]["layers"]]
def cmask(idxs):
    m=np.ones(len(d),bool)
    for k in idxs: m&=M[:,k]
    return m
rr=[]
for i,idxs in enumerate(C_layers):
    mm=cmask(idxs)
    rr.append({"优先级":f"主{i+1}","来源":"v4C","规则(可实时判断)":rule_cn(idxs),"独立信号数":int(mm.sum()),
               "独立均值%":round(d["T1"].values[mm].mean(),2),"建议":"主方案"})
    if i==1:
        mm=cmask([idx["涨5-7"],idx["前20>40"],idx["量>=4"],idx["ret10>20"]])
        rr.append({"优先级":"N1（新增）","来源":"v2搜索","规则(可实时判断)":"突破时涨幅5-7% + 前20日涨幅>40% + 量能倍数≥4 + 前10日涨幅>20%",
                   "独立信号数":int(mm.sum()),"独立均值%":round(d["T1"].values[mm].mean(),2),"建议":"主方案（本轮新增）"})
v3r=[("R0","涨5-6% + 量能≥3 + 成交额≥2亿 + 前20日涨幅>50% + 10:00前"),
     ("R2","涨5-7% + 量能≥4 + 成交额≥1亿 + 前20日涨幅>40%"),
     ("R4","涨4-6% + 量能≥4 + 成交额≥2亿 + 前20日涨幅>30% + 高开"),
     ("R5","涨5-8% + 前20日跌幅>10% + 沉寂≥20 + 10:00前")]
for k,desc in v3r:
    g=v7[v7["层名"]==k]; rr.append({"优先级":f"补充{k}","来源":"v3","规则(可实时判断)":desc,
                                   "独立信号数":len(g),"独立均值%":round(g["T1"].mean(),2) if len(g) else np.nan,"建议":"主方案"})
rr.append({"优先级":"机会X1","来源":"扩展","规则(可实时判断)":"突破时涨幅8-10% + 高开>2% + 前5日涨幅>5% + 量能倍数≥4",
           "独立信号数":len(v7[v7["层名"]=="X1"]),"独立均值%":round(v7[v7["层名"]=="X1"]["T1"].mean(),2),"建议":"进取版才做（单条依赖）"})
RR=pd.DataFrame(rr)

dd=d.iloc[v7["row"].astype(int)].reset_index(drop=True)
sig=pd.DataFrame({
 "建议":np.where(v7["层名"].values=="X1","机会信号(进取版)","主方案"),
 "日期":v7["日期"].values,"层名":v7["层名"].values,"股票代码":v7["股票代码"].values,"股票名称":v7["股票名称"].values,
 "突破时刻":dd["突破时刻"].values,"突破时涨幅%":dd["突破时涨幅"].values,"买入价(元)":dd["买入价"].values,
 "量能倍数":dd["量能倍数"].values,"累计成交额(亿元)":dd["突破时累计成交额(亿元)"].values,
 "开盘跳空%":dd["开盘跳空"].values,"沉寂天数":dd["沉寂天数"].values,"前20日涨幅%":dd["前20日涨幅"].values,
 "T+1收益%":v7["T1"].values})
sig=sig.sort_values("日期")
mo=[]
for mth,g in v7.groupby("月"):
    mo.append({"月份":int(mth),"信号数":len(g),"平均T+1%":round(g["T1"].mean(),2),"胜率%":round((g["T1"]>0).mean()*100,1)})
MO=pd.DataFrame(mo)
ms=[]
for label,df in [("稳健版",steady),("进取版",adv)]:
    for mth in ["202601","202602","202603","202604","202605","202606","202607","202608","202609","202610","202611"]:
        ds=[x for x in dates if x>=mth+"01"]
        if not ds: continue
        f,dd,t,_=sim(df,10000,ds[0])
        ms.append({"版本":label,"起点":ds[0],"期末":round(f,0),"收益":f"{(f/10000-1)*100:,.0f}%","笔数":len(t),"回撤":f"{dd:.1f}%"})
MS=pd.DataFrame(ms)
T=ta[["buy_date","code","name","layer","buy_px","shares","buy_amt","sell_date","sell_px","net","cost","pnl","ret"]].copy()
T.columns=["买入日","代码","名称","层","买入价","股数","买入金额","卖出日","卖出价","卖出净额","总成本","盈亏","收益率%"]
used=set(ta["buy_date"]); SK=v7[~v7["日期"].isin(used)][["日期","层名","股票代码","股票名称","买入价","T1"]].copy()
SK.columns=["日期","层名","股票代码","股票名称","买入价(元)","T+1收益%"]
DESC=pd.DataFrame([
["突破时涨幅%","(突破分钟收盘价/昨收-1)×100","1分钟K线+日线","突破时刻实时"],
["量能倍数","(累计量/时间进度)/前5日均量","1分钟K线+日线","突破时刻实时"],
["前10日涨幅/前20日涨幅","(T-1收盘/T-1-N日收盘-1)×100","日线前复权","T-1已知"],
["距60日最高价/量比/MA排列","T-1及之前日线","日线前复权","T-1已知"],
["T+1收益%","(次日收盘/买入价-1)×100","日线前复权","仅评估"]],columns=["字段","公式","来源","时点"])
AUD=pd.DataFrame([["选股字段","突破时刻实时 + T-1日线（特征已按股票序列修复，无停牌NaN）","通过"],
["禁用字段","收盘站上/分类/当日收盘涨幅/T+1等未入规则","通过"],
["特征修复","ret20缺失56→0, dist60hi缺失535→0, ma/volratio 190→0","本轮完成"],
["买入价严格复核","v4C已做下一分钟开盘价复核差异<0.03pp","通过"],
["N1新规则","11条 10.34%，去最大1条9.14%，7个月全正，严格邻域8-11%","通过"],
["机会X1","7条中1条+32.49%贡献大部分增量，去极值后3.30%","标注风险"]],columns=["检查项","口径","结论"])
rob=[["分月正收益","11/11",""],
["N1 独立","11条 10.34% 胜81.8%",""],["N1 去最大1/2","9.14% / 7.69%",""],["N1 参数邻域","前20>50:11.31%, 量>=5:8.23%, 涨5-6:8.35%, 放宽ret10>10降到5.26%",""],
["进取版去X1","稳健版为去X1版本",""]]
for q,(a,b) in {1:("20260101","20260401"),2:("20260401","20260701"),3:("20260701","20261001"),4:("20261001","20261301")}.items():
    s=v7[(v7["日期"]>=a)&(v7["日期"]<b)]; rob.append([f"Q{q}",f"{len(s)}条 均{s['T1'].mean():.2f}%",""])
ROB=pd.DataFrame(rob,columns=["验证项","结果","说明"])

xlsx=os.path.join(FINAL,"_最优方案v7_含新规则N1.xlsx")
with pd.ExcelWriter(xlsx,engine="openpyxl") as w:
    OV.to_excel(w,sheet_name="方案总览",index=False)
    RR.to_excel(w,sheet_name="规则说明",index=False)
    sig.to_excel(w,sheet_name="信号明细122条",index=False)
    MO.to_excel(w,sheet_name="分月表现",index=False)
    MS.to_excel(w,sheet_name="多起点模拟",index=False)
    T.to_excel(w,sheet_name="1万元交易明细_进取",index=False)
    SK.to_excel(w,sheet_name="跳过信号_可补做",index=False)
    DESC.to_excel(w,sheet_name="指标口径说明",index=False)
    AUD.to_excel(w,sheet_name="无未来函数审计",index=False)
    ROB.to_excel(w,sheet_name="稳健性验证",index=False)
    widths={"方案总览":[24,9,10,9,14,11,11,9,16,9,11],"规则说明":[12,8,80,11,11,30],
            "信号明细122条":[16,10,7,10,12,10,12,12,10,18,10,10,12,11],"分月表现":[9,9,11,9],
            "多起点模拟":[8,10,10,10,7,9],"1万元交易明细_进取":[10,10,12,8,10,9,11,10,10,11,11,11,9],
            "跳过信号_可补做":[10,8,10,12,11,11],"指标口径说明":[22,45,20,12],
            "无未来函数审计":[14,60,18],"稳健性验证":[18,60,25]}
    for sh,wd in widths.items():
        ws=w.sheets[sh]
        for i,ww in enumerate(wd): ws.column_dimensions[get_column_letter(i+1)].width=ww
        ws.freeze_panes="A2"
log("Excel:",xlsx)

rep=[]
rep.append("# v7：特征修复 + 新规则 N1\n")
rep.append("## 本轮做了什么\n")
rep.append("1. 修复日线特征：改为按股票序列计算（严格截至 T-1），ret20 缺失 56→0、dist60hi 缺失 535→0、ma/volratio 缺失 190→0。")
rep.append("2. 用修复后的矩阵重跑 1-4 条件搜索，发现旧版被 NaN 漏掉的新规则 N1：涨5-7% + 前20日涨幅>40% + 量能≥4 + 前10日涨幅>20%。")
rep.append("3. N1 独立 11 条、均值 10.34%、胜率 81.8%、7 个月全正、去最大 1 条后 9.14%，插入主2 后作为第 3 优先级。")
rep.append("\n## 结果\n")
rep.append(f"- **v7 稳健版（推荐）：{len(steady)} 条**，平均 {metrics(steady)['mean']:.2f}%，1 万元严格模拟 **{fs:,.0f} 元（+{(fs/10000-1)*100:,.0f}%）**，回撤 {dds:.1f}%。")
rep.append(f"- **v7 进取版：{len(adv)} 条**，平均 {metrics(adv)['mean']:.2f}%，1 万元严格模拟 **{fa:,.0f} 元（+{(fa/10000-1)*100:,.0f}%）**，回撤 {dda:.1f}%。")
rep.append(f"- 对比：v6 稳健版 72.62 万、v6 进取版 110.56 万、v5 65.46 万、v4C 43.15 万、v3 20.03 万。")
rep.append("\n## N1 新规则稳健性\n")
rep.append("| 检查 | 结果 |")
rep.append("|---|---|")
rep.append("| 独立样本 | 11 条，均值 10.34%，胜率 81.8% |")
rep.append("| 去最大 1/2 条 | 9.14% / 7.69% |")
rep.append("| 分月 | 202601 5.94、202603 5.17、202604 7.02、202605 16.36、202606 18.81、202608 8.24、202610 20.46 |")
rep.append("| 严格邻域 | 前20>50 11.31%、量>=5 8.23%、涨5-6 8.35% |")
rep.append("| 放宽邻域 | ret10>10 降到 5.26%、量>=3 降到 3.44%（说明阈值不能放宽） |")
rep.append("\n## 风险\n")
rep.append("- 仍是 2026 单年样本内结果；N1 只有 11 条，虽然多方向稳健，但样本小。")
rep.append("- 机会信号 X1 仍只建议小仓位：7 条中 1 条 +32.49% 贡献大部分增量，去极值后 6 条仅 3.30%。")
rep.append("- 最大回撤约 -21%，单笔 -13% 级别亏损需接受。")
open(os.path.join(FINAL,"_方案v7_报告.md"),"w",encoding="utf-8").write("\n".join(rep))
log("报告:",os.path.join(FINAL,"_方案v7_报告.md"))