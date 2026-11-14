# -*- coding: utf-8 -*-
"""v4 阶段1：全量特征 + 原子条件库（严格截至 T-1 或 T 时刻）"""
import os, json
import numpy as np, pandas as pd

BASE = r"D:\NEW\cyb_ma_monitor"
OUT  = r"D:\NEW2\v4\out"
os.makedirs(OUT, exist_ok=True)
XLS = os.path.join(BASE, "_2026盘中上穿明细_全列对齐版.xlsx")
DIA = os.path.join(BASE, "cache", "ohlcv_cyb.parquet")
MS  = os.path.join(BASE, "_market_sentiment.csv")

def log(*a): print(*a, flush=True)

# ---------------- 基础池 ----------------
df = pd.read_excel(XLS, sheet_name="全部20920条")
df["股票代码"] = df["股票代码"].astype(str).str.zfill(6)
df["日期"] = df["日期"].astype(str)
df["T1"] = pd.to_numeric(df["T+1收益"], errors="coerce")
df["突破分钟"] = df["突破时刻"].apply(
    lambda t: int(str(t)[:2])*60+int(str(t)[3:5]) if isinstance(t,str) and ":" in str(t) else np.nan)
df["月"] = df["日期"].str[:6]
df["半年"] = np.where(df["日期"] < "20260701", "H1", "H2")

# ---------------- 日线宽表 ----------------
dia = pd.read_parquet(DIA).sort_values(["code","d"]).reset_index(drop=True)
dia["d"] = dia["d"].astype(str)
C = dia.pivot(index="d", columns="code", values="c").sort_index()
O = dia.pivot(index="d", columns="code", values="o").sort_index()
H = dia.pivot(index="d", columns="code", values="h").sort_index()
L = dia.pivot(index="d", columns="code", values="l").sort_index()
V = dia.pivot(index="d", columns="code", values="v").sort_index()
dates = C.index.astype(str)
log(f"日线宽表: {C.shape}")

# 涨停买不到（统一 20%，创业板）
prev_close = C.shift(1)
lim = (prev_close*1.2).round(2)
# 候选池 T-1 收盘
pc = prev_close.stack().rename("prev_close")
pc.index.names = ["日期","股票代码"]
df = df.merge(pc, left_on=["日期","股票代码"], right_index=True, how="left")
df["涨停买不到"] = ((df["买入价"] >= (df["prev_close"]*1.2).round(2)*0.995) & df["prev_close"].notna()).astype(int)
log(f"涨停买不到: {int(df['涨停买不到'].sum())}")
d = df[(df["涨停买不到"]==0) & df["T1"].notna()].copy()
log(f"可用池: {len(d)} 条, {d['日期'].nunique()} 天")

# ---------------- 宽表特征（全部 shift 1 = 截至 T-1） ----------------
feat = {}
feat["ret1"]  = C.pct_change(1, fill_method=None)
feat["ret3"]  = C.pct_change(3, fill_method=None)
feat["ret5"]  = C.pct_change(5, fill_method=None)
feat["ret10"] = C.pct_change(10, fill_method=None)
feat["ret20"] = C.pct_change(20, fill_method=None)
feat["ret60"] = C.pct_change(60, fill_method=None)
amp = (H-L)/C.shift(1)
feat["amp5"]  = amp.rolling(5).mean()
feat["amp20"] = amp.rolling(20).mean()
feat["amp60"] = amp.rolling(60).mean()
for p in (5,10,20,60,120,250):
    feat[f"ma{p}"] = C.rolling(p).mean()
feat["ma_bull"] = ((feat["ma5"]>feat["ma10"]) & (feat["ma10"]>feat["ma20"]) & (feat["ma20"]>feat["ma60"])).astype(float)
feat["ma20_slope"] = feat["ma20"]/feat["ma20"].shift(5)-1
feat["ma60_slope"] = feat["ma60"]/feat["ma60"].shift(5)-1
feat["dist60hi"]  = C/feat["ma60"].rolling(1).max()*0  # placeholder
feat["dist60hi"]  = C/C.rolling(60).max()-1
feat["dist250hi"] = C/C.rolling(250).max()-1
feat["volratio"]  = V.rolling(5).mean()/V.rolling(20).mean()
feat["prev_chg"]  = C.pct_change(1, fill_method=None)
feat["prev_amp"]  = amp
# 连板数
is_lim = (C >= (prev_close*1.2).round(2)*0.995) & C.notna() & prev_close.notna()
cs = is_lim.cumsum()
reset = cs.where(~is_lim).ffill().fillna(0)
feat["limit_streak"] = cs - reset

# 全部 shift(1) 表示"截至 T-1"
F = {}
for name, w in feat.items():
    s = w.shift(1).stack().rename(name)
    s.index.names = ["日期","股票代码"]
    F[name] = s
FEAT = pd.DataFrame(F)
log(f"特征宽表 stack: {FEAT.shape}")
d = d.merge(FEAT, left_on=["日期","股票代码"], right_index=True, how="left")
log(f"merge 后: {len(d)} 行, 重名列: {[c for c in d.columns if c.endswith('_x') or c.endswith('_y')]}")

# ---------------- 市场情绪（T-1） ----------------
ms = pd.read_csv(MS)
ms["日期"] = ms["日期"].astype(str)
ms = ms.sort_values("日期").reset_index(drop=True)
ms_cols = ["limit_up","limit_down","up_ratio","avg_ret","strong5","big_move","lianban"]
ms_shift = ms[["日期"]+ms_cols].copy()
for c in ms_cols: ms_shift[c] = ms_shift[c].shift(1)
ms_map = ms_shift.set_index("日期")
d = d.merge(ms_map, left_on="日期", right_index=True, how="left", suffixes=("","_mkt"))
log(f"市场特征 merge 后: {len(d)} 行")
for c in ms_cols: log(f"  {c}: 缺失 {d[c].isna().sum()}")

# ---------------- 原子条件库 ----------------
conds = {}
def add(name, mask, field, op, thr, timing):
    conds[name] = {"mask": mask.values if hasattr(mask,"values") else mask,
                   "field": field, "op": op, "thr": thr, "timing": timing}

# 涨幅（T）
for lo,hi in [(3,5),(4,6),(5,6),(5,7),(5,8),(6,8),(7,10),(8,10),(5,10),(4,10)]:
    add(f"涨{lo}-{hi}", (d["突破时涨幅"]>=lo)&(d["突破时涨幅"]<hi), "突破时涨幅", "区间", [lo,hi], "T")
# 量能（T）
for t in (1.5,2,3,4,5,6):
    add(f"量>={t}", d["量能倍数"]>=t, "量能倍数", ">=", t, "T")
# 成交额（T）
for t in (0.5,1,2,3,5):
    add(f"额>={t}亿", d["突破时累计成交额(亿元)"]>=t, "突破时累计成交额(亿元)", ">=", t, "T")
# 前20日涨幅（T-1）
for t in (50,40,30,20,10,0):
    add(f"前20>{t}", d["前20日涨幅"]>t, "前20日涨幅", ">", t, "T-1")
for t in (-10,-20):
    add(f"前20<{t}", d["前20日涨幅"]<t, "前20日涨幅", "<", t, "T-1")
# 沉寂天数（T-1）
for t in (10,20,30,50):
    add(f"沉寂>={t}", d["沉寂天数"]>=t, "沉寂天数", ">=", t, "T-1")
# 突破时刻（T）
for t,nm in [(570,"开盘首分"),(600,"10点前"),(630,"10点半前"),(690,"上午")]:
    add(nm, d["突破分钟"]<=t, "突破分钟", "<=", t, "T")
# 开盘跳空（T）
add("高开", d["开盘跳空"]>0, "开盘跳空", ">", 0, "T")
add("高开>1", d["开盘跳空"]>1, "开盘跳空", ">", 1, "T")
add("高开>2", d["开盘跳空"]>2, "开盘跳空", ">", 2, "T")
add("低开", d["开盘跳空"]<0, "开盘跳空", "<", 0, "T")
add("低开<-1", d["开盘跳空"]<-1, "开盘跳空", "<", -1, "T")
# 日线派生（T-1）
add("昨涨>0", d["prev_chg"]>0, "prev_chg", ">", 0, "T-1")
add("昨涨>3", d["prev_chg"]>0.03, "prev_chg", ">", 3, "T-1")
add("昨跌<-2", d["prev_chg"]<-0.02, "prev_chg", "<", -2, "T-1")
add("ret5>0", d["ret5"]>0, "ret5", ">", 0, "T-1")
add("ret5>5", d["ret5"]>0.05, "ret5", ">", 5, "T-1")
add("ret5>10", d["ret5"]>0.10, "ret5", ">", 10, "T-1")
add("ret5<-5", d["ret5"]<-0.05, "ret5", "<", -5, "T-1")
add("ret10>10", d["ret10"]>0.10, "ret10", ">", 10, "T-1")
add("ret10>20", d["ret10"]>0.20, "ret10", ">", 20, "T-1")
add("ret20>20", d["ret20"]>0.20, "ret20", ">", 20, "T-1")
add("ret60>30", d["ret60"]>0.30, "ret60", ">", 30, "T-1")
add("多头排列", d["ma_bull"]>0.5, "ma_bull", "==", 1, "T-1")
add("MA20上翘", d["ma20_slope"]>0, "ma20_slope", ">", 0, "T-1")
add("MA60上翘", d["ma60_slope"]>0, "ma60_slope", ">", 0, "T-1")
add("近60日高", d["dist60hi"]>-0.03, "dist60hi", ">", -3, "T-1")
add("远60日高", d["dist60hi"]<-0.10, "dist60hi", "<", -10, "T-1")
add("远250日高", d["dist250hi"]<-0.30, "dist250hi", "<", -30, "T-1")
add("量比>1.2", d["volratio"]>1.2, "volratio", ">", 1.2, "T-1")
add("量比>1.5", d["volratio"]>1.5, "volratio", ">", 1.5, "T-1")
add("振幅20>4", d["amp20"]>0.04, "amp20", ">", 4, "T-1")
add("振幅20>6", d["amp20"]>0.06, "amp20", ">", 6, "T-1")
add("昨日涨停", d["limit_streak"]>=1, "limit_streak", ">=", 1, "T-1")
add("昨日连板", d["limit_streak"]>=2, "limit_streak", ">=", 2, "T-1")
add("低价<20", d["买入价"]<20, "买入价", "<", 20, "T")
add("高价>50", d["买入价"]>50, "买入价", ">", 50, "T")
# 市场（T-1）
add("昨涨停>=50", d["limit_up"]>=50, "limit_up", ">=", 50, "T-1")
add("昨涨停>=80", d["limit_up"]>=80, "limit_up", ">=", 80, "T-1")
add("昨上涨占比>50", d["up_ratio"]>50, "up_ratio", ">", 50, "T-1")
add("昨上涨占比>60", d["up_ratio"]>60, "up_ratio", ">", 60, "T-1")
add("昨创业均涨>0", d["avg_ret"]>0, "avg_ret", ">", 0, "T-1")
add("昨创业均涨>1", d["avg_ret"]>1, "avg_ret", ">", 1, "T-1")
add("昨弱势(占比<30)", d["up_ratio"]<30, "up_ratio", "<", 30, "T-1")

# 矩阵
names = list(conds.keys())
M = np.column_stack([conds[n]["mask"].astype(bool) for n in names]) if names else np.zeros((len(d),0),bool)
log(f"原子条件: {len(names)} 个, 矩阵 {M.shape}")
for n in names:
    m = conds[n]["mask"].astype(bool)
    v = d["T1"].values[m[:len(d)]]
    if len(v) < 5: continue
    conds[n]["n"] = int(m.sum()); conds[n]["mean"] = round(float(v.mean()),2)
log("条件样例（n>=20，按均值前 15）:")
rows = [(n, conds[n]["n"], conds[n]["mean"]) for n in names if conds[n].get("n",0)>=20]
for n,cn,cm in sorted(rows, key=lambda x:-x[2])[:15]:
    log(f"  {n}: {cn} 条, T1 {cm}%")

# 输出
out_cols = ["日期","星期","股票代码","股票名称","突破时刻","突破分钟","突破时涨幅","买入价","量能倍数",
            "突破时累计成交额(亿元)","开盘跳空","沉寂天数","前20日涨幅","当日收盘涨幅","T1","月","半年",
            "ret1","ret3","ret5","ret10","ret20","ret60","amp5","amp20","amp60",
            "ma5","ma10","ma20","ma60","ma120","ma250","ma_bull","ma20_slope","ma60_slope",
            "dist60hi","dist250hi","volratio","prev_chg","prev_amp","limit_streak",
            "limit_up","limit_down","up_ratio","avg_ret","strong5","big_move","lianban"]
keep = [c for c in out_cols if c in d.columns]
d[keep].to_parquet(os.path.join(OUT,"features.parquet"), index=False)
np.savez_compressed(os.path.join(OUT,"cond_matrix.npz"), M=M, names=np.array(names))
with open(os.path.join(OUT,"conditions.json"),"w",encoding="utf-8") as f:
    json.dump({n:{k:v for k,v in c.items() if k!="mask"} for n,c in conds.items()}, f, ensure_ascii=False, indent=1)
log("已保存 features.parquet / cond_matrix.npz / conditions.json")