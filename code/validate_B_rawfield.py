# -*- coding: utf-8 -*-
"""
验证 B：CMORPH 原始场断点交叉检验
===================================================================
目的：判别事件目录中的 3 处阶跃断点（2005/06、2014/15、2020/21）
      来源于 (i) CMORPH 输入场本身，还是 (ii) 本项目的事件检测管线。

判别逻辑（decisive test）：
  原始场跳变  + 目录跳变  -> 断点源自 CMORPH 输入（数据层，非管线缺陷）
  原始场平稳  + 目录跳变  -> 断点源自检测算法（管线缺陷，必须修）

核心创新诊断：湿区空间连通性（mean neighbour count）
  若"碎片化"假说成立（同一场雨被切成更多更小事件），
  则原始场湿区的空间连通性应在断点后下降。
  这是把"事件层碎片化"直接归因到"输入场纹理变化"的关键证据。

数据：E:/date download/CMORPH/{year}/{mm}/{dd}/CMORPH_V1.0_ADJ_8km-30min_YYYYMMDDHH.nc
      变量 cmorph, 单位 mm/hr(rain rate), 每文件 2 个 30-min 时步
      网格 1649(lat, -60~60) x 4948(lon, 0~360)

抽样设计：26 年 x {1月, 7月} x {1, 11, 21 日} x 24 文件/日 = 每年 144 文件
      同一日历月跨年比较 -> 消除季节效应，仅留年际/时段差异
输出：validation_B_rawfield.json
"""
import os, sys, json, time, glob
import numpy as np
from netCDF4 import Dataset

RAW_ROOT = r"E:\date download\CMORPH"
OUT = r"F:\output_CMORPH_2000_2025\validation_B_rawfield.json"

YEARS = list(range(2000, 2026))          # 26 年，与事件目录一致
MONTHS = [1, 7]                          # 冬/夏各一，消除季节混淆
DAYS = [1, 11, 21]                       # 每月 3 天
WET_THR = 0.1                            # mm/hr，检测脚本 global_rain_threshold
THRS = [0.05, 0.1, 0.2]                  # 多阈值敏感性（对应气候自适应范围）

# 纬度带（CMORPH 仅覆盖 ±60）
BANDS = [(-60, -40, "60-40S"), (-40, -20, "40-20S"), (-20, 0, "20-0S"),
         (0, 20, "0-20N"), (20, 40, "20-40N"), (40, 60, "40-60N")]


def neighbour_stats(wet):
    """湿区空间连通性：平均邻居数 = 2*(邻接对数)/湿格点数
    wet: 2D bool 数组
    返回 (mean_neighbours, n_wet)
    值域 [0,4]；越小越碎片化（孤立像元多），越大越成片。
    """
    n_wet = int(wet.sum())
    if n_wet == 0:
        return np.nan, 0
    h = int(np.count_nonzero(wet[:, :-1] & wet[:, 1:]))
    v = int(np.count_nonzero(wet[:-1, :] & wet[1:, :]))
    return 2.0 * (h + v) / n_wet, n_wet


def process_field(d, lat_idx_bands):
    """对单个 2D 场(mm/hr)计算全部诊断量"""
    fin = np.isfinite(d)
    n_valid = int(fin.sum())
    if n_valid == 0:
        return None
    out = {"n_valid": n_valid}

    # 全球：多阈值湿格点率 + 条件平均雨强
    for t in THRS:
        wet = fin & (d > t)
        nw = int(wet.sum())
        out[f"wet_frac_{t}"] = nw / n_valid
        out[f"cond_mean_{t}"] = float(d[wet].mean()) if nw else 0.0

    # 无条件平均降水率（水量守恒代理）
    out["mean_rate"] = float(np.nanmean(np.where(fin, d, np.nan)))

    # 湿格点雨强分位数（主阈值）
    wet_main = fin & (d > WET_THR)
    vals = d[wet_main]
    if vals.size:
        qs = np.percentile(vals, [50, 90, 99, 99.9])
        out["q50"], out["q90"], out["q99"], out["q999"] = [float(x) for x in qs]
    else:
        out["q50"] = out["q90"] = out["q99"] = out["q999"] = 0.0

    # ★ 核心：湿区空间连通性（全球 + 分带）
    mn, nw = neighbour_stats(wet_main)
    out["mean_neigh"] = float(mn) if np.isfinite(mn) else None
    out["n_wet_main"] = nw

    # 孤立像元比例（连通性的另一面：邻居数=0 的湿格点）
    W = wet_main
    nb = np.zeros(W.shape, dtype=np.uint8)
    nb[:, :-1] += W[:, 1:]
    nb[:, 1:] += W[:, :-1]
    nb[:-1, :] += W[1:, :]
    nb[1:, :] += W[:-1, :]
    out["isolated_frac"] = float((W & (nb == 0)).sum() / nw) if nw else None

    # 分纬度带
    for lo, hi, name in BANDS:
        sl = lat_idx_bands[name]
        db = d[sl, :]
        fb = np.isfinite(db)
        nvb = int(fb.sum())
        if nvb == 0:
            continue
        wb = fb & (db > WET_THR)
        nwb = int(wb.sum())
        out[f"wet_frac__{name}"] = nwb / nvb
        out[f"mean_rate__{name}"] = float(np.nanmean(np.where(fb, db, np.nan)))
        out[f"cond_mean__{name}"] = float(db[wb].mean()) if nwb else 0.0
        mnb, _ = neighbour_stats(wb)
        out[f"mean_neigh__{name}"] = float(mnb) if np.isfinite(mnb) else None
    return out


def main():
    t_start = time.time()
    # 用首个可用文件建立纬度带索引
    lat = None
    for y in YEARS:
        fs = glob.glob(os.path.join(RAW_ROOT, str(y), "01", "01", "*.nc"))
        if fs:
            with Dataset(fs[0]) as ds:
                lat = ds["lat"][:].data if hasattr(ds["lat"][:], "data") else np.asarray(ds["lat"][:])
            break
    if lat is None:
        print("FATAL: 找不到任何原始文件"); sys.exit(1)
    lat_idx_bands = {name: np.where((lat >= lo) & (lat < hi))[0] for lo, hi, name in BANDS}
    print(f"纬度网格: {lat.size} 点, {lat.min():.2f}~{lat.max():.2f}")
    for lo, hi, n in BANDS:
        print(f"  {n}: {lat_idx_bands[n].size} 行")
    # 转成切片以加速
    lat_idx_bands = {n: slice(int(v.min()), int(v.max()) + 1) for n, v in lat_idx_bands.items()}

    results = {}   # results[(year, month)] = list of per-timestep dicts
    n_files = 0
    for y in YEARS:
        for m in MONTHS:
            key = f"{y}-{m:02d}"
            recs = []
            for dd in DAYS:
                ddir = os.path.join(RAW_ROOT, str(y), f"{m:02d}", f"{dd:02d}")
                if not os.path.isdir(ddir):
                    continue
                for hh in range(24):
                    fp = os.path.join(ddir, f"CMORPH_V1.0_ADJ_8km-30min_{y}{m:02d}{dd:02d}{hh:02d}.nc")
                    if not os.path.exists(fp):
                        continue
                    try:
                        with Dataset(fp) as ds:
                            arr = ds["cmorph"][:]
                    except Exception as e:
                        print(f"  !! 读取失败 {os.path.basename(fp)}: {e}")
                        continue
                    a = arr.filled(np.nan) if np.ma.isMaskedArray(arr) else np.asarray(arr)
                    n_files += 1
                    for ti in range(a.shape[0]):
                        r = process_field(a[ti], lat_idx_bands)
                        if r:
                            recs.append(r)
            if recs:
                results[key] = recs
                print(f"{key}: {len(recs)} 时步  wet_frac={np.mean([r['wet_frac_0.1'] for r in recs]):.4f} "
                      f"mean_rate={np.mean([r['mean_rate'] for r in recs]):.4f} "
                      f"mean_neigh={np.nanmean([r['mean_neigh'] for r in recs if r['mean_neigh']]):.3f}",
                      flush=True)
            else:
                print(f"{key}: 无数据", flush=True)

    # ---- 聚合到 (year, month) 均值 ----
    agg = {}
    for key, recs in results.items():
        a = {}
        keys_all = set()
        for r in recs:
            keys_all |= set(r.keys())
        for k in keys_all:
            vals = [r[k] for r in recs if r.get(k) is not None and np.isfinite(r[k])]
            a[k] = float(np.mean(vals)) if vals else None
        a["n_timesteps"] = len(recs)
        agg[key] = a

    payload = {
        "meta": {
            "raw_root": RAW_ROOT, "years": YEARS, "months": MONTHS, "days": DAYS,
            "wet_threshold_mm_per_hr": WET_THR, "thresholds": THRS,
            "n_files_read": n_files, "grid": [int(lat.size), 4948],
            "lat_range": [float(lat.min()), float(lat.max())],
            "elapsed_sec": round(time.time() - t_start, 1),
            "note": "cmorph 单位 mm/hr (rain rate); 每文件 2 个 30-min 时步",
        },
        "per_year_month": agg,
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
    print(f"\n完成: {n_files} 文件, {sum(len(v) for v in results.values())} 时步, "
          f"{time.time()-t_start:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
