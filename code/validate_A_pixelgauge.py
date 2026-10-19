# -*- coding: utf-8 -*-
"""
验证 A：事件检测的命中率 / 虚警率（pixel-as-gauge 自洽验证）
===================================================================
思路
----
把 CMORPH 原始格点的 30-min 时间序列当作"伪雨量计"，用**与 DBSCAN 完全独立**
的经典单点 IETD 方法重建该格点的降雨事件序列作为参照，再与事件目录做时间匹配，
输出 POD / FAR / CSI 等技巧评分。

这验证的是"事件检测算法是否忠实还原了输入场中的降雨过程"，
不是"CMORPH 是否等于真实降水"（后者需要真实雨量计，见 §接口说明）。

设计要点
--------
1. 参照方法（真值侧）：经典 IETD 法
   - 湿判定 rate > threshold
   - 相邻湿段干期 < IETD 则合并为同一事件
   - 过滤 min_rain_per_event / min_duration_hours
   参数与 rainfall_event_detection.py CONFIG 对齐（含气候自适应的纬度带版本）
2. 目录侧：事件仅提供 center_lon/lat + n_grids，无多边形
   -> 用等效圆半径近似覆盖范围 r = sqrt(n_grids/pi) * 0.072771 deg
   （此近似是本验证的主要局限，已在报告中声明）
3. 抽样月份特意选在 3 个断点前后，可同时回答：
   "检测算法的技巧评分是否随时段系统性劣化？"

输出：validation_A_pixelgauge.json
"""
import os, sys, json, time, calendar
import numpy as np
import pandas as pd
from netCDF4 import Dataset

RAW_ROOT = r"E:\date download\CMORPH"
CAT_DIR = r"F:\output_CMORPH_2000_2025"
OUT = os.path.join(CAT_DIR, "validation_A_pixelgauge.json")

# 断点前后成对月份（同为 1 月，消除季节效应）
TARGET_MONTHS = [(2005, 1), (2006, 1),      # 断点 1: 2005/06
                 (2014, 1), (2015, 1),      # 断点 2: 2014/15
                 (2020, 1), (2021, 1)]      # 断点 3: 2020/21

N_PIX_PER_BAND = 40          # 每纬度带抽样格点数
BANDS = [(-60, -40, "60-40S"), (-40, -20, "40-20S"), (-20, 0, "20-0S"),
         (0, 20, "0-20N"), (20, 40, "20-40N"), (40, 60, "40-60N")]
GRID_DEG = 0.072771376
DT_H = 0.5                   # 时间步长(h)

# 与 rainfall_event_detection.py CONFIG 对齐
LAT_BAND_PARAMS = [(-23.5, 23.5, 0.05, 4), (23.5, 35, 0.10, 6),
                   (35, 55, 0.15, 8), (55, 90, 0.20, 12)]
GLOBAL_THR, GLOBAL_IETD = 0.10, 6
MIN_RAIN_EVENT = 0.5         # mm
MIN_DUR_H = 0.5              # h


def band_params(lat):
    """按纬度返回 (threshold mm/hr, IETD hours)，复刻检测脚本的自适应逻辑"""
    a = abs(lat)
    for lo, hi, thr, ietd in LAT_BAND_PARAMS:
        if lo <= a < hi or (lo < 0 and a < hi):
            return thr, ietd
    return GLOBAL_THR, GLOBAL_IETD


def reconstruct_events(rate, thr, ietd_h):
    """经典 IETD 单点事件重建
    rate: 1D array, mm/hr, 步长 DT_H
    返回 [(start_idx, end_idx, total_mm, dur_h), ...]
    """
    wet = np.isfinite(rate) & (rate > thr)
    if not wet.any():
        return []
    # 连续湿段
    d = np.diff(wet.astype(np.int8))
    starts = list(np.where(d == 1)[0] + 1)
    ends = list(np.where(d == -1)[0] + 1)      # exclusive
    if wet[0]:
        starts.insert(0, 0)
    if wet[-1]:
        ends.append(len(wet))
    segs = list(zip(starts, ends))
    # 按 IETD 合并
    gap_steps = int(round(ietd_h / DT_H))
    merged = [list(segs[0])]
    for s, e in segs[1:]:
        if s - merged[-1][1] < gap_steps:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    # 过滤
    out = []
    for s, e in merged:
        seg = rate[s:e]
        tot = float(np.nansum(seg) * DT_H)      # mm/hr * h = mm
        dur = (e - s) * DT_H
        if tot >= MIN_RAIN_EVENT and dur >= MIN_DUR_H:
            out.append((s, e, tot, dur))
    return out


def load_catalog_month(year, month):
    """读事件目录中指定年月的事件；返回 DataFrame(center_lon[-180,180], center_lat, n_grids, t0, t1)"""
    fp = os.path.join(CAT_DIR, f"rainfall_events_{year}_fullyear.csv")
    if not os.path.exists(fp):
        fp = os.path.join(CAT_DIR, f"rainfall_events_{year}{month:02d}_raw.csv")
        if not os.path.exists(fp):
            return None
    usecols = ["center_lon", "center_lat", "n_grids", "duration_hours",
               "start_time", "avg_precip_depth"]
    df = pd.read_csv(fp, usecols=usecols)
    df["start_time"] = pd.to_datetime(df["start_time"], errors="coerce")
    df = df[df["start_time"].dt.month == month]
    df = df.dropna(subset=["start_time"])
    df["t1"] = df["start_time"] + pd.to_timedelta(df["duration_hours"], unit="h")
    # 经度归一到 [-180,180]（原始 [0,360]）
    lon = df["center_lon"].to_numpy()
    df["lon180"] = np.where(lon > 180, lon - 360, lon)
    # 等效圆半径(度)
    df["r_deg"] = np.sqrt(np.maximum(df["n_grids"].to_numpy(), 1) / np.pi) * GRID_DEG
    return df.reset_index(drop=True)


def main():
    t_start = time.time()
    # ---- 网格与抽样 ----
    f0 = os.path.join(RAW_ROOT, "2010", "01", "01", "CMORPH_V1.0_ADJ_8km-30min_2010010100.nc")
    if not os.path.exists(f0):
        print("FATAL: 参考文件不存在", f0); sys.exit(1)
    with Dataset(f0) as ds:
        lat = np.asarray(ds["lat"][:]); lon = np.asarray(ds["lon"][:])
        ref = ds["cmorph"][:]
    ref = ref.filled(np.nan) if np.ma.isMaskedArray(ref) else np.asarray(ref)
    wet_ref = np.isfinite(ref[0]) & (ref[0] > 0.1)

    rng = np.random.default_rng(42)
    pix = []
    for lo, hi, name in BANDS:
        rows = np.where((lat >= lo) & (lat < hi))[0]
        # 只在参考场中"有雨潜力"的行里抽，避免全干区
        cand_i, cand_j = [], []
        for _ in range(N_PIX_PER_BAND * 60):
            i = int(rng.choice(rows)); j = int(rng.integers(0, lon.size))
            if np.isfinite(ref[0, i, j]):
                cand_i.append(i); cand_j.append(j)
            if len(cand_i) >= N_PIX_PER_BAND * 6:
                break
        # 优先选参考时刻附近有降水活动的格点（3x3 邻域有湿）
        scored = []
        for i, j in zip(cand_i, cand_j):
            i0, i1 = max(0, i - 1), min(lat.size, i + 2)
            j0, j1 = max(0, j - 1), min(lon.size, j + 2)
            scored.append(((wet_ref[i0:i1, j0:j1]).sum(), i, j))
        scored.sort(reverse=True)
        chosen = scored[:N_PIX_PER_BAND]
        if len(chosen) < N_PIX_PER_BAND:      # 兜底
            chosen = scored
        for _, i, j in chosen:
            lo180 = lon[j] - 360 if lon[j] > 180 else lon[j]
            pix.append({"band": name, "i": i, "j": j,
                        "lat": float(lat[i]), "lon": float(lo180)})
    print(f"抽样格点: {len(pix)} 个 (每带 {N_PIX_PER_BAND})")
    for lo, hi, n in BANDS:
        print(f"  {n}: {sum(1 for p in pix if p['band']==n)}")

    results = {}
    for (year, month) in TARGET_MONTHS:
        key = f"{year}-{month:02d}"
        ndays = calendar.monthrange(year, month)[1]
        nstep = ndays * 48
        II = np.array([p["i"] for p in pix]); JJ = np.array([p["j"] for p in pix])
        series = np.full((len(pix), nstep), np.nan, dtype=np.float32)
        t_read = time.time(); nf = 0
        for dd in range(1, ndays + 1):
            ddir = os.path.join(RAW_ROOT, str(year), f"{month:02d}", f"{dd:02d}")
            if not os.path.isdir(ddir):
                continue
            for hh in range(24):
                fp = os.path.join(ddir, f"CMORPH_V1.0_ADJ_8km-30min_{year}{month:02d}{dd:02d}{hh:02d}.nc")
                if not os.path.exists(fp):
                    continue
                try:
                    with Dataset(fp) as ds:
                        a = ds["cmorph"][:]
                except Exception:
                    continue
                a = a.filled(np.nan) if np.ma.isMaskedArray(a) else np.asarray(a)
                nf += 1
                base = (dd - 1) * 48 + hh * 2
                for ti in range(min(a.shape[0], 2)):
                    if base + ti < nstep:
                        series[:, base + ti] = a[ti][II, JJ]
        print(f"{key}: 读取 {nf} 文件 {time.time()-t_read:.0f}s, 有效率={np.isfinite(series).mean():.3f}", flush=True)

        cat = load_catalog_month(year, month)
        if cat is None:
            print(f"  !! 目录缺失 {key}"); continue
        t0_month = pd.Timestamp(year=year, month=month, day=1)
        cat_s = (cat["start_time"] - t0_month).dt.total_seconds().to_numpy() / 3600.0 / DT_H
        cat_e = (cat["t1"] - t0_month).dt.total_seconds().to_numpy() / 3600.0 / DT_H
        clon = cat["lon180"].to_numpy(); clat = cat["center_lat"].to_numpy()
        crad = cat["r_deg"].to_numpy(); cdep = cat["avg_precip_depth"].to_numpy()
        print(f"  目录事件 {len(cat)} 个", flush=True)

        per_pix = []
        for pi, p in enumerate(pix):
            thr, ietd = band_params(p["lat"])
            truth = reconstruct_events(series[pi], thr, ietd)
            # 覆盖该格点的目录事件（等效圆 + 经度环绕）
            dlon = np.abs(clon - p["lon"]); dlon = np.minimum(dlon, 360 - dlon)
            dlon = dlon * np.cos(np.deg2rad(p["lat"]))        # 经度按纬度收缩
            dlat = np.abs(clat - p["lat"])
            cover = np.sqrt(dlon ** 2 + dlat ** 2) <= crad
            ci = np.where(cover)[0]
            hits = 0
            dep_pairs = []
            for (s, e, tot, dur) in truth:
                ov = ci[(cat_s[ci] < e) & (cat_e[ci] > s)]
                if ov.size:
                    hits += 1
                    dep_pairs.append((tot, float(np.mean(cdep[ov]))))
            n_truth, n_cat = len(truth), int(ci.size)
            # 虚警：目录事件覆盖该格点但无任何真值事件时间重叠
            fa = 0
            for k in ci:
                if not any((cat_s[k] < e) and (cat_e[k] > s) for (s, e, _, _) in truth):
                    fa += 1
            per_pix.append({"band": p["band"], "lat": p["lat"], "lon": p["lon"],
                            "thr": thr, "ietd": ietd,
                            "n_truth": n_truth, "n_cat": n_cat,
                            "hits": hits, "misses": n_truth - hits, "false_alarms": fa,
                            "dep_pairs": dep_pairs})

        # ---- 汇总 ----
        def skill(recs):
            H = sum(r["hits"] for r in recs); M = sum(r["misses"] for r in recs)
            F = sum(r["false_alarms"] for r in recs)
            NT = sum(r["n_truth"] for r in recs); NC = sum(r["n_cat"] for r in recs)
            pod = H / NT if NT else None
            far = F / NC if NC else None
            csi = H / (H + M + F) if (H + M + F) else None
            bias = NC / NT if NT else None
            return dict(POD=pod, FAR=far, CSI=csi, bias_ratio=bias,
                        hits=H, misses=M, false_alarms=F, n_truth=NT, n_cat=NC,
                        n_pix=len(recs))

        allp = [r for r in per_pix if r["n_truth"] > 0]
        summary = {"overall": skill(allp),
                   "by_band": {n: skill([r for r in allp if r["band"] == n])
                               for _, _, n in BANDS}}
        # 雨深关系（格点累积 vs 目录簇平均雨深）
        xs = [a for r in per_pix for a, b in r["dep_pairs"]]
        ys = [b for r in per_pix for a, b in r["dep_pairs"]]
        if len(xs) > 10:
            x, y = np.array(xs), np.array(ys)
            summary["depth"] = {"n": len(x),
                                "pixel_mean_mm": float(x.mean()),
                                "catalog_mean_mm": float(y.mean()),
                                "ratio_cat_over_pix": float(y.mean() / x.mean()),
                                "pearson_r": float(np.corrcoef(x, y)[0, 1]),
                                "median_pixel": float(np.median(x)),
                                "median_catalog": float(np.median(y))}
        results[key] = summary
        o = summary["overall"]
        print(f"  => POD={o['POD']:.3f} FAR={o['FAR']:.3f} CSI={o['CSI']:.3f} "
              f"bias={o['bias_ratio']:.2f} (truth={o['n_truth']}, cat={o['n_cat']})", flush=True)

    payload = {"meta": {"method": "pixel-as-gauge, independent IETD reconstruction vs 3D-DBSCAN catalog",
                        "target_months": [f"{y}-{m:02d}" for y, m in TARGET_MONTHS],
                        "n_pixels": len(pix), "n_pix_per_band": N_PIX_PER_BAND,
                        "grid_deg": GRID_DEG, "dt_hours": DT_H,
                        "params": {"lat_band_params": LAT_BAND_PARAMS,
                                   "min_rain_per_event_mm": MIN_RAIN_EVENT,
                                   "min_duration_h": MIN_DUR_H},
                        "coverage_approx": "equivalent-circle radius from n_grids",
                        "elapsed_sec": round(time.time() - t_start, 1)},
               "pixels": pix, "results": results}
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
    print(f"\n完成 -> {OUT}  ({time.time()-t_start:.0f}s)")


if __name__ == "__main__":
    main()
