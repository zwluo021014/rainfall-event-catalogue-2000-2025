# -*- coding: utf-8 -*-
"""
验证 C：CMORPH 断点的 ERA5 独立交叉检验
===================================================================
目的：判定 CMORPH 事件目录/原始场在 2005/06、2014/15、2020/21 的阶跃
      是否为**真实的地球物理信号**，还是卫星反演产物的人工不连续。

判别逻辑
  CMORPH 跳变 + ERA5 同步跳变  -> 可能是真实气候/天气信号
  CMORPH 跳变 + ERA5 不跳变    -> 坐实为 CMORPH 反演的人工不连续（本文预期）
  并以"无已知断点的相邻年对"作为正常年际变率基线（CTRL）。

数据源（免 CDS 凭证）：ARCO-ERA5, Google Cloud 公开数据集，匿名读取
  gs://gcp-public-data-arco-era5/ar/1959-2022-1h-240x121_equiangular_with_poles_conservative.zarr
  变量 total_precipitation, 单位 m (1-h 累积), 维度顺序 (time, longitude, latitude)
  空间 1.5deg (240x121) —— 对"20 度纬度带平均降水的相对跳变"足够，且流量仅 ~11 MB/月

★ 方法要点：CMORPH 0.0728deg 与 ERA5 1.5deg 分辨率差异巨大，
  湿格点率/事件数等尺度依赖量**绝不可直接比较绝对值**。
  本脚本只比较【跨断点的相对跳变幅度】(after/before - 1)，
  分辨率带来的系统性偏差在做比值时自动消去。

输出：validation_C_era5.json
"""
import os, json, time
import numpy as np

OUT = r"F:\output_CMORPH_2000_2025\validation_C_era5.json"
CACHE = r"F:\output_CMORPH_2000_2025\_era5_month_cache.json"
ARCO = ("gs://gcp-public-data-arco-era5/ar/"
        "1959-2022-1h-240x121_equiangular_with_poles_conservative.zarr")
VAR = "total_precipitation"

# 断点前后成对月份（同为 1 月，消除季节效应）
PAIRS = [((2005, 1), (2006, 1), "BP1_2005/06"),
         ((2014, 1), (2015, 1), "BP2_2014/15"),
         ((2020, 1), (2021, 1), "BP3_2020/21")]
# 对照对：段内相邻年（无已知断点），给出正常年际变率基线
CONTROL = [((2009, 1), (2010, 1), "CTRL_2009/10"),
           ((2017, 1), (2018, 1), "CTRL_2017/18"),
           ((2002, 1), (2003, 1), "CTRL_2002/03")]

BANDS = [(-60, -40, "60-40S"), (-40, -20, "40-20S"), (-20, 0, "20-0S"),
         (0, 20, "0-20N"), (20, 40, "20-40N"), (40, 60, "40-60N")]
WET_THR = 0.1     # mm/h


def month_stats(ds, year, month):
    """取指定年月 ERA5 1-h 降水，返回 60S-60N 全球与分带诊断（面积加权）"""
    t0 = f"{year}-{month:02d}-01T00"
    # 到当月最后一天 23 时
    import calendar
    nd = calendar.monthrange(year, month)[1]
    t1 = f"{year}-{month:02d}-{nd:02d}T23"
    da = ds[VAR].sel(time=slice(t0, t1))
    lats = np.asarray(da["latitude"].values, dtype=np.float64)
    keep = np.where((lats >= -60) & (lats <= 60))[0]
    da = da.isel(latitude=keep)
    lats = lats[keep]
    arr = np.asarray(da.values, dtype=np.float32) * 1000.0        # m -> mm (1h 累积 => mm/h)
    # 维度 (time, longitude, latitude)
    out = {"n_timesteps": int(arr.shape[0]),
           "grid": [int(arr.shape[1]), int(arr.shape[2])]}
    fin = np.isfinite(arr)
    w = np.cos(np.deg2rad(lats)).astype(np.float32)               # 纬度权重
    W = np.broadcast_to(w[None, None, :], arr.shape)

    def wmean(mask):
        num = float(np.sum(np.where(mask, W, 0.0)))
        return num

    tot_w = wmean(fin)
    out["mean_rate"] = float(np.sum(np.where(fin, arr * W, 0.0)) / tot_w)
    for thr in (0.1, 0.2):
        wet = fin & (arr > thr)
        out[f"wet_frac_{thr}"] = wmean(wet) / tot_w
        out[f"cond_mean_{thr}"] = float(arr[wet].mean()) if wet.any() else 0.0
    v = arr[fin & (arr > WET_THR)]
    if v.size:
        q = np.percentile(v, [50, 90, 99, 99.9])
        out["q50"], out["q90"], out["q99"], out["q999"] = [float(x) for x in q]

    for lo, hi, name in BANDS:
        m = (lats >= lo) & (lats < hi)
        if not m.any():
            continue
        ab = arr[:, :, m]
        wb = np.broadcast_to(np.cos(np.deg2rad(lats[m])).astype(np.float32)[None, None, :], ab.shape)
        fb = np.isfinite(ab)
        tb = float(np.sum(np.where(fb, wb, 0.0)))
        out[f"mean_rate__{name}"] = float(np.sum(np.where(fb, ab * wb, 0.0)) / tb)
        wetb = fb & (ab > WET_THR)
        out[f"wet_frac__{name}"] = float(np.sum(np.where(wetb, wb, 0.0)) / tb)
        out[f"cond_mean__{name}"] = float(ab[wetb].mean()) if wetb.any() else 0.0
    return out


def load_cmorph_ref():
    """载入验证 B（CMORPH 原始场）结果用于并排比较"""
    p = r"F:\output_CMORPH_2000_2025\validation_B_rawfield.json"
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f).get("per_year_month", {})


def rel_change(a, b):
    if a is None or b is None:
        return None
    try:
        a = float(a); b = float(b)
    except (TypeError, ValueError):
        return None
    if a == 0 or not np.isfinite(a) or not np.isfinite(b):
        return None
    return float(b / a - 1.0)


def main():
    t_start = time.time()
    import xarray as xr
    print("=== 打开 ARCO-ERA5（匿名，1.5deg 1-h）===", flush=True)
    ds = xr.open_zarr(ARCO, storage_options=dict(token="anon"), chunks=None)
    print(f"  OK  time={ds.sizes['time']}  lon={ds.sizes['longitude']}  lat={ds.sizes['latitude']}",
          flush=True)

    need = sorted({ym for pr in (PAIRS + CONTROL) for ym in pr[:2]})
    # ERA5 统计缓存：下载昂贵，比较部分可反复重算
    era = {}
    if os.path.exists(CACHE):
        try:
            with open(CACHE, encoding="utf-8") as f:
                era = json.load(f)
            print(f"  复用 ERA5 缓存 {len(era)} 个月: {sorted(era)}", flush=True)
        except Exception:
            era = {}
    for (y, m) in need:
        key = f"{y}-{m:02d}"
        if key in era:
            continue
        t0 = time.time()
        try:
            era[key] = month_stats(ds, y, m)
            with open(CACHE, "w", encoding="utf-8") as f:
                json.dump(era, f, ensure_ascii=False)
            e = era[key]
            print(f"  {key}: mean_rate={e['mean_rate']:.4f} mm/h  wet0.1={e['wet_frac_0.1']:.4f}  "
                  f"S.Ocean={e.get('mean_rate__60-40S', float('nan')):.4f}  "
                  f"({time.time()-t0:.0f}s, {e['n_timesteps']} 时步)", flush=True)
        except Exception as ex:
            print(f"  !! {key} 失败: {type(ex).__name__}: {str(ex)[:200]}", flush=True)

    cm = load_cmorph_ref()
    if cm is None:
        print("\n警告: 未找到 validation_B_rawfield.json，仅输出 ERA5 侧结果")

    METRICS = (["mean_rate", "wet_frac_0.1", "cond_mean_0.1", "q50", "q90", "q99"]
               + [f"mean_rate__{n}" for _, _, n in BANDS]
               + [f"wet_frac__{n}" for _, _, n in BANDS]
               + [f"cond_mean__{n}" for _, _, n in BANDS])
    comparison = {}
    from collections import Counter
    for (a, b, label) in PAIRS + CONTROL:
        ka, kb = f"{a[0]}-{a[1]:02d}", f"{b[0]}-{b[1]:02d}"
        rec = {"pair": [ka, kb], "era5": {}, "cmorph": {}, "verdict": {}}
        for k in METRICS:
            e = rel_change(era.get(ka, {}).get(k), era.get(kb, {}).get(k))
            rec["era5"][k] = e
            c = rel_change(cm.get(ka, {}).get(k), cm.get(kb, {}).get(k)) if cm else None
            rec["cmorph"][k] = c
            if e is not None and c is not None:
                flag = ("ARTEFACT" if abs(c) > 3 * max(abs(e), 0.01)
                        else "CONSISTENT" if abs(c - e) < 0.5 * max(abs(c), abs(e), 0.01)
                        else "PARTIAL")
                rec["verdict"][k] = {"cmorph_rel": round(c, 4), "era5_rel": round(e, 4),
                                     "amplification": (round(c / e, 2) if abs(e) > 1e-6 else None),
                                     "flag": flag}
        comparison[label] = rec
        fl = [v["flag"] for v in rec["verdict"].values()]
        if fl:
            print(f"{label}: {dict(Counter(fl))}  (n={len(fl)} 指标)", flush=True)

    payload = {"meta": {"era5_source": ARCO, "variable": f"{VAR} (m/1h -> mm/h)",
                        "era5_grid_deg": 1.5, "lat_range": [-60, 60],
                        "area_weighting": "cos(latitude)", "wet_threshold_mm_per_h": WET_THR,
                        "note": ("CMORPH 0.0728deg vs ERA5 1.5deg: 只比较跨断点相对跳变幅度，"
                                 "不比较绝对值"),
                        "flag_rule": {"ARTEFACT": "|cmorph_rel| > 3x max(|era5_rel|,0.01)",
                                      "CONSISTENT": "|cmorph_rel - era5_rel| < 0.5x max(...)",
                                      "PARTIAL": "其余"},
                        "pairs": [p[2] for p in PAIRS], "controls": [p[2] for p in CONTROL],
                        "elapsed_sec": round(time.time() - t_start, 1)},
               "era5_monthly": era, "comparison": comparison}
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
    print(f"\n完成 -> {OUT} ({time.time()-t_start:.0f}s)")


# =====================================================================
# FALLBACK（若 ARCO-ERA5 不可用）：CDS API
#   1) 注册 https://cds.climate.copernicus.eu/ 获取 UID:KEY
#   2) 写 ~/.cdsapirc:
#        url: https://cds.climate.copernicus.eu/api
#        key: <UID>:<APIKEY>
#   3) pip install cdsapi
#   4) 每个目标月单独请求：
#        import cdsapi; c = cdsapi.Client()
#        c.retrieve("reanalysis-era5-single-levels", {
#            "product_type": "reanalysis", "variable": "total_precipitation",
#            "year": "2021", "month": "01",
#            "day": [f"{d:02d}" for d in range(1, 32)],
#            "time": [f"{h:02d}:00" for h in range(24)],
#            "area": [60, -180, -60, 180], "format": "netcdf",
#        }, "era5_tp_202101.nc")
#   5) 用 xarray 打开本地 nc 替换 month_stats 的数据读取部分即可
# =====================================================================

if __name__ == "__main__":
    main()
