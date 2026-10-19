# -*- coding: utf-8 -*-
"""字段间公式自洽 + 极值/截断校验 (抽样 4 个年份覆盖 4 个时段)"""
import os, json
import numpy as np
import pandas as pd

BASE = r"F:\output_CMORPH_2000_2025"
SAMP = [2003, 2010, 2018, 2024]   # A / B / C / D
ALL = ["event_id","start_time","end_time","duration_hours","total_rain_depth",
       "max_30min_rain","max_30min_time","mean_intensity","mean_intensity_all",
       "lon_min","lon_max","lat_min","lat_max","center_lon","center_lat",
       "n_grids","avg_precip_depth","max_30min_intensity","total_water_volume",
       "is_extreme_precip","is_extreme_max30","confidence","lon_range_circular",
       "crosses_dateline","location_type"]

out = {}
for y in SAMP:
    df = pd.read_csv(os.path.join(BASE, f"rainfall_events_{y}_fullyear.csv"))
    r = {}
    d = df.avg_precip_depth.to_numpy("float64")
    ng = df.n_grids.to_numpy("float64")
    dur = df.duration_hours.to_numpy("float64")
    st = pd.to_datetime(df.start_time, utc=True)
    en = pd.to_datetime(df.end_time, utc=True)
    dur_t = (en - st).dt.total_seconds().to_numpy() / 3600.0

    # 1) max_30min_intensity = 2 * max_30min_rain ?
    a = df.max_30min_intensity.to_numpy("float64"); b = 2 * df.max_30min_rain.to_numpy("float64")
    r["max30int_eq_2x"] = dict(max_abs_err=float(np.nanmax(np.abs(a - b))),
                               frac_ok=float(np.mean(np.abs(a - b) < 1e-3)))
    # 2) mean_intensity_all = avg_precip_depth / duration_hours ?
    c = df.mean_intensity_all.to_numpy("float64"); e = d / np.where(dur > 0, dur, np.nan)
    r["meanint_all_eq_depth_over_dur"] = dict(median_rel_err=float(np.nanmedian(np.abs(c - e) / np.abs(e))),
                                              frac_ok=float(np.mean(np.abs(c - e) / np.abs(e) < 0.02)))
    # 3) total_rain_depth = avg_precip_depth * n_grids ?
    f = df.total_rain_depth.to_numpy("float64"); g = d * ng
    r["totaldepth_eq_avg_x_n"] = dict(median_rel_err=float(np.nanmedian(np.abs(f - g) / np.abs(g))),
                                      frac_ok=float(np.mean(np.abs(f - g) / np.abs(g) < 0.02)))
    # 4) duration_hours == end - start ?
    r["duration_eq_end_minus_start"] = dict(median_abs_err_h=float(np.nanmedian(np.abs(dur - dur_t))),
                                            frac_ok=float(np.mean(np.abs(dur - dur_t) < 1e-6)))
    # 5) total_water_volume 系数
    cosl = np.cos(np.deg2rad(np.clip(df.center_lat.to_numpy("float64"), -89.9, 89.9)))
    den = d * ng * cosl
    ok = den > 1e-9
    ratio = df.total_water_volume.to_numpy("float64")[ok] / den[ok]
    r["twv_coefficient"] = dict(median=float(np.median(ratio)),
                                p05=float(np.percentile(ratio, 5)),
                                p95=float(np.percentile(ratio, 95)),
                                implied_grid_area_km2=float(np.median(ratio) / 1000.0))
    # 6) 极值/截断
    r["extreme"] = dict(
        max_depth=float(np.max(d)),
        n_gt_200=int((d > 200).sum()), n_gt_400=int((d > 400).sum()),
        n_ge_199=int(((d >= 199) & (d <= 200.0001)).sum()),   # 硬截断堆积
        n_gt_1000=int((d > 1000).sum()),
        max_max30=float(df.max_30min_rain.max()),
        n_m30_gt_100=int((df.max_30min_rain > 100).sum()),
        flag_precip_mismatch=int((df.is_extreme_precip != (d > 200).astype(int)).sum()),
        flag_max30_mismatch=int((df.is_extreme_max30 != (df.max_30min_rain > 100).astype(int)).sum()),
    )
    # 7) 置信度 / 边界一致性
    r["ranges"] = dict(
        conf_min=float(df.confidence.min()), conf_max=float(df.confidence.max()),
        n_conf_lt_03=int((df.confidence < 0.3).sum()),
        n_grids_min=int(df.n_grids.min()), dur_min=float(dur.min()), dur_max=float(dur.max()),
        lat_in_range=bool((df.center_lat.between(-60, 60)).all()),
        lon_in_0_360=bool((df.center_lon.between(0, 360)).all()),
        n_cross_dateline=int(df.crosses_dateline.sum()),
        center_inside_bbox=float(np.mean((df.center_lat >= df.lat_min - 1e-6) &
                                         (df.center_lat <= df.lat_max + 1e-6))),
    )
    # 8) 最小网格阈值: 海洋25/陆地5
    isl = (df.location_type == "内陆")
    r["min_grid_rule"] = dict(land_min=int(df.n_grids[isl].min()),
                              ocean_min=int(df.n_grids[~isl].min()),
                              land_violations=int((df.n_grids[isl] < 5).sum()),
                              ocean_violations=int((df.n_grids[~isl] < 25).sum()))
    out[y] = r
    print(f"--- {y} ---")
    for k, v in r.items():
        print(f"   {k}: {v}", flush=True)

with open(os.path.join(BASE, "formula_validation_v5.json"), "w", encoding="utf-8") as fp:
    json.dump(out, fp, ensure_ascii=False, indent=1)
print("\nDONE")
