"""陆地子集独立统计 —— Paper5 数据集论文补充材料
口径：F:\\output_CMORPH_2000_2025\\rainfall_events_YYYY_fullyear.csv
26 年表头已核实完全一致（29 字段，MD5 指纹 e7271443），location_type 为第 25 列。
只用标准库，避免 7.8 GB 全量载入内存。
"""
import csv
import json
import math
import os
from collections import defaultdict

SRC = r"F:\output_CMORPH_2000_2025"
OUT_JSON = os.path.join(SRC, "land_subset_stats.json")
OUT_CSV = os.path.join(SRC, "land_subset_by_year.csv")

LAND_KEYS = ("内陆",)


def quantile(sorted_vals, q):
    if not sorted_vals:
        return None
    n = len(sorted_vals)
    idx = q * (n - 1)
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return sorted_vals[lo]
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (idx - lo)


# era 定义（与稿件一致）：A=2000–2005, B=2006–2014, C=2015–2020, D=2021–2025
def era_of(year):
    if year <= 2005:
        return "A"
    if year <= 2014:
        return "B"
    if year <= 2020:
        return "C"
    return "D"

csv.field_size_limit(10 * 1024 * 1024)

land_depth = []          # 陆地事件 total_rain_depth 抽样（用于分位数）
land_vol = []            # total_water_volume
land_dur = []            # duration_hours
land_grids = []          # n_grids

per_year = []
land_by_era = defaultdict(lambda: {"n": 0})
all_by_era = defaultdict(lambda: {"n": 0})
land_type_all = defaultdict(int)
land_continent = defaultdict(int)
land_climate = defaultdict(int)
land_country = defaultdict(int)
land_extreme_precip = 0
land_extreme_max30 = 0
land_dateline = 0
land_ocean_named = 0
ocean_named = 0

for year in range(2000, 2026):
    path = os.path.join(SRC, f"rainfall_events_{year}_fullyear.csv")
    n_all = 0
    n_land = 0
    n_ocean = 0
    era = era_of(year)
    with open(path, "r", encoding="utf-8", newline="") as f:
        rd = csv.DictReader(f)
        for row in rd:
            n_all += 1
            all_by_era[era]["n"] += 1
            lt = row.get("location_type", "")
            if lt in LAND_KEYS:
                n_land += 1
                land_by_era[era]["n"] += 1
                land_type_all[lt] += 1
                if row.get("continent"):
                    land_continent[row["continent"]] += 1
                if row.get("climate_zone"):
                    land_climate[row["climate_zone"]] += 1
                if row.get("country"):
                    land_country[row["country"]] += 1
                if row.get("ocean"):
                    land_ocean_named += 1
                if row.get("crosses_dateline") in ("1", "True", "true"):
                    land_dateline += 1
                try:
                    if int(float(row.get("is_extreme_precip", 0) or 0)) == 1:
                        land_extreme_precip += 1
                except ValueError:
                    pass
                try:
                    if int(float(row.get("is_extreme_max30", 0) or 0)) == 1:
                        land_extreme_max30 += 1
                except ValueError:
                    pass
                # 仅陆地事件全量留存分位数（规模可控）
                try:
                    land_vol.append(float(row["total_water_volume"]))
                except (ValueError, KeyError, TypeError):
                    pass
                try:
                    land_dur.append(float(row["duration_hours"]))
                except (ValueError, KeyError, TypeError):
                    pass
                try:
                    land_grids.append(float(row["n_grids"]))
                except (ValueError, KeyError, TypeError):
                    pass
                if n_land % 10 == 1:      # 10% 等距抽样用于 depth（控制内存）
                    try:
                        land_depth.append(float(row["total_rain_depth"]))
                    except (ValueError, KeyError, TypeError):
                        pass
            else:
                n_ocean += 1
                if row.get("ocean"):
                    ocean_named += 1
    per_year.append({
        "year": year, "era": era, "n_all": n_all, "n_land": n_land,
        "n_ocean": n_ocean, "land_frac": round(n_land / n_all, 6) if n_all else None,
    })
    print(f"{year} ({era}): all={n_all:>7} land={n_land:>7} "
          f"ocean={n_ocean:>8} land_frac={n_land/n_all:.4f}", flush=True)

tot_all = sum(r["n_all"] for r in per_year)
tot_land = sum(r["n_land"] for r in per_year)
tot_ocean = sum(r["n_ocean"] for r in per_year)

land_depth.sort()
land_vol.sort()
land_dur.sort()
land_grids.sort()

stats = {
    "source": SRC,
    "scope": "26 per-year tables, location_type == 内陆",
    "total_all": tot_all,
    "total_land": tot_land,
    "total_ocean": tot_ocean,
    "land_frac_total": round(tot_land / tot_all, 6),
    "land_frac_by_era": {k: round(land_by_era[k]["n"] / v["n"], 6)
                         for k, v in all_by_era.items()},
    "land_by_era": {k: v["n"] for k, v in sorted(land_by_era.items())},
    "all_by_era": {k: v["n"] for k, v in sorted(all_by_era.items())},
    "land_total_water_volume_km3": {
        "p5": quantile(land_vol, 0.05), "p25": quantile(land_vol, 0.25),
        "median": quantile(land_vol, 0.50), "p75": quantile(land_vol, 0.75),
        "p95": quantile(land_vol, 0.95),
        "sum": sum(land_vol) if land_vol else None,
    },
    "land_total_rain_depth_mm_sampled10pct": {
        "n_sampled": len(land_depth),
        "p5": quantile(land_depth, 0.05), "p25": quantile(land_depth, 0.25),
        "median": quantile(land_depth, 0.50), "p75": quantile(land_depth, 0.75),
        "p95": quantile(land_depth, 0.95),
    },
    "land_duration_hours": {
        "p5": quantile(land_dur, 0.05), "median": quantile(land_dur, 0.50),
        "p95": quantile(land_dur, 0.95),
    },
    "land_n_grids": {
        "p5": quantile(land_grids, 0.05), "median": quantile(land_grids, 0.50),
        "p95": quantile(land_grids, 0.95),
    },
    "land_extreme_precip_n": land_extreme_precip,
    "land_extreme_max30_n": land_extreme_max30,
    "land_crosses_dateline_n": land_dateline,
    "land_ocean_field_named_n": land_ocean_named,
    "ocean_field_named_n": ocean_named,
    "land_continent_top10": dict(sorted(land_continent.items(),
                                        key=lambda kv: -kv[1])[:10]),
    "land_climate_zone_all": dict(sorted(land_climate.items(),
                                         key=lambda kv: -kv[1])),
    "land_country_top20": dict(sorted(land_country.items(),
                                      key=lambda kv: -kv[1])[:20]),
    "land_country_n_unique": len(land_country),
    "per_year": per_year,
}

with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(stats, f, ensure_ascii=False, indent=2)

with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["year", "era", "n_all", "n_land",
                                      "n_ocean", "land_frac"])
    w.writeheader()
    w.writerows(per_year)

print("\n=== LAND SUBSET SUMMARY ===")
print(f"total_all   = {tot_all:,}")
print(f"total_land  = {tot_land:,}")
print(f"total_ocean = {tot_ocean:,}")
print(f"land_frac   = {tot_land/tot_all:.4f}")
print("land_by_era =", stats["land_by_era"])
print("land_frac_by_era =", stats["land_frac_by_era"])
print("land_volume median =", stats["land_total_water_volume_km3"]["median"])
print("land_duration median =", stats["land_duration_hours"]["median"])
print("land_continents =", stats["land_continent_top10"])
print(f"wrote {OUT_JSON}")
print(f"wrote {OUT_CSV}")
