#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Validation D — Level-1 EXTERNAL validation of the CMORPH rainfall-event catalogue
against real rain gauges (GHCN-Daily, NOAA NCEI).

Design
------
We turn the pixel-as-gauge self-consistency test (Validation A) into a true *external*
test by replacing the CMORPH pixel with an independent ground-truth gauge pixel.

For every selected gauge we build a daily "wet/dry" series from GHCN-Daily PRCP.
From the catalogue we build, for the gauge's nearest CMORPH grid cell, the set of
calendar days on which at least one event is present (event *occurrence* proxy).
A 2x2 contingency (gauge wet/dry x catalogue event-present/absent) per gauge-day
yields POD / FAR / CSI. Aggregated overall and by era (A 2000-05, B 2006-14,
C 2015-20, D 2021-25).

Only the union of gauge target cells is materialised while streaming the catalogue,
so memory stays small even for the full 26-year record.

Usage
-----
  python validate_D_gauge.py --sample 200 --years 2000-2025
  python validate_D_gauge.py --sample 30  --years 2015,2020   # quick pilot
"""
import os, sys, json, argparse, datetime, collections
import numpy as np
import pandas as pd
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
GHCN_INV_URL = "https://www.ncei.noaa.gov/pub/data/ghcn/daily/ghcnd-inventory.txt"
GHCN_DLY_URL = "https://www.ncei.noaa.gov/pub/data/ghcn/daily/all/{id}.dly"
SRC_NC = r"E:/date download/CMORPH/2010/01/01/CMORPH_V1.0_ADJ_8km-30min_2010010100.nc"
LAT_LIMIT = 60.0

ERA = {}  # year -> era label
for y in range(2000, 2006): ERA[y] = "A"
for y in range(2006, 2015): ERA[y] = "B"
for y in range(2015, 2021): ERA[y] = "C"
for y in range(2021, 2026): ERA[y] = "D"


def log(*a):
    print(*a, flush=True)


# ---------------- CMORPH coordinate arrays ----------------
def load_coords():
    import xarray as xr
    ds = xr.open_dataset(SRC_NC)
    lat = np.asarray(ds["lat"].values, dtype=float)
    lon = np.asarray(ds["lon"].values, dtype=float)
    ds.close()
    return lat, lon


def cell_index(latv, lonv, LAT, LON):
    """nearest CMORPH grid cell index for a geographic point (analytic, uniform grid)."""
    LAT_max = LAT.max(); dlat = abs(LAT[0] - LAT[1])
    ilat = int(np.clip(round((LAT_max - latv) / dlat), 0, LAT.size - 1))
    dlon = LON[1] - LON[0]
    ilon = int(np.clip(round((lonv % 360.0) / dlon), 0, LON.size - 1))
    return ilat * LON.size + ilon


# ---------------- GHCN inventory / station select ----------------
def select_gauges(sample_n, min_span_years=20):
    inv = os.path.join(HERE, "ghcnd-inventory.txt")
    if not os.path.exists(inv):
        log("download inventory ...")
        urllib.request.urlretrieve(GHCN_INV_URL, inv)
    recs = []
    with open(inv) as f:
        for line in f:
            t = line.split()
            if len(t) < 6 or t[3] != "PRCP":
                continue
            try:
                sid = t[0]
                lat = float(t[1]); lon = float(t[2])
                fy = int(t[4]); ly = int(t[5])
            except ValueError:
                continue
            if abs(lat) > LAT_LIMIT:
                continue
            if ly - fy < min_span_years:
                continue
            recs.append((sid, lat, lon, fy, ly))
    log(f"PRCP stations with >= {min_span_years}y record in +-{LAT_LIMIT}: {len(recs)}")
    # stratified sample by latitude band for global spread
    bands = [(-60, -30), (-30, 0), (0, 30), (30, 60)]
    per = max(1, sample_n // len(bands))
    chosen = []
    for lo, hi in bands:
        grp = [r for r in recs if lo <= r[1] < hi]
        rng = np.random.default_rng(42)
        if len(grp) > per:
            grp = [grp[i] for i in rng.choice(len(grp), per, replace=False)]
        chosen += grp
    if len(chosen) < sample_n:  # top up globally
        rng = np.random.default_rng(7)
        extra = [r for r in recs if r not in chosen]
        chosen += [extra[i] for i in rng.choice(len(extra), min(sample_n - len(chosen), len(extra)), replace=False)]
    log(f"selected gauges: {len(chosen)}")
    return chosen


def download_dly(sid):
    out = os.path.join(HERE, "ghcnd_dly", sid + ".dly")
    if os.path.exists(out):
        return out
    os.makedirs(os.path.dirname(out), exist_ok=True)
    try:
        urllib.request.urlretrieve(GHCN_DLY_URL.format(id=sid), out)
        return out
    except Exception as e:
        return None


def parse_dly(path):
    """Return dict {(y,m,d): prcp_tenths_mm} for PRCP element.
    GHCN .dly fixed-width: ID(11) YEAR(4) MONTH(2) ELEM(4), then 31 days each
    occupying 8 chars = 5-digit value (tenths mm) + 3 flag chars."""
    series = {}
    with open(path) as f:
        for line in f:
            if len(line) < 29 or line[17:21].strip() != "PRCP":
                continue
            try:
                y = int(line[11:15]); m = int(line[15:17])
            except ValueError:
                continue
            for i in range(31):
                s = line[21 + i * 8: 26 + i * 8]
                try:
                    v = int(s)
                except ValueError:
                    continue
                if v == -9999:
                    continue
                series[(y, m, i + 1)] = v
    return series


# ---------------- catalogue event-occurrence per gauge cell ----------------
def _idx_grid(vals, axis_max, d):
    return np.clip(np.round((axis_max - vals) / d).astype(int), 0, None)


def build_catalogue_wet(cat_years, gauge_cells):
    """Mark, for each gauge cell, the set of calendar days on which an event's
    bounding box covers the gauge point (event *coverage* proxy, not center-only)."""
    LAT_max = LAT.max(); dlat = abs(LAT[0] - LAT[1])
    dlon = LON[1] - LON[0]
    wet = {c: set() for c in gauge_cells}
    gcell = np.array(list(gauge_cells.keys()))
    gilat = np.array([gauge_cells[c][0] for c in gcell])
    gilon = np.array([gauge_cells[c][1] for c in gcell])
    for yr in cat_years:
        fn = os.path.join(HERE, f"rainfall_events_{yr}_fullyear.csv")
        if not os.path.exists(fn):
            log(f"  skip missing {fn}")
            continue
        log(f"  stream {yr} ...")
        for ch in pd.read_csv(fn, usecols=["start_time", "lat_min", "lat_max",
                                           "lon_min", "lon_max"], chunksize=500000):
            dt = pd.to_datetime(ch["start_time"], errors="coerce")
            dk = dt.dt.strftime("%Y-%m-%d").values
            ok = (~np.isnan(ch["lat_min"].values)) & (~np.isnan(ch["lon_min"].values))
            if not ok.any():
                continue
            lat_min_idx = _idx_grid(ch["lat_min"].values, LAT_max, dlat)
            lat_max_idx = _idx_grid(ch["lat_max"].values, LAT_max, dlat)
            lon_min_idx = np.clip(np.round((ch["lon_min"].values % 360.0) / dlon).astype(int), 0, LON.size - 1)
            lon_max_idx = np.clip(np.round((ch["lon_max"].values % 360.0) / dlon).astype(int), 0, LON.size - 1)
            lat_min_idx = np.clip(lat_min_idx, 0, LAT.size - 1)
            lat_max_idx = np.clip(lat_max_idx, 0, LAT.size - 1)
            lon_min_idx = np.clip(lon_min_idx, 0, LON.size - 1)
            lon_max_idx = np.clip(lon_max_idx, 0, LON.size - 1)
            for j in range(len(gcell)):
                # CMORPH latitude decreases with index: northern bound = lat_max_idx (smaller idx)
                lat_ok = (lat_max_idx <= gilat[j]) & (gilat[j] <= lat_min_idx)
                if not lat_ok.any():
                    continue
                lmin = lon_min_idx[lat_ok]; lmax = lon_max_idx[lat_ok]
                # longitude wrap (event crossing 0/360)
                lon_ok = np.where(lmin <= lmax,
                                  (lmin <= gilon[j]) & (gilon[j] <= lmax),
                                  (gilon[j] >= lmin) | (gilon[j] <= lmax))
                hit = np.where(lat_ok)[0][lon_ok]
                for h in hit:
                    wet[gcell[j]].add(dk[h])
    return wet


# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=200)
    ap.add_argument("--years", default="2000-2025")
    ap.add_argument("--min_span", type=int, default=20)
    args = ap.parse_args()
    if "-" in args.years and args.years.count("-") == 1 and args.years.replace("-", "").isdigit():
        a, b = args.years.split("-")
        cat_years = list(range(int(a), int(b) + 1))
    else:
        cat_years = [int(y) for y in args.years.split(",") if y.strip()]

    global LAT, LON
    LAT, LON = load_coords()
    log(f"CMORPH grid {LAT.size} x {LON.size}")

    gauges = select_gauges(args.sample, args.min_span)
    gauge_cells = {}
    cell_sid = {}
    gauge_series = {}
    for sid, lat, lon, fy, ly in gauges:
        p = download_dly(sid)
        if not p:
            continue
        ser = parse_dly(p)
        if not ser:
            continue
        ci = cell_index(lat, lon, LAT, LON)
        gauge_cells[ci] = (int(ci // LON.size), int(ci % LON.size))
        cell_sid[ci] = sid
        gauge_series[sid] = (ser, fy, ly)
    log(f"gauges with data: {len(gauge_series)} -> cells {len(gauge_cells)}")

    wet = build_catalogue_wet(cat_years, gauge_cells)
    streamed = set(cat_years)

    # contingency
    agg = collections.Counter()            # overall
    by_era = collections.defaultdict(collections.Counter)
    per_gauge = {}
    for ci, sid in cell_sid.items():
        ser, fy, ly = gauge_series[sid]
        wset = wet.get(ci, set())
        a = b = c = d = 0  # hit, miss, false_alarm, correct_neg
        for (y, m, day), prcp in ser.items():
            if y not in ERA:
                continue
            if y not in streamed:
                continue
            dk = f"{y:04d}-{m:02d}-{day:02d}"
            truth_wet = prcp > 0
            cat_wet = dk in wset
            if truth_wet and cat_wet:
                a += 1
            elif truth_wet and not cat_wet:
                b += 1
            elif (not truth_wet) and cat_wet:
                c += 1
            else:
                d += 1
            e = ERA.get(y, "?")
            by_era[e]["hit"] += (1 if (truth_wet and cat_wet) else 0)
            by_era[e]["miss"] += (1 if (truth_wet and not cat_wet) else 0)
            by_era[e]["fa"] += (1 if (not truth_wet and cat_wet) else 0)
            by_era[e]["cn"] += (1 if (not truth_wet and not cat_wet) else 0)
        n = a + b + c + d
        if n == 0:
            continue
        pod = a / (a + b) if (a + b) else None
        far = c / (a + c) if (a + c) else None
        csi = a / (a + b + c) if (a + b + c) else None
        per_gauge[sid] = {"era_range": f"{fy}-{ly}", "n": n,
                          "POD": round(pod, 3) if pod is not None else None,
                          "FAR": round(far, 3) if far is not None else None,
                          "CSI": round(csi, 3) if csi is not None else None}
        agg["hit"] += a; agg["miss"] += b; agg["fa"] += c; agg["cn"] += d

    def metrics(ct):
        a, b, c, d = ct["hit"], ct["miss"], ct["fa"], ct["cn"]
        return {"POD": round(a / (a + b), 3) if (a + b) else None,
                "FAR": round(c / (a + c), 3) if (a + c) else None,
                "CSI": round(a / (a + b + c), 3) if (a + b + c) else None,
                "n_gauge_days": a + b + c + d}

    out = {
        "meta": {"sample_n": args.sample, "cat_years": cat_years,
                 "n_gauges_used": len(per_gauge), "lat_limit": LAT_LIMIT,
                 "note": "event-occurrence proxy: catalogue wet-day = an event whose bounding box covers the gauge's nearest CMORPH cell"},
        "overall": metrics(agg),
        "by_era": {e: metrics(ct) for e, ct in by_era.items()},
        "per_gauge": per_gauge,
    }
    with open(os.path.join(HERE, "validation_D_gauge.json"), "w") as f:
        json.dump(out, f, indent=2)
    log("=== Validation D (gauge) overall ===", out["overall"])
    for e in sorted(out["by_era"]):
        log(f"  era {e}:", out["by_era"][e])
    log("written validation_D_gauge.json")


if __name__ == "__main__":
    main()
