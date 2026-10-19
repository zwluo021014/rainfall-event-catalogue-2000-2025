"""
validate_E_mrms.py - CONUS MRMS (MultiSensor QPE 1H) independent cross-check of the
CMORPH 30-min rainfall-event catalogue (pilot).

Method (pixel/event vs independent radar QPE):
  - For each CONUS CMORPH grid cell (0.072 deg) and each hour:
      * event_present = a CMORPH rainfall event bbox covers the cell that hour
      * mrms_wet      = MRMS MultiSensor QPE (1h accum) >= threshold  (wet = real rain)
    build 2x2 contingency -> POD / FAR / CSI.
  MRMS is an independent, high-resolution (1 km) radar+multi-sensor product, so this
  checks whether the satellite-derived events actually correspond to independently
  observed precipitation over CONUS.

Usage:
  python validate_E_mrms.py --year 2021 --start "2021-02-26 00:00" --end "2021-03-02 00:00" --thr 0.1
"""
import argparse, os, sys, gzip, shutil, pickle, csv, json, math
from datetime import datetime, timedelta
import numpy as np
import urllib.request

ROOT = os.path.dirname(os.path.abspath(__file__))
VENV = r"C:\Users\Administrator\.workbuddy\binaries\python\envs\default"
sys.path.insert(0, os.path.join(VENV, "Lib", "site-packages"))
os.environ["PATH"] = os.path.join(ROOT, "tools/eccodes/Library", "bin") + os.pathsep + os.environ.get("PATH", "")
import rioxarray

# ---- CMORPH CONUS sub-grid (0.072 deg) ----
LON0, LATN, RES = 235.0, 49.0, 0.072
NLON = int(round((293.0 - LON0) / RES)) + 1   # ~806
NLAT = int(round((LATN - 25.0) / RES)) + 1    # ~334
LONS = LON0 + (np.arange(NLON) + 0.5) * RES    # 0-360
LATS = LATN - (np.arange(NLAT) + 0.5) * RES    # descending 49..25

# MRMS 0.01 deg grid extents (from sample)
MRMS_LON0, MRMS_LAT1 = -129.995, 54.995
MRMS_NX, MRMS_NY = 7000, 3500

def local_i(lon):
    return int(min(NLON - 1, max(0, round((lon - LON0) / RES))))

def local_j(lat):
    return int(min(NLAT - 1, max(0, round((LATN - lat) / RES))))

# precompute MRMS nearest-neighbour index into CMORPH grid
_lon_deg = np.where(LONS > 180, LONS - 360, LONS)
I_M = np.clip(np.round((_lon_deg + 129.995) / 0.01).astype(int), 0, MRMS_NX - 1)   # (NLON,)
J_M = np.clip(np.round((54.995 - LATS) / 0.01).astype(int), 0, MRMS_NY - 1)        # (NLAT,)

# ---- CONUS event extraction (cached) ----
def conus_events(year):
    pkl = os.path.join(ROOT, f"conus_events_{year}.pkl")
    if os.path.exists(pkl):
        with open(pkl, "rb") as f:
            return pickle.load(f)
    csvp = os.path.join(ROOT, f"rainfall_events_{year}_fullyear.csv")
    evs = []
    with open(csvp, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                clon = float(row["center_lon"]); clat = float(row["center_lat"])
            except Exception:
                continue
            if not (235.0 <= clon <= 293.0 and 25.0 <= clat <= 49.0):
                continue
            try:
                st = datetime.strptime(row["start_time"], "%Y-%m-%d %H:%M:%S")
                en = datetime.strptime(row["end_time"], "%Y-%m-%d %H:%M:%S")
                lm = float(row["lon_min"]); lx = float(row["lon_max"])
                la = float(row["lat_min"]); lt = float(row["lat_max"])
                m30 = float(row["max_30min_rain"]); ng = int(row["n_grids"])
            except Exception:
                continue
            evs.append(dict(st=st, en=en, lm=lm, lx=lx, la=la, lt=lt, m30=m30, ng=ng))
    with open(pkl, "wb") as f:
        pickle.dump(evs, f)
    print(f"cached {len(evs)} CONUS events for {year}")
    return evs

# ---- MRMS download + read ----
MRMS_BASE = "https://noaa-mrms-pds.s3.amazonaws.com/CONUS/MultiSensor_QPE_01H_Pass1_00.00/"
CACHE = os.path.join(ROOT, "mrms_cache")
os.makedirs(CACHE, exist_ok=True)

def _download_with_retry(url, dest, n_try=6, base=2.0):
    last = None
    for k in range(n_try):
        try:
            urllib.request.urlretrieve(url, dest)
            return
        except Exception as ex:
            last = ex
            if os.path.exists(dest):
                try: os.remove(dest)
                except OSError: pass
            import time
            time.sleep(base * (2 ** k) + 0.5 * k)
    raise RuntimeError(f"MRMS download failed after {n_try} tries: {url} ({last})")

def mrms_file(dt):
    ymd = dt.strftime("%Y%m%d")
    hh = dt.strftime("%H")
    fname = f"MRMS_MultiSensor_QPE_01H_Pass1_00.00_{ymd}-{hh}0000.grib2"
    gz = os.path.join(CACHE, fname + ".gz")
    grb = os.path.join(CACHE, fname)
    if os.path.exists(grb):
        return grb
    if not os.path.exists(gz):
        url = MRMS_BASE + ymd + "/" + fname + ".gz"
        _download_with_retry(url, gz)
    with gzip.open(gz, "rb") as fi, open(grb, "wb") as fo:
        shutil.copyfileobj(fi, fo)
    return grb

def mrms_wet_mask(dt, thr):
    grb = mrms_file(dt)
    da = rioxarray.open_rasterio(grb)
    arr = da.values[0].astype(float)            # (NY, NX) mm, 1h accum
    sub = arr[J_M[:, None], I_M[None, :]]       # (NLAT, NLON)
    sub = np.where(np.isfinite(sub), sub, 0.0)
    return sub >= thr, sub

def vol_capture(em, sub):
    """Fraction of observed MRMS precipitation MASS inside CMORPH event areas."""
    wet = sub > 0.0
    tot = float(np.sum(sub[wet]))
    cap = float(np.sum(sub[wet & em]))
    return (cap / tot) if tot > 0 else 0.0, tot

def event_mask(hour_start, hour_end, events):
    mask = np.zeros((NLAT, NLON), dtype=bool)
    for e in events:
        if e["en"] >= hour_start and e["st"] < hour_end:
            i0, i1 = local_i(e["lm"]), local_i(e["lx"])
            j0, j1 = local_j(e["lt"]), local_j(e["la"])   # lt>la => j0<j1
            if j0 > j1:
                j0, j1 = j1, j0
            if i0 > i1:
                i0, i1 = i1, i0
            mask[j0:j1 + 1, i0:i1 + 1] = True
    return mask

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2021)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--thr", type=float, default=0.1)
    a = ap.parse_args()
    events = conus_events(a.year)
    s = datetime.strptime(a.start, "%Y-%m-%d %H:%M")
    e = datetime.strptime(a.end, "%Y-%m-%d %H:%M")
    hit = miss = fa = cr = 0
    vol_total = 0.0
    vol_captured = 0.0
    per_day = {}
    dt = s
    nsteps = 0
    while dt < e:
        he = dt + timedelta(hours=1)
        em = event_mask(dt, he, events)
        mw, sub = mrms_wet_mask(dt, a.thr)
        hit += int(np.sum(em & mw))
        miss += int(np.sum(~em & mw))
        fa += int(np.sum(em & ~mw))
        cr += int(np.sum(~em & ~mw))
        vc, vt = vol_capture(em, sub)
        vol_total += vt
        vol_captured += vt * vc
        nsteps += 1
        key = dt.strftime("%Y-%m-%d")
        d = per_day.setdefault(key, [0, 0, 0, 0])
        d[0] += int(np.sum(em & mw)); d[1] += int(np.sum(~em & mw))
        d[2] += int(np.sum(em & ~mw)); d[3] += int(np.sum(~em & ~mw))
        dt = he
    pod = hit / (hit + miss) if (hit + miss) else 0
    far = fa / (hit + fa) if (hit + fa) else 0
    csi = hit / (hit + miss + fa) if (hit + miss + fa) else 0
    out = dict(
        year=a.year, start=a.start, end=a.end, thr=a.thr,
        n_hours=nsteps, n_conus_events=len(events),
        hit=hit, miss=miss, fa=fa, cr=cr,
        POD=pod, FAR=far, CSI=csi,
        precip_mass_captured_frac=vol_captured / vol_total if vol_total > 0 else 0.0,
        precip_mass_total_mm=vol_total,
        per_day={k: dict(hit=v[0], miss=v[1], fa=v[2], cr=v[3],
                         POD=v[0]/(v[0]+v[1]) if (v[0]+v[1]) else 0,
                         CSI=v[0]/(v[0]+v[1]+v[2]) if (v[0]+v[1]+v[2]) else 0)
                 for k, v in sorted(per_day.items())},
    )
    with open(os.path.join(ROOT, f"validation_E_mrms_t{a.thr}.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
