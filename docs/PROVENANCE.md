# File provenance and integrity manifest

The reproducibility claim in the paper depends on shipping **one specific version** of
the detection script. Two versions exist on the working machine and they differ by
2,196 lines. Mixing them up would invalidate the catalogue's provenance.

## Authoritative versions

| File | SHA-256 | Lines | Status |
|---|---|---|---|
| `code/rainfall_event_detection.py` | `78084f7f5c06992ebc66063f2e645d6dedad363c35528b0eff451fcc31017d57` | 6,077 | **v18.0 — the production version. This is the one that generated the catalogue.** |
| ~~v17.0~~ | `9c90e6fa591b897a56495cdb6a72a7e9a0c9d2c59229a5131b3cf8256b3224c9` | 3,881 | **SUPERSEDED — do not ship, do not use for reproduction** |

`code/rainfall_event_detection_v18.0.py` is a byte-identical copy of
`code/rainfall_event_detection.py`, provided so the version is unambiguous from the
filename alone. Verify:

```bash
sha256sum code/rainfall_event_detection.py code/rainfall_event_detection_v18.0.py
# both must print 78084f7f5c06992ebc66063f2e645d6dedad363c35528b0eff451fcc31017d57
```

## Where the superseded copy lives

A v17.0 copy is still present at
`C:\Users\Administrator\Desktop\MouseWithoutBorders\rainfall_event_detection.py`.

That path is inside a **file-synchronisation folder**. It is retained only so the author
can compare versions if needed. It must not reach the public repository. If a v17.0
file is ever found in this repository, treat the release as compromised: the
delineation spec, the parameter table and the published catalogue refer to v18.0.

## Output-directory provenance

v18.0's `CONFIG` specifies `data_dir = E:\date download\CMORPH` and
`output_dir = F:\output_CMORPH_2000_2025`. That output path is what makes v18.0 the
production version: the catalogue in `F:\output_CMORPH_2000_2025` was written by it.

## Validation scripts

| Script | Purpose | Result artifact (local) |
|---|---|---|
| `validate_B_rawfield.py` | raw-field breakpoint test | `validation_B_*.json` |
| `validate_C_era5.py` | ERA5 cross-check | `validation_C_*.json` |
| `validate_A_pixelgauge.py` | event self-consistency | `validate_A.log` |
| `validate_D_gauge.py` | GHCN-Daily gauge cross-check | `validation_D_gauge.json` |
| `validate_E_mrms.py` | CONUS MRMS radar cross-check | `validation_E_mrms_t0.1.json` |
| `validate_formulas_v5.py` | formula self-consistency | `formula_validation_v5.json` |

## Third-party inputs fetched at validation time

These are downloaded by the scripts and deliberately **not** redistributed here:

| Input | Source | Size (local cache) |
|---|---|---|
| GHCN-Daily station inventory | `https://www.ncei.noaa.gov/pub/data/ghcn/daily/ghcnd-inventory.txt` | 36 MB |
| GHCN-Daily station data | `https://www.ncei.noaa.gov/pub/data/ghcn/daily/all/{id}.dly` | 109 MB (260 files) |
| CMORPH V1.0 ADJ NetCDF | NCEI endpoint, see paper Methods | — |

All three are NOAA NCEI public-domain products.