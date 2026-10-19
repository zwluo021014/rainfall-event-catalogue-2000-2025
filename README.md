# Rainfall Event Catalogue (2000–2025) — Code & Documentation

A 26-year global rainfall **event** catalogue derived from CMORPH V1.0 ADJ
(8 km / 30 min), with a documented four-era non-homogeneity partition.

**This repository holds code and documentation only — not the 2.6 GB data.**
The data are archived on Zenodo (DOI pending). See `docs/DATA_LOCATION.md`.

## What this record is

Event-based catalogues of precipitation are usually regional, short, or grid-aggregated.
This one spans the full 26-year satellite record at native 0.0728° / 30-min resolution
and preserves the object-level structure of precipitation systems.

The substantive finding is **not** the record size. Across 2000–2025 the catalogue
exhibits three step changes pinned to whole-month boundaries (2005/06, 2014/15,
2020/21). The conservative quantity — monthly water volume — stays within roughly
±10% across eras, while event counts rise up to +44% and median event depth falls
26–29%. The parsimonious reading is that the same rain is being cut into more, smaller
events: a change in the **event-partitioning layer**, not in precipitation physics.

**Practical consequence: analyse within one of the four eras (A/B/C/D), never pooled
across them.** Cross-era trend statements derived from this catalogue are invalid.

## Record at a glance

| Quantity | Value |
|---|---|
| Events | 9,409,781 |
| Period | 2000-01-01 – 2025-12-31 |
| Columns | 29 (identical schema in all 26 files) |
| Ocean / land | 7,789,305 (82.8%) / 1,620,476 (17.2%) |
| Eras | A 2000–2005, B 2006–2014, C 2015–2020, D 2021–2025 |

## Repository contents

```
docs/
  event_delineation_spec_v18.md   Source-level spec of the detection algorithm
  DATA_LOCATION.md                Where the data live and how to cite them
  LAND_SUBSET.md                  Land-subset statistics (Tables 2b/2c of the paper)
code/
  rainfall_event_detection.py     v18.0 — the production detection script (3D-DBSCAN + IETD)
  land_subset_stats.py            Land-subset statistics reported in the paper
  validate_A_pixelgauge.py        L2 — event-detection self-consistency
  validate_B_rawfield.py          L3 — raw-field breakpoint test (localises the steps)
  validate_C_era5.py              L3 — independent ERA5 cross-check
  validate_D_gauge.py             L1 — external GHCN-Daily rain-gauge cross-check
  validate_E_mrms.py              L1 — independent CONUS MRMS radar cross-check
```

## Reproducibility

The detection parameters are **year-independent**: the pipeline branches only on
latitude, Köppen climate zone, and calendar month — never on year. This is what makes
the observed step changes attributable to the input product rather than to a parameter
change in our own code.

`docs/event_delineation_spec_v18.md` specifies the ten-step pipeline
(→ grid event identification by adaptive IETD → cross-day merge → 3D-DBSCAN clustering →
dynamic splitting → attribute assembly → geographic classification → cross-month union →
yearly ID assignment → output) at the level needed for an independent implementation.

## Technical validation

| Level | Check | Result |
|---|---|---|
| L1 | GHCN-Daily gauges (rain-gauge cross-check) | POD 0.441 / FAR 0.518 / CSI 0.299 |
| L1 | CONUS MRMS MultiSensor QPE (radar) | POD 0.411 / CSI 0.176; 56.9% of observed precipitation mass captured |
| L2 | Pixel-as-gauge self-consistency | FAR 0.06–0.23; depth Pearson r ≈ 0.43 |
| L3 | Raw-field breakpoint test | All three steps present in the input field, not the algorithm |
| L3 | ERA5 cross-check | The 2020/21 step is CMORPH-specific (absent in ERA5) |

## Citation

The method paper is in preparation. Until it is published, please cite the dataset:

> Xie, P., Joyce, R., Wu, S., Yoo, S.-H., Yarosh, Y., Sun, F., Lin, R., & NOAA CDR
> Program (2019). *NOAA Climate Data Record (CDR) of CPC Morphing Technique (CMORPH)
> High Resolution Global Precipitation Estimates, Version 1*.
> NOAA National Centers for Environmental Information. https://doi.org/10.25921/w9va-q159

with methods per Xie et al. (2017), https://doi.org/10.1175/JHM-D-16-0168.1.

**Please cite the Zenodo DOI (see `docs/DATA_LOCATION.md`) when it is minted; a DOI is
being reserved and this file will be updated.**

## Known limitations

- **Single input source.** Derived from CMORPH V1.0 ADJ only. There is no independent
  precipitation product over the same record for cross-checking the catalogue itself.
- **Two of three step causes are not closed.** The 2020/21 step is shown to be
  CMORPH-specific by the ERA5 cross-check, but we could not locate a publicly citable
  technical note documenting the change of CMORPH input processing that is often cited
  as its explanation, so we do not rely on that attribution. The 2005/06 and 2014/15
  steps are attributed to passive-microwave observing-system evolution as a *leading
  hypothesis, not established*. In short: **where** the steps come from is established
  by the raw-field and ERA5 diagnostics; **why** they occur is closed only for the PMW
  era transitions.
- **Validation D station coverage is uneven.** 260 gauge files were downloaded and 250
  parsed, but only 136 stations contributed observations inside the 2000–2025 window;
  stations dropped for lack of in-window data are concentrated in Brazil, India and
  Finland. The aggregate numbers are correct as reported, but the sample is not a
  uniform global random sample.
- **CMORPH cold-season bias is inherited.** The source product underestimates snowfall and
  cold-season precipitation at mid- and high latitudes (documented in Xie et al. 2017).
- **Ocean-dominated.** 82.8% of events are oceanic. Hydological users should filter to
  `location_type == inland` (1,620,476 events) before any catchment analysis.

## Licence

Code: MIT. Data: CC-BY-4.0. Third-party inputs retain their own terms
(CMORPH CDR is CC-BY-4.0; GHCN-Daily is US Government public domain).