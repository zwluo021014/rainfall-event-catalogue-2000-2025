# Where the data are, and why they are not in this repository

## The short answer

The catalogue is ~2.6 GB as 26 per-year CSV files, and the largest single file is
**125.7 MB** (`rainfall_events_2012_fullyear.csv`). GitHub enforces a **hard 100 MB
per-file limit** — pushes above it are rejected outright. Twelve of the 26 files exceed
100 MB, so the data cannot be committed to this repository without either splitting
every file (which would break the schema-consistency guarantee the paper makes) or
paying for Git LFS.

Therefore: **code and documentation here, data on Zenodo.**

## Recommendation: GitHub + Zenodo, in two distinct roles

| | GitHub (this repo) | Zenodo |
|---|---|---|
| Holds | code, spec, validation scripts, docs | the 26 CSVs + data dictionary |
| Purpose | browsable, versioned, citable-as-software | DOI, long-term archival, FAIR |
| Free? | yes | yes (50 GB per record) |
| Limitation | 100 MB/file hard limit | none of practical concern here |

This is the standard pairing for data papers and is what Scientific Data reviewers
expect: a public, versioned code repository plus a DOI-bearing data archive.

### Why not Git LFS

Git LFS would work (2 GB per file, 1 GB free storage + bandwidth on a free account),
but it introduces friction for exactly the audience that matters most here — hydrologists
who want to `curl` a file. LFS also requires the consumer to have `git-lfs` installed,
and GitHub's LFS bandwidth quota is 1 GB/month on free accounts: downloading the full
2.6 GB catalogue would exhaust a month's quota. **Zenodo serves plain HTTP file
downloads with no client tooling, which is the right affordance for the data.**

### Why the data should not go in the repository even if they fit

A `git` repository is a poor container for a 2.6 GB append-only dataset: every clone
pulls the whole history, and any future correction rewrites large blobs. Zenodo gives
each release its own DOI and a stable landing page, which is what the paper's Data
availability statement needs.

## What to upload where

**Zenodo record (data):**
- `rainfall_events_2000_fullyear.csv` … `rainfall_events_2025_fullyear.csv` (26 files)
- data dictionary (revised)
- `land_subset_stats.json`, `land_subset_by_year.csv` — pre-computed land statistics so
  users need not re-scan 2.6 GB to reproduce Tables 2b/2c

**GitHub (code + docs, this repository):**
- everything in `code/` and `docs/`
- the 26-file schema note, so a user can verify the concatenation guarantee themselves

**Never upload:**
- `DEFECTIVE_DO_NOT_USE_rainfall_events_2000_2025_multiyear.csv` — this assembly is
  incomplete (contains 2012–2025 only, 5,117,196 rows instead of 9,409,781). It is
  retained locally for forensics only. Use the 26 per-year files.

**Do not upload** `ghcnd_dly/` (109 MB, 260 GHCN station files) or `ghcnd-inventory.txt`
(36 MB). These are third-party NOAA NCEI public-domain files that the validation script
re-downloads automatically from NCEI. Uploading them adds bulk and creates a
redistribution question that does not need to be asked.

## Verifying the concatenation guarantee

The paper states that all 26 files share one identical 29-column schema. A user can
check this before downloading everything:

```bash
for y in $(seq 2000 2025); do
  head -1 "rainfall_events_${y}_fullyear.csv" | md5sum
done | sort -u
```

This prints a single line if the schema claim holds.

## Citation placeholder

> Author(s). (2026). A 26-year (2000–2025) global rainfall event catalogue from
> CMORPH V1.0 ADJ 8 km/30 min, with documented step-change inhomogeneities.
> Zenodo. https://doi.org/10.5281/zenodo.XXXXXXX

**The DOI has not been minted yet.** `REPO_README.md` and
`manuscript_draft.md` both contain `XXXXXXX` placeholders that must be replaced before
submission. Minting the Zenodo record is an author action and cannot be automated from
here.