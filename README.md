# SEARCH Investigation

Cluster-ready smoke-test pipeline for breast cancer prediction tasks using the local SEARCH molecular datasets copied into this directory.

## Scope

- Works from the local `SEARCH/` directory only.
- Keeps raw and patient-level data out of git.
- Focuses first on BRIDGES-derived molecular tables.
- Supports optional array preprocessing for `iCOGS` and `OncoArray` when `plink` or `plink2` is available.

## Quick start

Build manifests:

```bash
python3 scripts/build_manifests.py
```

Run one local smoke test:

```bash
python3 scripts/run_smoke_test.py \
  --task grade_high_vs_low \
  --modality bridges_truncating \
  --model logistic_l2 \
  --seed 42
```

Render or submit Slurm jobs:

```bash
python3 scripts/launch_smoke_tests.py --tier tier1
python3 scripts/launch_smoke_tests.py --tier tier1 --submit
```

Summarize completed runs:

```bash
python3 scripts/summarize_results.py
```
