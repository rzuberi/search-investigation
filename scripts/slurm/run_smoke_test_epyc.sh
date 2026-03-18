#!/bin/bash
#SBATCH -J search-smoke
#SBATCH -p epyc
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH -o outputs/logs/slurm/%x-%j.out
#SBATCH -e outputs/logs/slurm/%x-%j.err

set -euo pipefail

cd /mnt/scratchc/fmlab/zuberi01/phd/search-investigation

python3 scripts/run_smoke_test.py \
  --task "${TASK:?set TASK}" \
  --modality "${MODALITY:?set MODALITY}" \
  --model "${MODEL:?set MODEL}" \
  --seed "${SEED:-42}" \
  --n-jobs "${N_JOBS:-8}"
