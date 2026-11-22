#!/bin/bash
# ============================================================================
# iNEXT on Sherlock: one array task per hexagon scale x guild (4 x 7 = 28 tasks)
#   task id = scale_index * 7 + guild_index
#   scales: 0=sf_10 1=sf_100 2=sf_250 3=sf_500
#   guilds: 0=trees 1=water 2=shrubgrass 3=urban 4=native 5=nonnative 6=all
# Submit from the folder that holds inext_sherlock.R:   sbatch inext_sherlock.sbatch
# ============================================================================
#SBATCH --job-name=inext_guilds
#SBATCH --partition=hns            # or your lab/owners partition if you have one
#SBATCH --array=0-27%14               # 28 tasks, at most 14 running at once
#SBATCH --cpus-per-task=8             # hexagons run in parallel across these cores
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=logs/inext_%A_%a.out
#SBATCH --error=logs/inext_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=oliverx@stanford.edu   # uncomment and add your SUNet ID

set -euo pipefail

SCALES=(sf_10 sf_100 sf_250 sf_500)
GUILDS=(trees water shrubgrass urban native nonnative all)
SCALE=${SCALES[$((SLURM_ARRAY_TASK_ID / ${#GUILDS[@]}))]}
GUILD=${GUILDS[$((SLURM_ARRAY_TASK_ID % ${#GUILDS[@]}))]}

NBOOT=${NBOOT:-100}                   # override for a test: sbatch --export=ALL,NBOOT=5 ...
LEVEL=${LEVEL:-0.8}                   # fixed sample coverage for the guild estimates
MIN_CHECKLISTS=${MIN_CHECKLISTS:-5}   # only used for the all-species hexagon curves
DATE_TAG=${DATE_TAG:-$(date +%m_%d_%y)}

IN_DIR=$SCRATCH/inext/input
OUT_DIR=$SCRATCH/inext/results/$SCALE

module purge
module load R/4.6.1                   # check available versions with: module spider R
unset R_LIBS R_LIBS_SITE              # keep the old ~/R_libs (built for an older R) off the library path
export R_LIBS_USER=$HOME/R/4.6        # packages installed once by setup (see README)
export OMP_NUM_THREADS=1              # parallelism comes from mclapply, not BLAS

mkdir -p "$OUT_DIR"
echo "Task $SLURM_ARRAY_TASK_ID -> $SCALE / $GUILD on $(hostname) with $SLURM_CPUS_PER_TASK cores"
echo "R library paths:"; Rscript --vanilla -e 'cat(.libPaths(), sep = "\n")'

Rscript --vanilla inext_sherlock.R "$SCALE" "$GUILD" "$IN_DIR" "$OUT_DIR" "$NBOOT" "$LEVEL" "$MIN_CHECKLISTS" "$DATE_TAG"
