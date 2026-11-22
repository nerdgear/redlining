# iNEXT on Sherlock

Runs the incidence-frequency iNEXT analysis for every hexagon scale × guild as a Slurm job array:
4 scales (sf_10, sf_100, sf_250, sf_500) × 7 guilds (trees, water, shrubgrass, urban, native,
nonnative, all) = 28 tasks. Inside each task, the hexagons run in parallel.

| File | What it does |
|---|---|
| `inext_sherlock.R` | One scale × one guild: diversity at a fixed sample coverage (`LEVEL`, default 0.8) for every hexagon. For `all`, also the grade-level curves, the hexagon curves at a common coverage, and the grade plot |
| `inext_sherlock.sbatch` | Job array `0-27` (task id = scale × 7 + guild), 8 cores / 16 GB / 12 h each |


## 1. Export the matrices (local, in P3_inext.ipynb)

Run the guild cell, the reshape cell and the export cell (see the chat). That writes to
`code/data/SF/inext_input/`:

- `inc_<scale>_<guild>.csv`: species × hex_id, first row = number of checklists
- `inc_grade_<scale>_all.csv`: species × grade, all species

## 2. Copy everything to Sherlock

```bash

# make the folders on Sherlock first (rsync only creates the last folder level)
ssh oliverx@login.sherlock.stanford.edu 'mkdir -p $SCRATCH/inext/input'

# copy the input matrices and the scripts (on your laptop)
rsync -av code/data/SF/inext_input/ oliverx@login.sherlock.stanford.edu:/scratch/users/oliverx/inext/input/
rsync -av code/SF/sherlock/         oliverx@login.sherlock.stanford.edu:/scratch/users/oliverx/inext/


```

`$SCRATCH` is fast, but files not touched for 90 days are purged, so copy the results back
(or to `$GROUP_HOME` / Oak) when the jobs finish.

## 3. One-time R setup on Sherlock

```bash
sh_dev -c 4 -t 01:00:00          # interactive session (don't install on the login node)
module spider R                  # pick a version; update `module load` in the .sbatch to match
module load R/4.6.1
mkdir -p ~/R/4.6                 # must match R's major.minor version
R -e 'install.packages(c("iNEXT","ggplot2","dplyr","stringr","scales"), lib = "~/R/4.6", repos = "https://cloud.r-project.org")'
```

## 4. Test, then submit

```bash
cd $SCRATCH/inext
mkdir -p logs                                   # Slurm won't create the log folder itself

# quick test: sf_500 trees (task 21) and sf_500 all (task 27), 5 bootstraps
sbatch --array=21,27 --time=00:30:00 --export=ALL,NBOOT=5 inext_sherlock.sbatch

# full run
sbatch inext_sherlock.sbatch
squeue -u $USER                                 # check progress
tail -f logs/inext_<jobid>_<taskid>.out
```

Task ids: `scale_index × 7 + guild_index`, with scales `0=sf_10 1=sf_100 2=sf_250 3=sf_500` and guilds
`0=trees 1=water 2=shrubgrass 3=urban 4=native 5=nonnative 6=all`.
For example, to rerun only sf_250 (all guilds): `sbatch --array=14-20 inext_sherlock.sbatch`.

Settings you can change at submit time with `--export=ALL,NAME=value`:
`NBOOT` (default 100), `LEVEL` (0.8), `MIN_CHECKLISTS` (5, only for the all-species curves), `DATE_TAG`.

## Output (`$SCRATCH/inext/results/<scale>/`)

Every guild (same as the notebook's `run_inext_group`):

- `<scale>_ebird_<guild>_inc_covstd_0.8.csv`: q = 0, 1, 2 at 80 % coverage, plus `T_checklists`, `S_obs`, `Q1`, `Q2`, `sc2T`, `beyond_2T`
- `<scale>_ebird_<guild>_inc_datainfo.csv`
- `<scale>_ebird_<guild>_inc_failed.csv`: hexagons iNEXT could not estimate (usually only 2–4 checklists)

`all` only:

- `SF_<size>hex_inc_{datainfo,asymp,sizebased,covbased,covstd}_{grade,hex}_<date>.csv`
- `SF_<size>hex_inext_grade_<date>.rds` and `SF_<size>hex_inext_speciesaccum_grades_<date>.png`

## Notes

- **Same method as the notebook.** Each hexagon is independent in iNEXT, so running them in parallel gives the same estimates as running them one after another.

- **Two coverage levels.** The guild files use the fixed `LEVEL` (0.8) so guilds and scales are comparable.
  The extra `all` hexagon curves use the common level from `estimateD(level = NULL)` (the lowest coverage any
  hexagon reaches at 2× its checklists), as in the earlier all-species run.

- **Filters.** The guild runs use every hexagon in the exported matrix (≥ 2 checklists from the reshape step).
  The extra `all` hexagon curves also need `MIN_CHECKLISTS` (5) checklists and ≥ 2 detected species.

- **Resources.** If a task fails with an out-of-memory error, raise `--mem`; if tasks wait a long time in the
  queue, lower `--cpus-per-task` or the `%14` limit on how many run at once.
