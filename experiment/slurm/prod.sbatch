#!/bin/bash
# E1 E2 E4 E5 E6 in one allocation: one runner, in the foreground, within the
# allocation's CPUs and memory. Submit with experiment/slurm/submit.sh prod.
#
# 90 minutes before the time limit SLURM sends USR1 to the runner, which then
# starts no new job; at the limit, jobs still running are stopped and stay
# pending. Submitting again runs only what is not done.
#SBATCH --signal=B:USR1@5400
source "$PSMC_SLURM_DIR/env.sh" || exit 1
set -euo pipefail
cd "$PROD"
# A process tree sample of every running job each second, not 20 a second:
# with 60 jobs at once the samples would keep the runner busy.
export PSMC_EXP_SAMPLE_SECONDS=1

# The allocation's CPUs (not nproc, which env.sh's OMP_NUM_THREADS=1 would
# override), two of them for the runner itself; 90% of its memory.
cores=$(( ${SLURM_CPUS_PER_TASK:?need sbatch --cpus-per-task} - 2 ))
mem_gb=$(( ${SLURM_MEM_PER_NODE:?need sbatch --mem} * 9 / 10 / 1024 ))
echo "$(hostname) job $SLURM_JOB_ID commit $(git rev-parse --short HEAD): $cores cores, $mem_gb GB for jobs"
psmc_env_report
# exec: the runner gets SLURM's USR1 itself (sbatch --signal=B: signals the batch shell).
exec "$PY" -m experiment.pipeline start --foreground --no-keep-awake --profile production \
  --runs-dir "$PROD/experiment/runs" --cores "$cores" --mem-gb "$mem_gb" E1 E2 E4 E5 E6
