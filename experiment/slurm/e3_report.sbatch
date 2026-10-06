#!/bin/bash
# E3's tables from the merged repeats (merge_e3.sh first). It starts a runner
# in $PROD's runs directory, like prod: submit.sh gives both one job name, so
# that SLURM never runs them at once. Submit with submit.sh e3-report.
source "$PSMC_SLURM_DIR/env.sh" || exit 1
set -euo pipefail
cd "$PROD"
psmc_env_report
exec "$PY" -m experiment.pipeline start --foreground --no-keep-awake --profile production \
  --runs-dir "$PROD/experiment/runs" --cores 2 --mem-gb 14 --only '^E3/report$' E3
