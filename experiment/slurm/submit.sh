#!/bin/bash
# Submit the experiment pipeline's SLURM jobs (see experiment/slurm/README.md).
#
#   submit.sh build            build psmc-rs and the tools in $PROD and run every test
#   submit.sh prod             E1 E2 E4 E5 E6 in $PROD/experiment/runs, one allocation
#   submit.sh e3 1 2 3 4 5     E3's timing runs, one exclusive node per repeat
#   submit.sh e3-report        E3's tables, once merge_e3.sh merged the repeats
#
# Arguments after `--` go to sbatch (e.g. `submit.sh prod -- --begin=now+1hour`).
#
# Every job waits for a build job that is still queued or running. prod and
# e3-report share one job name and run one at a time: both start a runner in
# $PROD's runs directory, which holds one runner. A resubmitted E3 repeat goes
# back to the node its first runs were timed on.
set -euo pipefail
DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=/dev/null
source "$DIR/site.env"
mkdir -p "$LOGS"
PY=$MAIN/experiment/.venv/bin/python

what=${1:?usage: submit.sh build|prod|e3 REPEAT...|e3-report [-- SBATCH_ARGS]}
shift
args=() extra=()
while [ $# -gt 0 ]; do
  if [ "$1" = "--" ]; then shift; extra=("$@"); break; fi
  args+=("$1"); shift
done
if [ "$what" != e3 ] && [ ${#args[@]} -gt 0 ]; then
  echo "submit.sh $what takes no arguments (sbatch options go after --): ${args[*]}" >&2; exit 2
fi

# Wait for the last build job if it is still queued or running.
deps=()
if [ "$what" != build ] && [ -f "$LOGS/build.jobid" ]; then
  b=$(cat "$LOGS/build.jobid")
  if squeue -h -j "$b" -o %i 2> /dev/null | grep -q .; then deps+=("afterok:$b"); fi
fi
dependency() { local IFS=,; [ $# -gt 0 ] && echo "--dependency=$*"; }

common=(--account="$ACCOUNT" --partition="$PARTITION" --chdir="$PROD" --nodes=1 --ntasks=1 --parsable)
run() { sbatch "${common[@]}" "$@" ${extra[@]+"${extra[@]}"}; }

case $what in
  build)
    id=$(run --job-name=psmcrs-build --constraint="$PROD_CONSTRAINT" --cpus-per-task=32 --mem=96G \
        --time=02:00:00 --output="$LOGS/%x-%j.out" --export=ALL,PSMC_SLURM_DIR="$DIR" "$DIR/build.sbatch")
    echo "$id" > "$LOGS/build.jobid" ;;
  prod)
    id=$(run --job-name=psmcrs-runs $(dependency singleton ${deps[@]+"${deps[@]}"}) \
        --constraint="$PROD_CONSTRAINT" --cpus-per-task="$PROD_CPUS" --mem="$PROD_MEM" --time="$PROD_TIME" \
        --output="$LOGS/psmcrs-prod-%j.out" --export=ALL,PSMC_SLURM_DIR="$DIR" "$DIR/prod.sbatch") ;;
  e3)
    [ ${#args[@]} -gt 0 ] || { echo "submit.sh e3 REPEAT..." >&2; exit 2; }
    repeats=$(cd "$PROD" && "$PY" -c 'from experiment.pipeline import config; print(config.get_profile("production").e3.repeats)')
    for r in "${args[@]}"; do
      if ! [[ $r =~ ^[1-9][0-9]*$ ]] || [ "$r" -gt "$repeats" ]; then
        echo "submit.sh e3: repeat '$r' is not one of 1..$repeats" >&2; exit 2
      fi
    done
    id=""
    for r in "${args[@]}"; do
      # The node of this repeat's earlier runs, if any were timed.
      host=$(cd "$PROD" && "$PY" - "$E3_SHARDS/rep$r" <<'EOF'
import json, pathlib, sys
hosts = {json.loads(p.read_text()).get("machine", {}).get("hostname")
         for p in pathlib.Path(sys.argv[1], "E3", "timings").rglob("*.json")}
hosts.discard(None)
if len(hosts) > 1:
    sys.exit(f"repeat timed on several hosts: {sorted(hosts)}")
print(next(iter(hosts), ""))
EOF
)
      pin=()
      [ -n "$host" ] && pin=(--nodelist="$host") && echo "repeat $r: back to $host"
      id+="$(run --job-name="psmcrs-e3-rep$r" $(dependency singleton ${deps[@]+"${deps[@]}"}) ${pin[@]+"${pin[@]}"} \
          --constraint="$E3_CONSTRAINT" --exclusive --mem=0 --no-requeue --time="$E3_TIME" \
          --output="$LOGS/%x-%j.out" --export=ALL,PSMC_SLURM_DIR="$DIR",E3_REPEAT="$r" "$DIR/e3.sbatch") "
    done ;;
  e3-report)
    id=$(run --job-name=psmcrs-runs $(dependency singleton ${deps[@]+"${deps[@]}"}) \
        --cpus-per-task=2 --mem=16G --time=01:00:00 \
        --output="$LOGS/psmcrs-e3-report-%j.out" --export=ALL,PSMC_SLURM_DIR="$DIR" "$DIR/e3_report.sbatch") ;;
  *)
    echo "unknown job '$what': build, prod, e3 or e3-report" >&2; exit 2 ;;
esac
echo "submitted $what: ${id}${deps[*]:+ (after build ${deps[*]#afterok:})}"
