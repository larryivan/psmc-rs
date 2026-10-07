#!/bin/bash
# Submit the experiment pipeline's SLURM jobs (see experiment/slurm/README.md).
#
#   submit.sh build            build psmc-rs and the tools in $PROD and run every test
#   submit.sh prod             E1 E2 E4 E5 E6 in $PROD/experiment/runs, one allocation
#   submit.sh e3 1 2 3 4 5     E3's timing runs, one exclusive node per repeat
#   submit.sh e3-report        E3's tables, once merge_e3.sh merged the repeats
#   submit.sh e7-prep [N...]   E7's data stage: setup, chromosomes N (default 1-22), samples
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

what=${1:?usage: submit.sh build|prod|e3 REPEAT...|e3-report|e7-prep [CHROM...] [-- SBATCH_ARGS]}
shift
args=() extra=()
while [ $# -gt 0 ]; do
  if [ "$1" = "--" ]; then shift; extra=("$@"); break; fi
  args+=("$1"); shift
done
if [ "$what" != e3 ] && [ "$what" != e7-prep ] && [ ${#args[@]} -gt 0 ]; then
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
# sbatch options, then the extra ones (which override them), then the script:
# anything after the script would be the script's own arguments.
run() { sbatch "${common[@]}" "${@:1:$#-1}" ${extra[@]+"${extra[@]}"} "${@: -1}"; }

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
  e7-prep)
    # Setup, then one task per chromosome (E7_PREP_PARALLEL at once: each
    # downloads its VCF), then the samples once every chromosome is done.
    chroms=${args[*]:-$(seq -s ' ' 1 22)}
    for c in $chroms; do
      [[ $c =~ ^([1-9]|1[0-9]|2[0-2])$ ]] || { echo "submit.sh e7-prep: chromosome '$c' is not one of 1..22" >&2; exit 2; }
    done
    stage() { echo "--export=ALL,PSMC_SLURM_DIR=$DIR,E7_STAGE=$1"; }
    setup=$(run --job-name=psmcrs-e7-setup $(dependency ${deps[@]+"${deps[@]}"}) --cpus-per-task=2 --mem=8G \
        --time=03:00:00 --output="$LOGS/%x-%j.out" "$(stage setup)" "$DIR/e7_prep.sbatch")
    chrom=$(run --job-name=psmcrs-e7-chrom --dependency=afterok:"$setup" \
        --array="${chroms// /,}%${E7_PREP_PARALLEL:-4}" --constraint="$PROD_CONSTRAINT" \
        --cpus-per-task="${E7_PREP_CPUS:-16}" --mem="${E7_PREP_MEM:-64G}" --time="${E7_PREP_TIME:-12:00:00}" \
        --output="$LOGS/%x-%A_%a.out" "$(stage chrom)" "$DIR/e7_prep.sbatch")
    samples=$(run --job-name=psmcrs-e7-samples --dependency=afterok:"$chrom" --cpus-per-task=16 --mem=32G \
        --time=04:00:00 --output="$LOGS/%x-%j.out" "$(stage samples)" "$DIR/e7_prep.sbatch")
    id="$setup $chrom $samples" ;;
  *)
    echo "unknown job '$what': build, prod, e3, e3-report or e7-prep" >&2; exit 2 ;;
esac
echo "submitted $what: ${id}${deps[*]:+ (after build ${deps[*]#afterok:})}"
