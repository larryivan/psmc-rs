#!/bin/bash
# Submit the experiment pipeline's SLURM jobs: submit.sh --help, and README.md.
set -euo pipefail

usage() {
  cat << 'EOF'
Usage: submit.sh JOB [ARG...] [-- SBATCH_OPTION...]

  build            build psmc-rs and C psmc in $PROD and run every test
  prod             E1 E2 E4 E5 E6 (and E7 once e7-prep is done) in
                   $PROD/experiment/runs, one allocation
  e3 REPEAT...     E3's timing runs of these repeats, one exclusive node
                   each (e3 1 2 3 4 5: all of them)
  e3-report        E3's tables, once merge_e3.sh merged the repeats
  e7-prep [N...]   E7's inputs: setup, chromosomes N (default 1-22), samples

Options after -- go to sbatch (e.g. submit.sh prod -- --begin=now+1hour).

Every job waits for a build job that is still queued or running. prod and
e3-report share one job name and run one at a time: both start a runner in
$PROD's runs directory, which holds one runner. A resubmitted E3 repeat goes
back to the node its first runs were timed on. The settings are in site.env,
copied from site.env.example; see README.md.
EOF
}
die() { echo "submit.sh: $*" >&2; exit 2; }

case ${1:-} in
  '') usage >&2; exit 2 ;;
  -h | --help) usage; exit 0 ;;
esac
what=$1
shift
args=() extra=()
while [ $# -gt 0 ]; do
  if [ "$1" = "--" ]; then shift; extra=("$@"); break; fi
  args+=("$1"); shift
done
case $what in
  e3 | e7-prep) ;;
  build | prod | e3-report)
    [ ${#args[@]} -eq 0 ] || die "$what takes no arguments (sbatch options go after --): ${args[*]}" ;;
  *) die "unknown job '$what': build, prod, e3, e3-report or e7-prep (submit.sh --help)" ;;
esac

DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
[ -f "$DIR/site.env" ] || die "$DIR/site.env is missing: copy site.env.example there and fill it in"
# shellcheck source=/dev/null
source "$DIR/site.env"
mkdir -p "$LOGS"
PY=$MAIN/experiment/.venv/bin/python

# The last build job, while it is queued or running: every other job waits for it.
build_job=""
if [ "$what" != build ] && [ -f "$LOGS/build.jobid" ]; then
  b=$(cat "$LOGS/build.jobid")
  # The user's jobs: squeue -j fails for a job that has left the queue, as when squeue fails.
  queue=$(squeue -h -u "$USER" -o %i) ||
    die "cannot tell whether build job $b is still queued or running: squeue failed (above)"
  if grep -qxF "$b" <<< "$queue"; then build_job=$b; fi
fi
# prod, e3-report and each E3 repeat: one job of a name at a time, after the build.
singleton=--dependency=singleton${build_job:+,afterok:$build_job}
job_env=--export=ALL,PSMC_SLURM_DIR=$DIR

common=(--account="$ACCOUNT" --partition="$PARTITION" --chdir="$PROD" --nodes=1 --ntasks=1 --parsable)
# sbatch options, then the extra ones (which override them), then the script:
# anything after the script would be the script's own arguments.
run() { sbatch "${common[@]}" "${@:1:$#-1}" ${extra[@]+"${extra[@]}"} "${@: -1}"; }

# The node that the runs of E3 repeat $1 were timed on, if any were.
timed_host() {
  "$PY" - "$E3_SHARDS/rep$1" "$1" << 'EOF'
import json, pathlib, sys
hosts = {json.loads(p.read_text()).get("machine", {}).get("hostname")
         for p in pathlib.Path(sys.argv[1], "E3", "timings").rglob("*.json")}
hosts.discard(None)
if len(hosts) > 1:
    sys.exit(f"submit.sh e3: repeat {sys.argv[2]} was timed on several hosts: {', '.join(sorted(hosts))}")
print(next(iter(hosts), ""))
EOF
}

case $what in
  build)
    id=$(run --job-name=psmcrs-build --constraint="$PROD_CONSTRAINT" --cpus-per-task=32 --mem=96G \
        --time=02:00:00 --output="$LOGS/%x-%j.out" "$job_env" "$DIR/build.sbatch")
    echo "$id" > "$LOGS/build.jobid" ;;
  prod)
    id=$(run --job-name=psmcrs-runs "$singleton" \
        --constraint="$PROD_CONSTRAINT" --cpus-per-task="$PROD_CPUS" --mem="$PROD_MEM" --time="$PROD_TIME" \
        --output="$LOGS/psmcrs-prod-%j.out" "$job_env" "$DIR/prod.sbatch") ;;
  e3)
    [ ${#args[@]} -gt 0 ] || die "e3 needs the repeats to run, e.g. submit.sh e3 1 2 3 4 5"
    repeats=$(cd "$PROD" && "$PY" -c 'from experiment.pipeline import config; print(config.get_profile("production").e3.repeats)')
    # Every repeat checked, and the node of its timed runs found, before any is submitted.
    hosts=()
    for r in "${args[@]}"; do
      if ! [[ $r =~ ^[1-9][0-9]*$ ]] || [ "$r" -gt "$repeats" ]; then
        die "e3: repeat '$r' is not one of 1..$repeats"
      fi
      host=$(timed_host "$r")
      hosts+=("$host")
    done
    ids=()
    for i in "${!args[@]}"; do
      r=${args[$i]} host=${hosts[$i]}
      [ -z "$host" ] || echo "repeat $r: back to $host"
      jid=$(run --job-name="psmcrs-e3-rep$r" "$singleton" ${host:+"--nodelist=$host"} \
          --constraint="$E3_CONSTRAINT" --exclusive --mem=0 --no-requeue --time="$E3_TIME" \
          --output="$LOGS/%x-%j.out" "$job_env,E3_REPEAT=$r" "$DIR/e3.sbatch")
      ids+=("$jid")
    done
    id=${ids[*]} ;;
  e3-report)
    id=$(run --job-name=psmcrs-runs "$singleton" \
        --cpus-per-task=2 --mem=16G --time=01:00:00 \
        --output="$LOGS/psmcrs-e3-report-%j.out" "$job_env" "$DIR/e3_report.sbatch") ;;
  e7-prep)
    # Setup, then one task per chromosome (E7_PREP_PARALLEL at once: each
    # downloads its VCF), then the samples once every chromosome is done.
    chroms=${args[*]:-$(seq -s ' ' 1 22)}
    for c in $chroms; do
      [[ $c =~ ^([1-9]|1[0-9]|2[0-2])$ ]] || die "e7-prep: chromosome '$c' is not one of 1..22"
    done
    setup=$(run --job-name=psmcrs-e7-setup ${build_job:+"--dependency=afterok:$build_job"} \
        --cpus-per-task=2 --mem=8G --time=03:00:00 --output="$LOGS/%x-%j.out" \
        "$job_env,E7_STAGE=setup" "$DIR/e7_prep.sbatch")
    chrom=$(run --job-name=psmcrs-e7-chrom --dependency=afterok:"$setup" \
        --array="${chroms// /,}%${E7_PREP_PARALLEL:-4}" --constraint="$PROD_CONSTRAINT" \
        --cpus-per-task="${E7_PREP_CPUS:-16}" --mem="${E7_PREP_MEM:-64G}" --time="${E7_PREP_TIME:-12:00:00}" \
        --output="$LOGS/%x-%A_%a.out" "$job_env,E7_STAGE=chrom" "$DIR/e7_prep.sbatch")
    samples=$(run --job-name=psmcrs-e7-samples --dependency=afterok:"$chrom" --cpus-per-task=16 --mem=32G \
        --time=04:00:00 --output="$LOGS/%x-%j.out" "$job_env,E7_STAGE=samples" "$DIR/e7_prep.sbatch")
    id="$setup $chrom $samples" ;;
esac
echo "submitted $what: $id${build_job:+ (after build $build_job)}"
