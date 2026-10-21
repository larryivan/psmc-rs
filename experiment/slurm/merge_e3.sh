#!/bin/bash
# Merge E3's repeats ($E3_SHARDS/rep*, one runs directory per node) into the
# production runs directory, so that E3/report reads every run.
#
# It refuses while a repeat's job or runner lives, when a repeat's directory
# is missing, when one repeat was timed on more than one host, and when the
# shards' simulations differ: every shard simulated the same genomes from the
# same seeds, so they must be byte-identical. Fits and timing records are per
# repeat and do not overlap; every shard also records the same simulation jobs
# (provenance/sim/), of which the last one copied is kept. Logs are kept per
# shard under E3/logs/shards/ (they are not committed). Afterwards every
# simulation and timing run must be done for the current code and tools.
set -euo pipefail
fail() { echo "merge_e3: $*" >&2; exit 1; }
DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
[ -f "$DIR/site.env" ] || fail "$DIR/site.env is missing: copy site.env.example there and fill it in"
# shellcheck source=/dev/null
source "$DIR/site.env"
PY=$MAIN/experiment/.venv/bin/python
dest=$PROD/experiment/runs/E3
repeats=$(cd "$PROD" && "$PY" -c 'from experiment.pipeline import config; print(config.get_profile("production").e3.repeats)')

queue=$(squeue -h -u "$USER" -o '%j %i %T')
if grep -q '^psmcrs-e3-rep' <<< "$queue"; then
  fail "E3 jobs are still queued or running: $(grep '^psmcrs-e3-rep' <<< "$queue" | tr '\n' ' ')"
fi
for r in $(seq 1 "$repeats"); do
  s=$E3_SHARDS/rep$r
  [ -d "$s/E3" ] || fail "repeat $r has no runs ($s/E3 is missing)"
  [ ! -e "$s/_runner/runner.pid" ] || fail "a runner of repeat $r still lives or died without cleaning up ($s/_runner/runner.pid)"
  hosts=$(cd "$PROD" && "$PY" - "$s/E3/timings" <<'EOF'
import json, pathlib, sys
print(" ".join(sorted({json.loads(p.read_text()).get("machine", {}).get("hostname", "?")
                       for p in pathlib.Path(sys.argv[1]).rglob("*.json")})))
EOF
)
  [ "$(wc -w <<< "$hosts")" -le 1 ] || fail "repeat $r was timed on several hosts: $hosts"
done

# Identical simulations in every shard (compared with the first).
ref=$E3_SHARDS/rep1/E3
for r in $(seq 2 "$repeats"); do
  for d in sims truth; do
    diff -rq "$ref/$d" "$E3_SHARDS/rep$r/E3/$d" > /dev/null \
      || fail "$E3_SHARDS/rep$r/E3/$d differs from $ref/$d: $(diff -rq "$ref/$d" "$E3_SHARDS/rep$r/E3/$d" | head -3)"
  done
done

mkdir -p "$dest"
for r in $(seq 1 "$repeats"); do
  s=$E3_SHARDS/rep$r
  rsync -a --exclude 'logs/' "$s/E3/" "$dest/"
  mkdir -p "$dest/logs/shards/rep$r"
  rsync -a "$s/E3/logs/" "$dest/logs/shards/rep$r/"
  echo "merged repeat $r: $(find "$s/E3/timings" -name '*.json' | wc -l) timing records"
done

# Every simulation and timing run done for the current code and tools?
status=$(cd "$PROD" && PSMC_SLURM_DIR=$DIR bash -c 'source "$PSMC_SLURM_DIR/env.sh" && "$PY" -m experiment.pipeline status \
  --profile production --runs-dir "$PROD/experiment/runs" --only "^E3/(sim|run)/" --jobs E3') ||
  fail "cannot tell whether every run is done: the pipeline's status failed (above)"
not_done=$(grep -E '^  (pending|running|failed|blocked) ' <<< "$status" || true)
if [ -n "$not_done" ]; then
  echo "$not_done" | head -20 >&2
  fail "$(wc -l <<< "$not_done") simulations or runs are not done for the current code and tools"
fi
echo "every repeat merged and done; now: submit.sh e3-report"
