#!/bin/bash
# One repeat of E3's timing runs, alone on an exclusive node, in its own runs
# directory ($E3_SHARDS/rep<R>). Each repeat's speed-ups are paired on one
# node (submit.sh pins a resubmitted repeat to its first node); every repeat
# runs on the same CPU model ($E3_CONSTRAINT). Submit with
# experiment/slurm/submit.sh e3 1 2 3 4 5, then merge with merge_e3.sh.
#SBATCH --signal=B:USR1@7200
source "$PSMC_SLURM_DIR/env.sh" || exit 1
set -euo pipefail
cd "$PROD"
r=${E3_REPEAT:?E3_REPEAT is not set}
runs=$E3_SHARDS/rep$r
mkdir -p "$runs"
echo "$(hostname) job $SLURM_JOB_ID commit $(git rev-parse --short HEAD): E3 repeat $r in $runs"
lscpu | sed -n 's/^Model name: *//p'
psmc_env_report

# Is the idle node quiet enough for the timing protocol? A node whose own
# daemons keep more cores busy than the protocol allows would defer every run
# until the time limit: stop at once instead, and say what kept it busy.
"$PY" - <<'EOF'
import sys

from experiment.pipeline import config
from experiment.pipeline.experiments.e3_run import machine_activity

p = config.get_profile("production").e3.timing
act = machine_activity(30)
print(f"idle node: {act['busy_cores']:.2f} cores busy ({act['busy_percent']:.2f}% of all); the protocol allows "
      f"{p.quiet_max_cores} before a run and {p.max_other_cores} of other processes during one")
print(f"  of them: user-space processes {act['process_cores']}, kernel threads {act['kernel_thread_cores']}, "
      f"interrupts {act['interrupt_cores']} cores")
for b in act["busiest"]:
    print(f"  {b['name']} (pid {b['pid']}): {b['cpu_percent']}% of a core")
if act["busy_cores"] > p.quiet_max_cores:
    print("the idle node is busier than the protocol allows: no run started", file=sys.stderr)
    raise SystemExit(3)
EOF

# The simulations (identical in every shard) and this repeat's runs. E3's runs
# are exclusive: the runner starts each alone, whatever --cores says.
exec "$PY" -m experiment.pipeline start --foreground --no-keep-awake --profile production \
  --runs-dir "$runs" --cores "${SLURM_CPUS_ON_NODE:?}" --mem-gb 64 --only "^E3/(sim/|run/.*/rep$r\$)" E3
