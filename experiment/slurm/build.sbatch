#!/bin/bash
# Build psmc-rs 0.2 and C psmc on a compute node, from the production
# worktree, and run every Rust and Python test there.
# Submit with experiment/slurm/submit.sh build; it ends with "all passed".
source "$PSMC_SLURM_DIR/env.sh" || exit 1
set -uo pipefail
cd "$PROD" || exit 1
export CARGO_BUILD_JOBS=${SLURM_CPUS_PER_TASK:-4}
fail=0
step() { echo; echo "== $1 ($(date +%T))"; }
check() { local rc=$?; [ $rc -eq 0 ] && echo "ok" || { echo "FAILED (exit $rc)"; fail=1; }; }

echo "== $(hostname), job ${SLURM_JOB_ID:-}, $(date -Is)"
lscpu | sed -n 's/^Model name: *//p'
psmc_env_report
echo "commit $(git rev-parse HEAD)"
step "the production worktree is clean"
[ -z "$(git status --porcelain --untracked-files=no)" ]; check

step "psmc-rs 0.2: cargo build --release"
cargo build --release --locked --offline; check
target/release/psmc-rs --version

step "cargo test (every test target)"
cargo test --locked --offline --no-fail-fast 2>&1 | tee "$LOGS/build-${SLURM_JOB_ID:-x}-cargo-test.log" \
  | grep -E '^test result:|FAILED|panicked'; [ "${PIPESTATUS[0]}" -eq 0 ]; check

step "C psmc with a recorded build (tools --build-c)"
if [ -f "$PSMC_EXP_TOOLS/psmc_c/BUILD.json" ]; then
  echo "BUILD.json exists: keeping that build (tools below checks that it describes psmc_c/psmc)"
else
  "$PY" -m experiment.pipeline tools --build-c --force; check
fi

step "every tool's identity (tools)"
"$PY" -m experiment.pipeline tools; check   # exits 1 if a tool is missing

step "Python tests of the pipeline"
"$PY" -m unittest discover -s experiment/tests 2>&1 | tail -4; [ "${PIPESTATUS[0]}" -eq 0 ]; check

echo; echo "== $( [ $fail -eq 0 ] && echo 'all passed' || echo 'SOMETHING FAILED' ) $(date -Is)"
exit $fail
