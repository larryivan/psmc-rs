# The environment of every job of experiment/slurm (sourced, not run).
#
# Every job gets the same environment, whatever shell submitted it (sbatch
# --export=ALL passes that shell's): the pipeline identifies psmc-rs by rustc
# and its build flags, so a job with another compiler or other flags would
# rebuild it, re-key every psmc-rs job and halt the runners using the old one.

_slurm_dir=${PSMC_SLURM_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}
if [ ! -f "$_slurm_dir/site.env" ]; then
  echo "env.sh: $_slurm_dir/site.env is missing (copy site.env.example)" >&2
  return 1 2>/dev/null || exit 1
fi

# Forget the submitting shell's modules, build settings and pipeline settings
# (CARGO_HOME and RUSTUP_HOME stay: they say where rustup and cargo live).
if type module > /dev/null 2>&1 || source /etc/profile.d/modules.sh 2> /dev/null; then
  module purge > /dev/null 2>&1 || true
fi
for _v in $(compgen -e | grep -E '^(PSMC_|CARGO_|RUSTC|RUSTFLAGS$|RUSTDOCFLAGS$|RUSTUP_TOOLCHAIN$|PYTHON|CONDA_|MAMBA_|VIRTUAL_ENV$|CC$|CXX$|CFLAGS$|CXXFLAGS$|CPPFLAGS$|LDFLAGS$|LD_LIBRARY_PATH$|LIBRARY_PATH$|CPATH$|C_INCLUDE_PATH$|PKG_CONFIG_PATH$|OMP_|OPENBLAS_|MKL_|NUMEXPR_|RAYON_)' \
    | grep -vE '^(CARGO_HOME|RUSTUP_HOME)$'); do
  unset "$_v"
done

# shellcheck source=/dev/null
source "$_slurm_dir/site.env"
export PSMC_SLURM_DIR=$_slurm_dir
unset _slurm_dir _v

# Rust: the pinned toolchain through rustup's proxies, the release profile of
# Cargo.toml and no flags (the release binary picks its SIMD kernels at run
# time), offline (the login node fetched the crates: cargo fetch --locked).
export RUSTUP_TOOLCHAIN=$RUST_TOOLCHAIN CARGO_NET_OFFLINE=true
# Python 3.13 with experiment/pipeline/requirements.lock, and htslib's bgzip,
# tabix and bcftools 1.24 from the same environment, ahead of anything else.
export PATH=$MAIN/experiment/.venv/bin:${CARGO_HOME:-$HOME/.cargo}/bin:$PATH
export PY=$MAIN/experiment/.venv/bin/python
export PSMC_EXP_TOOLS=$MAIN/experiment/.tools
# C psmc's build flags (tools --build-c), if the site sets them.
[ -n "${PSMC_C_CFLAGS:-}" ] && export PSMC_C_CFLAGS

# The runner's machine lock in this job's /tmp: runners of different jobs never
# wait for each other through the shared home (SLURM keeps jobs apart).
export PSMC_EXP_MACHINE_LOCK=/tmp/psmc-exp-$USER-${SLURM_JOB_ID:-login}/machine.lock
mkdir -p "$(dirname "$PSMC_EXP_MACHINE_LOCK")"

# The runner sets each job's thread variables to its cores; nothing else threads.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 MPLBACKEND=Agg

# What this job will use, for its log.
psmc_env_report() {
  echo "env: $(rustc -V) via $(command -v rustc); $(gcc --version | head -1); $("$PY" -V) at $PY; $(bgzip --version | head -1)"
}
