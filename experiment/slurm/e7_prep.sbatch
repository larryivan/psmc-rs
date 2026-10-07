#!/bin/bash
# E7's data stage (experiment/pipeline/kg1000.py) into $PROD/experiment/runs/E7/inputs.
# Submit with experiment/slurm/submit.sh e7-prep, which chains three jobs:
#
#   E7_STAGE=setup     the sources, samples, mask and chunk plan
#   E7_STAGE=chrom     one chromosome per array task (chr<task id>): its VCF
#                      downloaded to node-local disk in byte ranges and checked
#                      against the manifest (or read from $E7_LOCAL_SOURCE),
#                      every chunk, every sample's psmcfa; the copy deleted.
#                      chr21's task then runs the pilot.
#   E7_STAGE=samples   every sample's psmcfa of chr1-22; MANIFEST.json
#
# Everything temporary (the downloaded VCF, the chunk BCFs, the per-sample
# split files: up to ~140 GB for chr1 of the joint VCFs) is on node-local
# disk, $E7_LOCAL_TMP, and removed when the task ends. site.env may set
# E7_INPUT (default joint), E7_LOCAL_SOURCE and E7_LOCAL_TMP (default /tmp).
source "$PSMC_SLURM_DIR/env.sh" || exit 1
set -euo pipefail
cd "$PROD"
kg=("$PY" -m experiment.pipeline.kg1000)
common=(--profile production --runs-dir "$PROD/experiment/runs" --input "${E7_INPUT:-joint}")
[ -n "${E7_LOCAL_SOURCE:-}" ] && common+=(--local-source "$E7_LOCAL_SOURCE")
cores=${SLURM_CPUS_PER_TASK:?need sbatch --cpus-per-task}
echo "$(hostname) job $SLURM_JOB_ID commit $(git rev-parse --short HEAD): E7 ${E7_STAGE:?} ${SLURM_ARRAY_TASK_ID:-}"
psmc_env_report
echo "env: $(bcftools --version | head -1)"

case $E7_STAGE in
  setup)
    "${kg[@]}" check "${common[@]}"
    exec "${kg[@]}" setup "${common[@]}" ;;
  chrom)
    chrom=chr${SLURM_ARRAY_TASK_ID:?an array task: the chromosome number}
    tmp=${E7_LOCAL_TMP:-/tmp}/psmc-e7-$USER-$SLURM_JOB_ID-$chrom
    mkdir -p "$tmp"
    trap 'rm -rf "$tmp"' EXIT
    dirs=(--tmp-dir "$tmp" --local-tmp-dir "$tmp")
    keep=()
    [ "$chrom" = chr21 ] && keep=(--keep-source)  # the pilot reads the copy again
    "${kg[@]}" chrom "$chrom" "${common[@]}" "${dirs[@]}" --jobs "$cores" --chunk-jobs 4 \
      ${keep[@]+"${keep[@]}"}
    if [ "$chrom" = chr21 ]; then
      "${kg[@]}" pilot "$chrom" "${common[@]}" "${dirs[@]}"
    fi ;;
  samples)
    exec "${kg[@]}" assemble-samples "${common[@]}" --jobs "$cores" ;;
  *)
    echo "E7_STAGE must be setup, chrom or samples, not '$E7_STAGE'" >&2; exit 2 ;;
esac
