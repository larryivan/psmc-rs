#!/bin/bash
# The diagnostic build: psmc-rs's sources at COMMIT (whose src/ tree must be dcf29ca, the production's) with
# diag.patch, built like the production (release profile, no RUSTFLAGS) in WORKDIR, never in this checkout.
# Usage: scripts/build.sh COMMIT WORKDIR   (prints the binary's path; JOBS sets cargo's -j, default 4)
set -euo pipefail
C=$(cd "$(dirname "$0")/.." && pwd)
REPO=$(git -C "$C" rev-parse --show-toplevel)
commit=$1; work=$2
tree=$(git -C "$REPO" rev-parse "$commit:src")
[ "$tree" = dcf29ca23bcbb0b6ca54cd24ea1a2efeb7db0910 ] || { echo "$commit: src/ is $tree, not dcf29ca" >&2; exit 1; }
mkdir -p "$work/src"
git -C "$REPO" archive "$commit" src examples tests Cargo.toml Cargo.lock | tar -x -C "$work/src"
(cd "$work/src" && patch -p1 --quiet < "$C/diag.patch")
(cd "$work/src" && env -u RUSTFLAGS -u CARGO_ENCODED_RUSTFLAGS CARGO_TARGET_DIR="$work/target" \
    cargo build --release --locked -j "${JOBS:-4}" >&2)
echo "$work/target/release/psmc-rs"
