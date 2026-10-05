#!/usr/bin/env bash
# Checks the binary of a Linux release archive before it is published (see
# release.yml):
#
#   .github/scripts/check-linux-binary.sh ARCHIVE GLIBC
#
# - it needs no glibc symbol version newer than GLIBC, the oldest glibc it
#   was linked for;
# - in containers of older distributions, `psmc-rs --version` prints what it
#   prints here and nothing on standard error, and a short fit runs. CentOS 7
#   has glibc 2.17; Rocky Linux 8 (glibc 2.28) stands for the RHEL 8 family
#   of many clusters; on Ubuntu 22.04 (glibc 2.35) a binary linked against a
#   newer glibc prints "weak version `GLIBC_2.39' not found" on every run.
#
# Run it from the repository root (the fit reads tests/data); it needs
# docker and objdump. IMAGES="IMAGE..." replaces the list of images.

set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "usage: $0 ARCHIVE GLIBC" >&2
  exit 2
fi
archive=$1 floor=$2
read -r -a images <<< "${IMAGES:-centos:7 rockylinux:8 ubuntu:22.04}"
data=$PWD/tests/data/c_reference
input=papuan_chr1_3seg.psmcfa

fail() {
  printf 'FAIL: %s\n' "$*" >&2
  exit 1
}

# Whether version $1 (x.y or x.y.z) is newer than $2.
newer() {
  awk -v a="$1" -v b="$2" 'BEGIN {
    n = split(a, x, "."); m = split(b, y, ".")
    for (i = 1; i <= (n > m ? n : m); i++)
      if (x[i] + 0 != y[i] + 0) exit (x[i] + 0 > y[i] + 0 ? 0 : 1)
    exit 1
  }'
}

[ -f "$data/$input" ] || fail "no $data/$input: run this from the repository root"
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
mkdir "$work/bin"
tar -xzf "$archive" -C "$work/bin"
[ "$(find "$work/bin" -type f -name psmc-rs | wc -l)" -eq 1 ] ||
  fail "$archive does not hold one psmc-rs"
bin=$(find "$work/bin" -type f -name psmc-rs)
bin_dir=$(dirname "$bin")

# The glibc symbol versions the binary needs, from its version references.
needs=$(objdump -p "$bin" |
  awk '/^Version References/ { on = 1; next } on && $4 ~ /^GLIBC_/ { print $4 }' | sort -u)
[ -n "$needs" ] || fail "no glibc version reference in $bin (objdump -p)"
too_new="" newest=0
for need in $needs; do
  v=${need#GLIBC_}
  case "$v" in
    '' | *[!0-9.]*) too_new="$too_new $need" ;; # GLIBC_ABI_DT_RELR, GLIBC_PRIVATE
    *)
      if newer "$v" "$floor"; then too_new="$too_new $need"; fi
      if newer "$v" "$newest"; then newest=$v; fi
      ;;
  esac
done
[ -z "$too_new" ] || fail "the binary needs glibc newer than $floor:$too_new"
printf 'ok: the newest glibc symbol version needed is GLIBC_%s (floor %s)\n' "$newest" "$floor"

version=$(awk -F '"' '/^version = "/ { print $2; exit }' Cargo.toml)
[ -n "$version" ] || fail "no version in Cargo.toml"
expected="psmc-rs $version"

for image in "${images[@]}"; do
  res=$work/$(printf '%s' "$image" | tr -c 'A-Za-z0-9._-' '_')
  mkdir "$res"
  s=0
  docker run --rm --platform linux/amd64 --user "$(id -u):$(id -g)" \
    -v "$bin_dir:/psmc:ro" -v "$data:/data:ro" -v "$res:/res" "$image" sh -c '
      ldd --version 2>&1 | head -n 1 > /res/glibc
      /psmc/psmc-rs --version > /res/version.out 2> /res/version.err || exit 3
      /psmc/psmc-rs --threads 2 "/data/$1" /res/fit.json 2 > /res/fit.out 2> /res/fit.err || exit 4
    ' sh "$input" || s=$?
  glibc=$(cat "$res/glibc" 2> /dev/null || true)
  case "$s" in
    0) ;;
    3) fail "$image ($glibc): psmc-rs --version failed: $(cat "$res/version.err")" ;;
    4) fail "$image ($glibc): the fit failed: $(tail -n 5 "$res/fit.err")" ;;
    *) fail "$image: docker run exited with $s" ;;
  esac
  [ "$(cat "$res/version.out")" = "$expected" ] ||
    fail "$image ($glibc): psmc-rs --version printed '$(cat "$res/version.out")', not '$expected'"
  [ ! -s "$res/version.err" ] ||
    fail "$image ($glibc): psmc-rs --version wrote to standard error: $(cat "$res/version.err")"
  [ -s "$res/fit.json" ] || fail "$image ($glibc): the fit wrote no $res/fit.json"
  printf 'ok: %s (%s): %s, nothing on standard error, a 2-iteration fit\n' \
    "$image" "$glibc" "$expected"
done
