#!/bin/bash
# E1 bottleneck rep01 refitted by the production binary on another CPU model; bytes compared with the committed fit.
set -euo pipefail
P=/ibex/user/hany0c/psmc-rs-prod; B=$P/target/release/psmc-rs; R=$P/experiment/runs; T=$(mktemp -d)
echo "host $(hostname); $(lscpu | grep -m1 "Model name"); glibc $(ldd --version | head -1 | awk "{print \$NF}")"
echo "binary $(sha256sum $B | cut -c1-16)"
grep -o -m1 -w -E "avx2|fma" /proc/cpuinfo | sort -u | tr "\n" " "; echo
/usr/bin/time -f "wall %e s, maxrss %M KB" $B $R/E1/sims/bottleneck/rep01/genome.psmcfa $T/rep01.rs.json 25 --pattern "4+25*2+4+6" --t-max 15 --tr-ratio 5 --smooth-lambda 0 --mstep-iters 1000 --bin-size 100 --mu 2.5e-08 --gen-years 25 --threads 1 --no-progress --psmc-out $T/rep01.rs.psmc > $T/out.txt 2>&1 || { cat $T/out.txt; exit 1; }
tail -1 $T/out.txt
grep -o "\"simd\": *\"[^\"]*\"" $T/rep01.rs.json || true
for f in rep01.rs.json:5efb49006e98f131cd719486970532ca83823ed77dc4e9af7ac980cb68c419b2 rep01.rs.psmc:70d315ecc225d3cb8059f7a228ae294930ee8f8e7b38826cf54a92c594c64eea; do
  n=${f%%:*}; want=${f#*:}; got=$(sha256sum $T/$n | cut -d" " -f1)
  [ "$got" = "$want" ] && echo "$n IDENTICAL" || echo "$n DIFFERS ($got)"
done
rm -rf $T
