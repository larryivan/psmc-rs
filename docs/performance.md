# Performance and determinism

- [Determinism](#determinism)
- [Threads](#threads)
- [Memory](#memory)
- [The E-step](#the-e-step)
- [The M-step](#the-m-step)
- [The CPU kernels](#the-cpu-kernels)
- [Measured on a laptop](#measured-on-a-laptop)

Use a release build, and `-q --no-progress` for batch runs. The
manuscript's controlled timings, on nodes of a cluster, are in
[validation.md](validation.md#speed-and-memory).

## Determinism

The results do not depend on `--threads` or `--batch-size`: a run writes
the same bytes (JSON, `.psmc`, TMRCA and bootstrap files) with any thread
count, which the tests check by running the binary. Partial sums are
combined in a fixed order, whatever the threads.

Gzip outputs are compressed by zlib-rs, whose version `Cargo.lock` pins:
another version of psmc-rs may compress the same text to other bytes, so
compare the `.gz` outputs of two versions decompressed (`gzip -dc`).

Across CPUs, the results are identical to the bit where the same kernels
run with the same `log` and `exp` from the system's math library
([the CPU kernels](#the-cpu-kernels)); elsewhere they differ in the last
digits ([validation.md](validation.md#across-platforms)).

## Threads

Sequences are processed in parallel, each on one thread, so `--threads`
(default one per core) helps inputs of many sequences. A single long
sequence runs on one thread, and t threads are up to t times as fast only
on at least t sequences of similar length. The threads left without a
sequence add no CPU time. The bootstrap replicates and the TMRCA decoding
run in parallel too.

## Memory

Instead of whole forward and backward matrices, the E-step keeps one
forward vector every 128 bins and recomputes each 128-bin segment for the
backward pass. Its memory still grows with the sequence length, but at
1/128 of the rate: about bins / 128 × states × 8 bytes per sequence in
flight, plus about 0.14 MB per worker thread (0.23 MB with callable counts
in bins of 100). With pattern `4+25*2+4+6` (64 states), one iteration on
one thread peaked at 19 MB on Papuan chr1 (2.49 M bins) and at 106 MB on a
single sequence of eight copies of it (19.9 M bins).

Models of more than 256 states use dense kernels, which cache forward rows
within a process-wide budget of `PSMC_ALPHA_CACHE_MB` (2048; 0 disables the
cache).

## The E-step

With models of up to 256 states, the E-step's forward and backward steps
take O(states) time per bin instead of O(states²): below and above its
diagonal, PSMC's transition matrix is a factor of the row times a factor of
the column, the rank-one triangles with which Harris et al. (2014) decode
coalescent HMMs in linear time.

With models of any size, so do the expected transition counts. C psmc's EM
Q, and the M-step with it, needs the counts only summed over each
triangle's rows and columns and over the diagonal, 5 × states sums, which
the E-step adds up from each bin's prefix and suffix sums instead of the
whole states × states matrix of counts. This is also why psmc-rs writes no
`QD` line in its `.psmc` output ([outputs.md](outputs.md#psmc-for-psmc_plotpl)).

## The M-step

The M-step needs little memory and time, however many λ are free. It
differentiates its objective in reverse mode, one row of the transition
matrix at a time, from the sums of the expected transition counts: an
evaluation holds O(states) values and costs O(states) time, 7.6 µs with 64
states on one thread, about what the E-step spends on 24 bins.

An M-step evaluates its objective once per free parameter for the
curvatures that scale its search, once or twice per L-BFGS iteration, and
once per free parameter again for the check before it stops, so its time
grows with the number of free parameters times the states. Measured on one
thread of an Apple M1 Pro, three runs of 25 EM iterations each:

- with pattern `4+25*2+4+6` (31 parameters) on a simulated 100 Mb
  chromosome (E3's), an M-step took 11 to 12 iterations, 74 to 102
  evaluations and 0.55 to 0.84 ms, against E-steps of 0.31 s (0.2%);
- with one λ per interval (`--n-steps` alone) on the 180,000-bin test
  input, M-steps over 104 parameters took 2.6 to 5.1 ms and over 304 took
  22 to 48 ms, against E-steps of 0.09 s and 4.1 s.

One EM iteration there on one thread peaked at 5.8, 7.5, 325 and 437 MB of
resident memory with 101, 201, 301 and 401 intervals and took 0.13, 0.24,
6.1 and 11.0 s (medians of three runs). Above 256 intervals the memory is
the E-step's cache of forward rows (`PSMC_ALPHA_CACHE_MB`): without it, the
iteration with 401 intervals peaked at 13 MB.

## The CPU kernels

On x86-64 the E-step, the log-likelihood and the TMRCA decoding run code
compiled with AVX2 and FMA on CPUs that have them (Intel Core since
Haswell, AMD since Excavator and Zen), picked at run time, so the release
binaries need no `-C target-cpu` flags for full speed. They fitted a 1.9 Gb
genome (HLpelCas1: 1,220 sequences, 19.4 M bins; pattern `4+25*2+4+6`, 5 EM
iterations, 1 thread) in 43.4 s on an AMD EPYC 7702 alone on its node, as
fast as a `-C target-cpu=x86-64-v3` build (44.1 s), against 79.2 s for the
portable code (medians of three runs). On aarch64 (Apple silicon and
others) the kernels use NEON.

The AVX2 and FMA code fuses the multiply-adds of the O(states) forward and
backward products and of the transition sums, as the NEON code does; the
other x86-64 kernels never fuse, and no sum changes its order with the
vector width. Its results are therefore identical to the bit to an
`x86-64-v3` build's, and on every CPU with AVX2 and FMA that gets the same
`log` and `exp` from the system's math library (glibc picks a variant of
them by CPU): the production's binary refitted a simulated genome byte for
byte on AMD EPYC 7702 and 9655 and Intel Xeon Gold 6148 and 6248
(`experiment/crossplatform/x86_cpus/`).

x86-64 CPUs without AVX2 or FMA, and virtual machines and emulators that
hide them, run the portable code, which does not fuse those multiply-adds
(each would be a call to libm's `fma`). Its results differ in the last
digits: by a relative 1e-14 or so in an E-step (2.2e-14 at most on the test
inputs), which the M-step, stopping on a flat optimum, turns into about
1e-8 to 1e-6 in the fitted parameters (on an EPYC 7702, 7.8e-9 on a
60,000-bin test input, three records of 20,000 bins, after 3 EM
iterations, and 8.6e-7 on the 180,000-bin one after 5 with the pattern
`4+25*2+4+6`).

`PSMC_RS_SIMD=portable` forces the portable code, to compare speeds or
results; the output JSON's `simd` and the report say which kernels ran
(`avx2+fma`, `portable` or `neon`).

## Measured on a laptop

On the inputs in `experiment/real_data` with pattern `4+25*2+4+6`, on an
Apple M1 Pro (8 cores, 16 GB, macOS 27, Rust 1.97.1, release builds)
shared with other work; wall time and peak resident memory from
`/usr/bin/time -l`:

| input | EM iterations | threads | psmc-rs | C psmc 0.6.5 (`-O1`) |
| --- | --- | --- | --- | --- |
| Papuan chr1: 1 sequence, 2.49 M bins | 20 | 1 | 15.4 s, 19 MB | 322 s, 2.63 GB |
| | 20 | 8 | 15.8 s, 20 MB | |
| HLhydTec1: 34 sequences, 18.5 M bins | 5 | 1 | 30.3 s, 50 MB | 595 s, 3.41 GB |
| | 5 | 8 | 6.5 s, 98 MB | |

Medians of three runs, interleaved in one session: psmc-rs at load averages
of 2.1 to 4.3, C psmc at 2.0 to 10.5 (its Papuan runs followed an 8-thread
fit, which the 1-minute average still counted). C psmc's 3.41 GB on
HLhydTec1 is its peak memory footprint, which `/usr/bin/time -l` also
gives: the other work's memory pressure kept its resident set below that.
C psmc is single-threaded. The JSON outputs at 1 and 8 threads were
byte-identical.
