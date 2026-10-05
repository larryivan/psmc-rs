"""Every parameter of the manuscript's experiments, in one place.

No other module of the pipeline holds a number that shapes a result: the
experiments read their settings from here, and every job records the
settings it ran with in its provenance record. The file is organised as the
experimental design is:

1. inference settings shared by every PSMC fit (both tools);
2. scaling to years and individuals;
3. simulation settings and the demographic models;
4. the seed scheme;
5. the experiments E1-E6;
6. the runner's resource budget and cost model (scheduling only);
7. the profiles: ``production`` (the manuscript), ``smoke`` (a small
   end-to-end check of the same code paths) and ``tiny`` (unit tests);
8. E2's settings in the smoke and tiny profiles.

Times of demographic models are in generations and sizes are diploid
individuals, as in msprime, unless a comment says otherwise.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from typing import Dict, Optional, Tuple

# ---------------------------------------------------------------------------
# 1. Inference: every PSMC fit, C psmc and psmc-rs alike
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Inference:
    """The settings of one PSMC fit, passed explicitly to both tools.

    The defaults are the canonical human recipe of the psmc README,
    ``psmc -N25 -t15 -r5 -p "4+25*2+4+6"``, which psmc-rs takes as
    ``N_ITER 25 --t-max 15 --tr-ratio 5 --pattern 4+25*2+4+6``, plus
    ``--smooth-lambda 0`` (C psmc's objective has no penalty).
    """

    n_iter: int = 25  # EM iterations: C -N, psmc-rs N_ITER
    t_max: float = 15.0  # initial maximum time, in 2*N0 generations: C -t, --t-max
    tr_ratio: float = 5.0  # initial theta0/rho0: C -r, --tr-ratio
    pattern: str = "4+25*2+4+6"  # free-parameter pattern (64 intervals, 28 free lambdas): C -p, --pattern
    smooth_lambda: float = 0.0  # psmc-rs --smooth-lambda; 0 is C psmc's objective
    mstep_iters: int = 100  # psmc-rs --mstep-iters (its default; C's Hooke-Jeeves runs to convergence)
    bin_size: int = 100  # bases per bin: fq2psmcfa -s, psmc-rs --bin-size


INFERENCE = Inference()

# ---------------------------------------------------------------------------
# 2. Scaling to years and individuals (plots and metrics only; the fits do
#    not depend on it). N0 = theta / (4 * mu * bin_size), years = 2*N0*t*g.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Scaling:
    mu: float  # mutation rate per bp per generation
    gen_years: float  # years per generation


# Simulations are scaled with the rate they were simulated with.
SIMULATION_SCALING = Scaling(mu=2.5e-8, gen_years=25.0)

# ---------------------------------------------------------------------------
# 3. Simulation: direct msprime (Hudson coalescent, JC69 mutations, discrete
#    genome), one diploid individual (two haplotypes).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Genome:
    """The simulated genome: independent chromosomes of equal length."""

    n_chromosomes: int = 10
    chromosome_length: int = 50_000_000  # bp

    @property
    def length(self) -> int:
        return self.n_chromosomes * self.chromosome_length

    def names(self) -> Tuple[str, ...]:
        return tuple(f"chr{i}" for i in range(1, self.n_chromosomes + 1))


@dataclass(frozen=True)
class Simulation:
    """How one diploid genome is simulated (see :mod:`.simulate`).

    Chromosomes are simulated independently (unlinked), each with its own
    ancestry and mutation seed. theta/rho = 2.5 per bp.
    """

    mutation_rate: float = 2.5e-8  # per bp per generation
    recombination_rate: float = 1.0e-8  # per bp per generation
    ancestry_model: str = "hudson"  # msprime.sim_ancestry(model=...)
    mutation_model: str = "jc69"  # msprime.JC69()
    discrete_genome: bool = True
    ploidy: int = 2  # one diploid individual: two haplotypes
    bin_size: int = INFERENCE.bin_size  # psmcfa bins: K if the bin holds a heterozygous site, else T


SIMULATION = Simulation()
GENOME_500MB = Genome(n_chromosomes=10, chromosome_length=50_000_000)


@dataclass(frozen=True)
class SizeChange:
    """A change of the population size, as ``msprime.Demography.add_population_parameters_change``.

    From ``time`` (generations ago) back, the size is ``size`` (diploid
    individuals; ``None`` keeps the size reached at ``time``) and changes
    with ``growth_rate`` (per generation, msprime's convention: going back
    in time, N(t) = N(time) * exp(-growth_rate * (t - time))).
    """

    time: float
    size: Optional[float]
    growth_rate: float = 0.0


@dataclass(frozen=True)
class DemographicModel:
    """A single-population model: the present size and its changes back in time."""

    name: str
    initial_size: float
    changes: Tuple[SizeChange, ...]
    source: str
    # The ms definition, when the model is given in ms units: N0 and the -eN
    # events as (time in 4*N0 generations, size / N0).
    ms_n0: Optional[float] = None
    ms_events: Tuple[Tuple[float, float], ...] = ()


def ms_model(name: str, n0: float, events: Tuple[Tuple[float, float], ...], source: str) -> DemographicModel:
    """A piecewise-constant model from ms ``-eN`` events.

    ms measures time in units of 4*N0 generations and sizes as ratios to N0,
    so ``-eN t x`` is a change at ``t * 4 * N0`` generations to ``x * N0``
    individuals, with no growth.
    """
    changes = tuple(SizeChange(time=t * 4.0 * n0, size=x * n0, growth_rate=0.0) for t, x in events)
    return DemographicModel(name, float(n0), changes, source, ms_n0=float(n0), ms_events=tuple(events))


_MS_SOURCE = (
    "ms-style model of the psmc-rs 0.1 manuscript pipeline "
    "(experiment/scripts/simulate_msprime_to_psmcfa.py), converted exactly"
)

CONSTANT = ms_model("constant", 10_000, (), _MS_SOURCE)
BOTTLENECK = ms_model(
    "bottleneck", 20_000, ((0.01, 0.05), (0.015, 0.5), (0.05, 0.25), (0.5, 0.5)), _MS_SOURCE
)
EXPANSION = ms_model(
    "expansion", 10_000, ((0.01, 0.1), (0.06, 1.0), (0.2, 0.5), (1.0, 1.0), (2.0, 2.0)), _MS_SOURCE
)

# Zigzag_1S14 of the stdpopsim 0.3.0 HomSap catalog (Schiffels & Durbin 2014,
# Nat Genet 46:919, doi:10.1038/ng.3015): N0 = 5 * 14312, five epochs of
# exponential growth and decline, then a constant ancient size N0 / 10. The
# numbers are copied from stdpopsim's definition (catalog/HomSap/
# demographic_models.py, _zigzag); demography.check_zigzag_against_stdpopsim
# compares the resulting msprime.Demography with stdpopsim's own, epoch by
# epoch, with msprime's DemographyDebugger. stdpopsim's generation time (30)
# is not used: like every simulated model it is scaled with g = 25.
ZIGZAG = DemographicModel(
    name="zigzag",
    initial_size=71_560.0,
    changes=(
        SizeChange(time=33.333334976, size=None, growth_rate=0.02302578256),
        SizeChange(time=133.3334544, size=None, growth_rate=-0.0057564631),
        SizeChange(time=533.33324512, size=None, growth_rate=0.00143911578),
        SizeChange(time=2133.3352704, size=None, growth_rate=-0.0003597785),
        SizeChange(time=8533.329632, size=None, growth_rate=8.9944801565e-05),
        SizeChange(time=34133.318528, size=7_156.0, growth_rate=0.0),
    ),
    source="stdpopsim 0.3.0 HomSap/Zigzag_1S14 (Schiffels & Durbin 2014), converted to msprime.Demography",
)
ZIGZAG_STDPOPSIM = ("HomSap", "Zigzag_1S14")

MODELS: Dict[str, DemographicModel] = {m.name: m for m in (CONSTANT, BOTTLENECK, EXPANSION, ZIGZAG)}
MODEL_ORDER: Tuple[str, ...] = ("constant", "bottleneck", "expansion", "zigzag")

# ---------------------------------------------------------------------------
# 4. Seeds
# ---------------------------------------------------------------------------

SEED_NAMESPACE = "psmc-rs-manuscript"


def seed(experiment: str, model: str, replicate: int, stage: str) -> int:
    """The seed of one random stage of one replicate, in [1, 2**31 - 1].

    It is the first 8 bytes of SHA-256 over the UTF-8 string
    ``"psmc-rs-manuscript/<experiment>/<model>/<replicate>/<stage>"``, read
    as a big-endian integer, modulo 2**31 - 1, plus 1. ``stage`` names what
    the seed drives, with the chromosome where there is one:
    ``ancestry/chr3``, ``mutations/chr3``, ``mask/chr3``, ``bootstrap``.
    Replicates count from 1. A seed therefore depends only on its own key:
    adding a model, replicate or chromosome changes no other seed, and the
    seeds do not depend on the order in which jobs run. Every simulation
    records its seeds in its truth record, and every job in its provenance
    record.
    """
    if replicate < 0:
        raise ValueError(f"replicate must be >= 0, got {replicate}")
    key = "/".join((SEED_NAMESPACE, experiment, model, str(int(replicate)), stage))
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**31 - 1) + 1


# ---------------------------------------------------------------------------
# 5. Experiments
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TimeGrid:
    """The common time grid of the accuracy metrics (see :mod:`.metrics`).

    RMSE of log10 Ne is evaluated at ``points`` log-spaced times between the
    ``quantiles`` of the true pairwise coalescence time of the simulated
    model, in years. PSMC infers Ne(t) from the distribution of the two
    haplotypes' coalescence times along the genome; before the 1% quantile
    and after the 99% one, less than 1% of the genome carries information,
    so both tools extrapolate their first and last lambda there and an error
    measures the time discretisation, not the inference. The window depends
    on the model only: it is the same for both tools and every replicate.
    ``fixed_years`` is a second, model-independent window reported next to
    it, for robustness.
    """

    quantiles: Tuple[float, float] = (0.01, 0.99)
    points: int = 200
    fixed_years: Tuple[float, float] = (1e4, 1e6)


@dataclass(frozen=True)
class E1:
    """Simulated accuracy and equivalence (Fig 2, Table 1, Fig S1).

    For every replicate both tools fit the identical psmcfa. Metrics: RMSE of
    log10 Ne against the truth and between the tools on the time grid,
    log-likelihood by EM iteration, final theta, rho and lambdas, and wall
    time and peak RSS of every run (recorded only: timing claims come from E3).
    """

    models: Tuple[str, ...] = MODEL_ORDER
    replicates: int = 10
    genome: Genome = GENOME_500MB
    simulation: Simulation = SIMULATION
    inference: Inference = INFERENCE
    scaling: Scaling = SIMULATION_SCALING
    grid: TimeGrid = TimeGrid()
    rs_threads: int = 1  # psmc-rs --threads per fit; many fits run in parallel instead


@dataclass(frozen=True)
class RealGenome:
    """One empirical genome of E2 (a psmcfa in experiment/real_data)."""

    sample_id: str
    file: str
    species: str
    label: str  # how figures and tables name it
    scaling: Scaling
    scaling_source: str


_TURTLE_SCALING_SOURCE = (
    "UNVERIFIED: mu 7.9e-9 and g 15 are the values of the psmc-rs 0.1 runs "
    "(experiment/scripts/run_real_data.py, experiment/runs/real_data/tables/"
    "real_data_rust_vs_c_fit_summary.csv); the repository records no citation "
    "for them. Confirm and cite before publication."
)
_HUMAN_SCALING_SOURCE = (
    "mu 1.25e-8: the pedigree-based human rate (Scally & Durbin 2012, Nat Rev Genet "
    "13:745); g 25 as in Li & Durbin 2011. Same values as the psmc-rs 0.1 runs."
)

REAL_GENOMES: Tuple[RealGenome, ...] = (
    RealGenome(
        "HLemySub1", "HLemySub1_diploid.psmcfa", "Emydura subglobosa",
        "Emydura subglobosa (HLemySub1)", Scaling(7.9e-9, 15.0), _TURTLE_SCALING_SOURCE,
    ),
    RealGenome(
        "HLhydTec1", "HLhydTec1_diploid.psmcfa", "Hydromedusa tectifera",
        "Hydromedusa tectifera (HLhydTec1)", Scaling(7.9e-9, 15.0), _TURTLE_SCALING_SOURCE,
    ),
    RealGenome(
        "HLpelCas1", "HLpelCas1_diploid.psmcfa", "Pelusios castaneus",
        "Pelusios castaneus (HLpelCas1)", Scaling(7.9e-9, 15.0), _TURTLE_SCALING_SOURCE,
    ),
    # One sequence, >chr1, of 2,487,383 bins: chromosome 1 only, not a genome.
    RealGenome(
        "Papuan", "Papuan_highlands_diploid.psmcfa", "Homo sapiens",
        "Homo sapiens (Papuan Highlands, chr1 only)", Scaling(1.25e-8, 25.0), _HUMAN_SCALING_SOURCE,
    ),
)


@dataclass(frozen=True)
class E2:
    """Real genomes (Fig 3; QC table S5): both tools, psmc-rs bootstrap and qc on each.

    Every genome's psmcfa is fitted by C psmc and psmc-rs 0.2 with
    ``inference``; psmc-rs also fits ``bootstrap_replicates`` C-compatible
    bootstrap replicates (splitfa trunks, fresh starts) and ``psmc-rs qc``
    reports on the input. Curves are scaled with each genome's own
    ``scaling``. The jobs and tables are in experiments/e2.py.
    """

    genomes: Tuple[RealGenome, ...] = REAL_GENOMES
    inference: Inference = INFERENCE
    bootstrap_replicates: int = 100
    bootstrap_trunk_bins: int = 500_000  # psmc-rs default (splitfa's 50 Mb at 100 bp)
    # A second bootstrap, with trunks of the size E5 also calibrates (50,000
    # bins, 5 Mb), of every genome whose default trunks are fewer than
    # ``few_trunks``: Papuan chr1 forms only 5 trunks of 50 Mb, so its default
    # band rests on 5 resampling units. E5 calibrates bootstraps of 50 Mb
    # chromosomes cut into 50 Mb and 5 Mb trunks; the turtles' default
    # bootstraps resample hundreds of whole scaffolds of very different
    # lengths (most bins lie in scaffolds shorter than a trunk), a regime E5
    # does not calibrate. None: no second bootstrap.
    fine_trunk_bins: Optional[int] = 50_000
    band: Tuple[float, float] = (0.025, 0.975)
    rs_threads: int = 7  # the bootstrap fits replicates in parallel
    # Seeds: seed("E2", sample_id, 0, "bootstrap") for the default trunks,
    # seed("E2", sample_id, 0, "bootstrap/trunk<bins>") for the fine ones.

    # -- Settings of the E2 jobs and tables (experiments/e2.py) --------------
    fit_threads: int = 1  # psmc-rs --threads of the main fit (the output does not depend on it)
    # The window of the curve comparisons. A real genome has no true Ne(t),
    # so the informative window of TimeGrid comes from the fitted model: the
    # 1% and 99% quantiles of the pairwise coalescence time under C psmc's
    # final model (the reference implementation), in years of the genome's
    # own scaling. Both tools and the bootstrap band are compared on it;
    # ``grid.fixed_years`` is reported next to it.
    grid: TimeGrid = TimeGrid()
    display_years: Tuple[float, float, int] = (1e2, 1e8, 600)  # dense log grid of the band table (Fig 3)
    panel_order: Tuple[str, ...] = ("Papuan", "HLemySub1", "HLhydTec1", "HLpelCas1")  # Fig 3 panels A-D
    # Below this many trunks psmc-rs warns that the bands are coarse
    # (FEW_TRUNKS in src/bin/psmc/run.rs); the tables and Fig 3 say so. Papuan
    # chr1 (2.49 M bins) forms 5 trunks of the default 500,000 bins.
    few_trunks: int = 20
    # Smoke and tiny profiles only: fit the first ``max_bins`` bins of each
    # genome (records in file order, the last one cut) instead of the whole
    # file. None (production) fits the packaged files themselves.
    max_bins: Optional[int] = None


# --- E3: how a run is timed, and the memory and time model of each tool ----


@dataclass(frozen=True)
class TimingProtocol:
    """How ``experiments/e3_run.py measure`` times one run of a tool.

    Before the run it waits ``settle_seconds``, then until the CPU use of the
    whole machine over ``quiet_window_seconds`` is at most
    ``quiet_cpu_percent`` of all cores. During the run it samples the RSS of
    the tool's process tree every ``rss_sample_seconds`` (the tools are
    single processes, whose own ``ru_maxrss`` is the exact peak; the larger
    is kept) and afterwards records the CPU time that other processes used
    while it ran. A run is only kept when it was clean: the machine became
    quiet within ``max_wait_seconds``, other processes used at most
    ``max_other_cpu_share`` of all cores during it, the machine did not sleep
    (wall-clock span within ``max_sleep_seconds`` of the monotonic run time)
    and, with ``require_ac_power``, it ran on AC power without Low Power
    Mode. Otherwise ``measure`` writes no record and exits with status 75,
    and the runner starts the run again later (it defers it).
    """

    settle_seconds: float = 10.0
    quiet_cpu_percent: float = 10.0
    quiet_window_seconds: float = 2.0
    max_wait_seconds: float = 180.0
    rss_sample_seconds: float = 0.5
    max_other_cpu_share: float = 0.05  # of all cores, over the run
    max_sleep_seconds: float = 2.0
    require_ac_power: bool = True


# For the smoke and tiny profiles: no waiting, one short look at the machine,
# and every run kept whatever else the machine does.
QUICK_TIMING = TimingProtocol(settle_seconds=0.0, quiet_window_seconds=0.2, max_wait_seconds=0.0,
                              rss_sample_seconds=0.05, quiet_cpu_percent=100.0, max_other_cpu_share=1.0,
                              max_sleep_seconds=1e9, require_ac_power=False)


@dataclass(frozen=True)
class ResourceModel:
    """Estimated peak memory and run time of one E3 run.

    The memory estimate decides which runs E3 skips (above
    ``E3.memory_limit_gb``); the time estimate only schedules the jobs and
    gives the ETA. MB are 2**20 bytes and GB 2**30, as the runner reports
    them. Measured with ``/usr/bin/time -l`` on simulated constant-model
    genomes (Apple M1 Pro, 2026-10-05; times under load, so generous).
    """

    # C psmc keeps the forward and backward matrices of one sequence at a
    # time (64 doubles per bin each, allocated bin by bin), so its peak
    # follows the longest sequence, not the genome: 207 MB for one 20 Mb
    # sequence, 105 MB for two or four of 10 Mb, 404 MB for one of 40 Mb,
    # 1011 MB for one of 100 Mb.
    c_mb_base: float = 6.0
    c_mb_per_mb_longest: float = 10.05
    # psmc-rs 0.1 (e50d69d), one thread or one sequence: the forward
    # variables of whole sequences as f32 (2.48 MB per Mb), largest first,
    # while they fit its 2 GiB cache budget (PSMC_ALPHA_CACHE_MB unset),
    # plus one f64 forward buffer of the longest sequence (4.96 MB per Mb)
    # and the input (one byte per bin): 752 MB for one 100 Mb sequence,
    # 2497 MB for ten (eight cached). Several threads and sequences: one f64
    # buffer per sequence in work, nothing cached: 3977 MB for ten 100 Mb
    # sequences on 8 threads.
    rs01_mb_base: float = 5.0
    rs01_mb_per_mb_input: float = 0.0095
    rs01_mb_per_mb_cache: float = 2.48
    rs01_cache_mb: float = 2048.0
    rs01_mb_per_mb_buffer: float = 4.96
    # psmc-rs 0.2: the input (one byte per bin) and a few MB per sequence in
    # work: 12 MB for one 100 Mb sequence, 21 MB for ten on one thread,
    # 49 MB for ten on 8 threads.
    rs02_mb_base: float = 10.8
    rs02_mb_per_mb_input: float = 0.0103
    rs02_mb_per_extra_thread: float = 4.0
    # Run time per Mb and EM iteration on one thread, input and output
    # included (25 iterations: C psmc about 1 h for 3000 Mb). psmc-rs runs
    # min(threads, sequences) sequences at once, each extra one adding
    # parallel_efficiency of a thread. psmc-rs 0.1 on one thread computes the
    # forward pass of a sequence it could not cache twice (+50% for it).
    c_s_per_mb_iter: float = 0.05
    rs01_s_per_mb_iter: float = 0.03
    rs01_uncached_extra: float = 0.5
    rs02_s_per_mb_iter: float = 0.006
    parallel_efficiency: float = 0.7


@dataclass(frozen=True)
class E3:
    """Performance (Fig 4, Table 2, Fig S4): run time and peak memory of C psmc, psmc-rs 0.1 and 0.2.

    One genome per size, made of ``chromosome_length`` chromosomes under the
    ``model``, simulated as in E1 with the seed key
    ``E3/<model>/<replicate>/...``: the chromosomes are the same for every
    size, so each genome is the first chromosomes of the next larger one.
    C psmc (one thread; it has no threads), psmc-rs 0.1 and psmc-rs 0.2 (at
    each of their thread counts) fit every size with the inference recipe,
    ``repeats`` times; psmc-rs 0.2 also fits the thread-scaling size at
    every count of ``thread_scaling_threads``. Every run is an exclusive
    runner job (nothing else runs), timed as ``timing`` says. A run whose
    estimated peak memory (``resources``) exceeds ``memory_limit_gb`` is not
    run, and the tables say so and why.
    """

    model: str = "constant"
    replicate: int = 1  # seeds: seed("E3", model, replicate, "ancestry/chrN"), for every size
    genome_sizes_mb: Tuple[int, ...] = (100, 300, 1000, 3000)
    chromosome_length: int = 100_000_000  # sizes are made of 100 Mb chromosomes: 1, 3, 10 and 30
    simulation: Simulation = SIMULATION
    scaling: Scaling = SIMULATION_SCALING  # psmc-rs --mu and --gen-years: the scaling of its report only
    repeats: int = 3
    inference: Inference = INFERENCE
    rs01_threads: Tuple[int, ...] = (1, 8)
    rs02_threads: Tuple[int, ...] = (1, 8)
    thread_scaling_size_mb: int = 3000
    thread_scaling_threads: Tuple[int, ...] = (1, 2, 4, 8)
    memory_limit_gb: float = 12.0  # skip any run whose estimated peak exceeds this (and say why)
    rs01_commit: str = "e50d69d"
    # psmc-rs 0.1 has no --tr-ratio; its default start rho0 = theta0/5 is the
    # -r5 of the recipe, so it gets the other settings (pattern, t_max,
    # smoothing 0, M-step iterations, EM iterations) and its default start.
    # It has no --bin-size either: it reads a psmcfa as 100 bp bins.
    timing: TimingProtocol = TimingProtocol()
    resources: ResourceModel = ResourceModel()


@dataclass(frozen=True)
class MaskSpec:
    """A simulated callable mask: uncallable runs of exponential length, placed at random."""

    name: str
    missing_fraction: float
    structure: str  # "clustered" or "dispersed"
    mean_run_bp: float


@dataclass(frozen=True)
class E4:
    """Input preparation (Fig 5): routes from a VCF and a callable mask to PSMC."""

    model: str = "bottleneck"
    replicates: int = 5
    genome: Genome = GENOME_500MB
    simulation: Simulation = SIMULATION
    inference: Inference = INFERENCE
    scaling: Scaling = SIMULATION_SCALING
    masks: Tuple[MaskSpec, ...] = (
        MaskSpec("clustered10", 0.1, "clustered", 20_000.0),
        MaskSpec("clustered30", 0.3, "clustered", 20_000.0),
        MaskSpec("dispersed10", 0.1, "dispersed", 20.0),
        MaskSpec("dispersed30", 0.3, "dispersed", 20.0),
    )
    no_missing_control: bool = True
    # exact: VCF + --vcf-mask; fq2psmcfa: consensus FASTA -> fq2psmcfa (asserted equal to exact);
    # multihetsep: psmc-rs convert --multihetsep (cross-checked once with generate_multihetsep.py);
    # nomask: the same VCF without the mask.
    routes: Tuple[str, ...] = ("exact", "fq2psmcfa", "multihetsep", "nomask")
    # Routes whose input records which sites were callable (the VCF with its
    # mask, and multihetsep's called-site counts), so that the callable bases
    # per called bin are known. With fq2psmcfa's binning rule a bin is missing
    # only when nearly all its bases are uncallable, so a dispersed mask
    # leaves almost every bin called but holding only about (1 - f) of its
    # bases: theta per bin, N0 = theta / (4 mu s) and with them every time
    # and size shrink by about (1 - f). The tables report these routes twice:
    # as fitted, and rescaled with s = callable bases per called bin (from
    # psmc-rs qc) instead of the bin size ("<route>_rescaled"). The
    # fq2psmcfa route (a psmcfa) and the nomask route (every site counted as
    # called) cannot be rescaled: their inputs do not know the callable sites.
    rescaled_routes: Tuple[str, ...] = ("exact", "multihetsep")
    multihetsep_crosscheck_replicates: int = 1
    grid: TimeGrid = TimeGrid()
    rs_threads: int = 1


@dataclass(frozen=True)
class E5:
    """Bootstrap calibration (Fig 6).

    The truth: ``truth_replicates`` independent genomes of ``model``, each
    fitted once by psmc-rs 0.2, give the sampling distribution of log10 Ne(t)
    at every time of the grid (its SD is the true sampling SD). The
    bootstrap: each of the first ``bootstrap_datasets`` genomes is
    bootstrapped by psmc-rs 0.2 as C psmc does (``splitfa`` trunks drawn with
    replacement up to the genome's length, every replicate fitted from a
    fresh start with the same settings as the fits) with
    ``bootstrap_replicates`` replicates at each trunk size of ``trunk_bins``.
    Outcomes per grid time: bootstrap SD / true sampling SD of log10 Ne
    (median over the datasets), and the share of datasets whose ``band``
    percentile band covers the true Ne(t).

    Seeds: the genomes ``seed("E5", model, r, "ancestry/<chrom>")`` and
    ``"mutations/<chrom>"``, as every simulation; the bootstraps
    ``seed("E5", model, r, "bootstrap/trunk<bins>")``, one per dataset and
    trunk size.
    """

    model: str = "bottleneck"
    genome: Genome = Genome(n_chromosomes=6, chromosome_length=50_000_000)  # 300 Mb
    simulation: Simulation = SIMULATION
    inference: Inference = INFERENCE
    scaling: Scaling = SIMULATION_SCALING
    truth_replicates: int = 50  # independent genomes, each fitted once
    bootstrap_datasets: int = 10  # the first genomes, bootstrapped
    bootstrap_replicates: int = 50
    # psmc-rs --bootstrap-block-size: 500,000 bins is its default and splitfa's
    # (50 Mb at 100 bp, here one trunk per chromosome: 6 trunks); 50,000 bins
    # cuts 5 Mb trunks (60).
    trunk_bins: Tuple[int, ...] = (500_000, 50_000)
    band: Tuple[float, float] = (0.025, 0.975)  # percentile band (psmc-rs's summary uses these)
    grid: TimeGrid = TimeGrid()
    rs_threads: int = 1  # psmc-rs --threads per fit
    # psmc-rs --threads per bootstrap run: one each, so that the 20 runs of
    # about 45 CPU minutes share the cores without idle threads at their ends
    # (results do not depend on it).
    bootstrap_threads: int = 1
    example_dataset: int = 1  # the dataset whose bands Fig 6 shows
    # Pointwise coverage over 10 datasets moves in steps of 0.1 (Clopper-Pearson
    # 95%: 0.55-1.0 at 9/10), and the coverage averaged over the window's
    # strongly correlated points has no interval of its own. The summary
    # therefore also resamples the datasets (with replacement, this many
    # times, seed stage "summary/<window>/trunk<B>") and gives 95% percentile
    # intervals of the window-averaged coverage and of the median SD ratio.
    dataset_resamples: int = 10_000


# --- E6: supplementary analyses S1-S5 ---------------------------------------
# Every setting of E6 is in this block (experiments/e6.py reads nothing else
# of its own). E6 reuses what other experiments made: E1's genomes, fits and
# settings (S1-S3: models, inference, scaling, threads and time grid come
# from profile.e1), E3's psmc-rs 0.2 runs (S4) and E2's genomes and bootstrap
# trunk (S5). Its costs only schedule the jobs; no result depends on them.


@dataclass(frozen=True)
class E6Costs:
    """Expected run time and peak memory of E6's own jobs, for scheduling and ETA only.

    Measured on the development machine with E1's 500 Mb bottleneck replicate
    (10 x 50 Mb): re-simulating it with its TMRCA intervals took 11.5 s and
    261 MB; decoding its psmc-rs fit on 4 threads (5 million bins, a gzipped
    TSV of 114 MB) 5.6 s and 46 MB; the S3 tables 13-17 s and 1.3 GB.
    """

    convert_s_per_mb: float = 0.05  # psmc-rs convert (VCF -> psmcfa + multihetsep; multihetsep -> psmcfa)
    convert_s_base: float = 2.0
    convert_gb: float = 0.5
    tmrca_truth_s_per_mb: float = 0.025  # msprime re-simulation from the truth record, TMRCA intervals
    tmrca_truth_s_base: float = 5.0
    tmrca_truth_gb: float = 1.0  # every chromosome's tree sequence is held until the track is written
    decode_s_per_mb: float = 0.015  # psmc-rs posterior decoding, per-bin TSV (gzip) included
    decode_gb: float = 0.5
    tmrca_report_s_per_mb: float = 0.04  # reading 5 million posterior rows, the metrics
    tmrca_report_s_base: float = 5.0
    tmrca_report_gb: float = 2.5
    table_s: float = 20.0  # S1, S2, S4 and S5 tables
    table_gb: float = 1.0
    qc_s: float = 5.0  # psmc-rs qc of a real genome (0.1 s for 1.9 Gb)
    qc_gb: float = 0.5
    figure_s: float = 30.0
    figure_gb: float = 1.0


@dataclass(frozen=True)
class E6:
    """Supplementary analyses built on E1-E3 (Figs S1-S4, Table S5).

    S1  EM convergence: E1's log-likelihood and parameter traces, both tools.
    S2  Format consistency: replicate ``format_replicate`` of every E1 model
        read as psmcfa (E1's psmc-rs fit), as its VCF (no missing data, so no
        mask) and as multihetsep (``psmc-rs convert --multihetsep`` of that
        VCF); the bins must agree except for the multihetsep's missing tail
        after the last heterozygous site, which is counted; the fits are compared.
    S3  TMRCA decoding: E1's psmc-rs fit of ``tmrca_model`` replicate
        ``tmrca_replicate`` decoded per bin (``--tmrca-out``) and compared with
        the true pairwise TMRCA re-simulated from the replicate's seeds.
    S4  Thread scaling and determinism: E3's psmc-rs 0.2 runs.
    S5  QC table: ``psmc-rs qc`` of E2's four real genomes.
    """

    # S1. EM traces. An iteration "converged" once the log-likelihood is
    # within this many units of the last iteration both tools evaluate
    # (C psmc never evaluates its final model), and a decrease larger than
    # the tolerance counts as non-monotone (both M-steps are numerical
    # searches, so EM's monotonicity holds only up to their precision).
    convergence_within_loglike: float = 1.0
    loglike_decrease_tolerance: float = 1e-3

    # S2. The E1 replicate of every model read as psmcfa, VCF and multihetsep.
    format_replicate: int = 1

    # S3. TMRCA decoding. psmc-rs writes the 2.5% and 97.5% quantiles of the
    # per-bin posterior, so the interval is the 95% one. The example track is
    # the first 5 Mb of chr1 (fixed, not chosen after looking), shortened to the
    # chromosome in profiles with shorter ones. Calibration is tabulated in
    # deciles of the true TMRCA and of the posterior mean; the density of
    # (true, posterior mean) is binned on a log10 grid of this many cells per axis.
    tmrca_model: str = "bottleneck"
    tmrca_replicate: int = 1
    tmrca_interval: float = 0.95
    tmrca_example_track_bp: int = 5_000_000
    tmrca_example_chrom: str = "chr1"
    tmrca_example_start_bp: int = 0
    tmrca_calibration_classes: int = 10
    tmrca_density_cells: int = 60
    tmrca_decode_threads: int = 4  # psmc-rs --threads; the posterior does not depend on it

    # S4. The experiment whose psmc-rs 0.2 runs are read (see experiments/e6.py).
    thread_scaling_experiment: str = "E3"

    # S5. psmc-rs qc of E2's genomes, with E2's bin size and bootstrap trunk
    # (--bootstrap-block-size), as E2 bootstraps them.
    qc_experiment: str = "E2"

    costs: E6Costs = E6Costs()


# ---------------------------------------------------------------------------
# 6. Runner: resource budget and cost model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Budget:
    """What the runner may use at once; exclusive jobs run alone."""

    cores: int = 7
    mem_gb: float = 12.0


@dataclass(frozen=True)
class Costs:
    """Expected run time and peak memory of the jobs, for scheduling and ETA only.

    Measured on the development machine (Apple M1 Pro, 8 cores, 16 GB) while
    other work kept its load average near 6, with 25 EM iterations of the
    64-interval pattern on one 50 Mb chromosome. Results never depend on them.
    """

    # msprime ancestry and mutations, psmcfa, VCF (bgzip, tabix) and records:
    # 3 s (constant) to 5 s (zigzag) and 0.2 GB per 50 Mb chromosome
    simulate_s_per_mb: float = 0.1
    simulate_s_base: float = 5.0
    simulate_gb: float = 1.0
    # C psmc: time grows with bins x iterations (85 s for 50 Mb x 25); memory
    # with the longest sequence, whose forward-backward matrices it holds at
    # once (0.51 GB for 50 Mb; 5 GB for one sequence of 500 Mb)
    c_psmc_s_per_mb_iter: float = 0.06
    c_psmc_gb_per_mb_longest: float = 0.0105
    c_psmc_gb_base: float = 0.1
    # psmc-rs 0.2, one thread: 9.1 s and 11 MB for 50 Mb x 25 (63 s, 32 MB for 500 Mb)
    rs02_s_per_mb_iter: float = 0.007
    rs02_gb: float = 0.2
    # Tables and figures
    report_s: float = 30.0
    report_gb: float = 1.0


# ---------------------------------------------------------------------------
# 7. Profiles
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Profile:
    """A complete set of experiment parameters."""

    name: str
    e1: E1 = E1()
    e2: E2 = E2()
    e3: E3 = E3()
    e4: E4 = E4()
    e5: E5 = E5()
    e6: E6 = E6()
    budget: Budget = Budget()
    costs: Costs = Costs()


PRODUCTION = Profile(name="production")

# -- E5 (bootstrap calibration) at the smoke and tiny sizes ----------------
# Smoke: 8 genomes of 2 x 2 Mb, the first 3 bootstrapped with 8 replicates at
# trunks of 5,000 and 2,000 bins (8 and 20 trunks, as production's 6 and 60).
# Tiny (unit tests): 3 genomes of one 300 kb chromosome, the first
# bootstrapped with 3 replicates at trunks of 1,000 and 500 bins.
_SMOKE_E5 = dict(truth_replicates=8, bootstrap_datasets=3, bootstrap_replicates=8, trunk_bins=(5_000, 2_000))
_TINY_E5 = dict(truth_replicates=3, bootstrap_datasets=1, bootstrap_replicates=3, trunk_bins=(1_000, 500))
# ---------------------------------------------------------------------------

# The same code paths at a size that runs in a minute or two: 2 models x 2
# replicates x 2 chromosomes of 2 Mb, 3 EM iterations.
_SMOKE_GENOME = Genome(n_chromosomes=2, chromosome_length=2_000_000)
_SMOKE_INFERENCE = replace(INFERENCE, n_iter=3)
SMOKE = Profile(
    name="smoke",
    e1=E1(models=("constant", "bottleneck"), replicates=2, genome=_SMOKE_GENOME, inference=_SMOKE_INFERENCE),
    e2=E2(bootstrap_replicates=4, inference=_SMOKE_INFERENCE, rs_threads=2),
    e3=E3(genome_sizes_mb=(2, 4), chromosome_length=2_000_000, repeats=1, inference=_SMOKE_INFERENCE,
          rs01_threads=(1, 2), rs02_threads=(1, 2), thread_scaling_size_mb=4, thread_scaling_threads=(1, 2),
          timing=QUICK_TIMING),
    e4=replace(E4(), replicates=1, genome=_SMOKE_GENOME, inference=_SMOKE_INFERENCE),
    e5=replace(E5(), genome=_SMOKE_GENOME, inference=_SMOKE_INFERENCE, **_SMOKE_E5),
)

# For the unit tests: one replicate of one model, one chromosome of 300 kb,
# 2 EM iterations (seconds end to end).
_TINY_GENOME = Genome(n_chromosomes=1, chromosome_length=300_000)
TINY = Profile(
    name="tiny",
    e1=E1(models=("bottleneck",), replicates=1, genome=_TINY_GENOME, inference=replace(INFERENCE, n_iter=2)),
    # E3: one genome of two 0.5 Mb chromosomes, every tool, two repeats, 2 EM iterations.
    e3=E3(genome_sizes_mb=(1,), chromosome_length=500_000, repeats=2, inference=replace(INFERENCE, n_iter=2),
          rs01_threads=(1,), rs02_threads=(1, 2), thread_scaling_size_mb=1, thread_scaling_threads=(1, 2),
          timing=QUICK_TIMING),
    # E4: two chromosomes (the multihetsep cross-check runs per chromosome),
    # the control and one mask of each structure, clustered runs shortened
    # so that a 300 kb chromosome holds several.
    e4=replace(
        E4(), replicates=1, genome=Genome(n_chromosomes=2, chromosome_length=300_000),
        inference=replace(INFERENCE, n_iter=2),
        masks=(MaskSpec("clustered30", 0.3, "clustered", 5_000.0), MaskSpec("dispersed30", 0.3, "dispersed", 20.0)),
    ),
    e5=replace(E5(), genome=_TINY_GENOME, inference=replace(INFERENCE, n_iter=2), **_TINY_E5),
)

PROFILES: Dict[str, Profile] = {p.name: p for p in (PRODUCTION, SMOKE, TINY)}

# ---------------------------------------------------------------------------
# 8. E4, input preparation: how its steps run, beyond the design in E4 above
# ---------------------------------------------------------------------------
#
# The routes from one masked VCF to PSMC (experiments/e4.py, inputprep.py):
#
#   exact        psmc-rs on the VCF, mask mode, --vcf-mask callable.bed.gz
#   fq2psmcfa    consensus FASTA (hom: base, het: IUPAC, uncallable: N) streamed
#                into fq2psmcfa -s <bin_size> (its defaults otherwise: -g 10000,
#                and -q, which needs qualities, does nothing), then psmc-rs on
#                the psmcfa; its bins must equal those of `psmc-rs convert` on
#                the exact route, and the job that compares them fails if not
#   multihetsep  `psmc-rs convert --multihetsep` of the exact route's input
#                (nr_called = callable sites since the previous row, inclusive),
#                read back by psmc-rs (uncalled sites at the start of each
#                interval); replicate 1 of every mask is also checked row for
#                row against msmc-tools' generate_multihetsep.py, run on each
#                chromosome with its mask
#   nomask       psmc-rs on the same VCF without --vcf-mask: uncallable sites
#                count as called homozygous, as with a variants-only VCF
#
# Masks: an alternating renewal process per chromosome (inputprep.py): callable
# stretches and uncallable runs of geometric length (the discrete exponential),
# with mean run length MaskSpec.mean_run_bp and mean callable stretch
# mean_run_bp * (1 - f) / f, so that the expected missing fraction is f. Records
# at uncallable sites are removed from the VCF (a caller does not call them).
#
# Seeds (section 4), experiment "E4", model "bottleneck": the simulation's
# ancestry/<chrom> and mutations/<chrom>; mask/<mask>/<chrom> for each mask; and
# reference/<chrom> for the ancestral bases of the consensus FASTA at sites
# without a record (uniform over ACGT, JC69's stationary distribution; they
# do not change any bin).

E4_CONTROL_MASK = "nomissing"  # the no-missing control: every site callable


@dataclass(frozen=True)
class E4Costs:
    """Expected run time and peak memory of E4's steps, for scheduling and ETA only.

    Measured on the development machine (load average 5 to 9) on one 50 Mb
    chromosome of the bottleneck model with the dispersed 30% mask, the
    largest (750,000 callable intervals), and scaled by genome size. Fits use
    :class:`Costs`' psmc-rs rate (measured here: 11.3 s for 50 Mb x 25
    iterations, 35 MB with the dispersed mask).
    """

    # draw the mask, write the BED, filter, bgzip and index the VCF: 0.7 s per 50 Mb;
    # 5.4 s and 0.65 GB for the whole 500 Mb genome
    mask_s_per_mb: float = 0.02
    mask_gb: float = 1.0
    # consensus FASTA streamed through fq2psmcfa (one chromosome at a time):
    # 2.5 s per 50 Mb; 28 s and 1.1 GB (Python and fq2psmcfa) for 500 Mb
    consensus_s_per_mb: float = 0.06
    consensus_gb: float = 1.5
    # psmc-rs convert and qc: one pass over the input, 0.15 s and 30 MB per 50 Mb
    convert_s_per_mb: float = 0.005
    convert_gb: float = 0.3
    # the bins of four routes compared: 0.14 s per 50 Mb
    bins_s_per_mb: float = 0.005
    bins_gb: float = 0.5
    # generate_multihetsep.py, a Python loop over every site: 36 s per 50 Mb
    crosscheck_s_per_mb: float = 0.75
    crosscheck_gb: float = 0.5
    # psmc-rs fit (the VCF routes hold the merged mask intervals)
    fit_gb: float = 0.3
    figure_s: float = 20.0


E4_COSTS = E4Costs()


def get_profile(name: str) -> Profile:
    """The profile called ``name`` (``production``, ``smoke`` or ``tiny``)."""
    try:
        return PROFILES[name]
    except KeyError:
        raise KeyError(f"unknown profile {name!r}; known: {', '.join(PROFILES)}") from None


# ---------------------------------------------------------------------------
# 8. E2 (real genomes) in the smoke and tiny profiles
# ---------------------------------------------------------------------------
# The real genomes hold 0.25 to 1.94 Gb in bins, and C psmc needs about 50
# minutes and up to 3.4 GB for each turtle. The smoke and tiny profiles run
# the same jobs on the first bins of each genome (job E2/subset/<sample>),
# with trunks small enough to leave several of them for the bootstrap.

# smoke: 20 Mb of each of the four genomes, 3 EM iterations, 4 bootstrap
# replicates on 2 Mb trunks (10 per genome, so each genome also gets the
# second bootstrap, on trunks of 0.5 Mb) (about a minute in all).
SMOKE = replace(SMOKE, e2=replace(SMOKE.e2, max_bins=200_000, bootstrap_trunk_bins=20_000, fine_trunk_bins=5_000))

# tiny (unit tests): 2 Mb of the human chromosome and of one turtle,
# 2 EM iterations, 2 bootstrap replicates on 500 kb trunks (seconds).
TINY = replace(
    TINY,
    e2=E2(
        genomes=tuple(g for g in REAL_GENOMES if g.sample_id in ("Papuan", "HLhydTec1")),
        inference=replace(INFERENCE, n_iter=2),
        bootstrap_replicates=2,
        bootstrap_trunk_bins=5_000,
        fine_trunk_bins=1_000,
        rs_threads=2,
        max_bins=20_000,
    ),
)

PROFILES = {p.name: p for p in (PRODUCTION, SMOKE, TINY)}


__all__ = [
    "Budget", "Costs", "DemographicModel", "E1", "E2", "E3", "E4", "E4Costs", "E4_COSTS", "E4_CONTROL_MASK",
    "E5", "E6", "Genome",
    "INFERENCE", "Inference", "MODELS", "MODEL_ORDER", "MaskSpec", "PROFILES", "Profile",
    "REAL_GENOMES", "RealGenome", "SIMULATION", "SIMULATION_SCALING", "Scaling", "Simulation",
    "SizeChange", "TimeGrid", "ZIGZAG_STDPOPSIM", "get_profile", "ms_model", "seed",
]

# ---------------------------------------------------------------------------
# 8. Figures (experiment/pipeline/figures): what the figures show. Their
#    style (type, sizes, colours) is in figures/style.py; the tables they
#    read are described in figures/schemas.py. No result depends on these.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Figures:
    """Choices of the figures that are not style: grids, axis ranges, examples."""

    # Replicate curves are summarised (medians, percentile bands) at this many
    # log-spaced times across the axis.
    curve_grid_points: int = 600
    # Time axis (years) of the Ne(t) panels of simulations (Fig 2, 5, 6, S2):
    # the fitted curves of every model start after 1 ky and the informative
    # windows end before 2.5 My (see TimeGrid).
    sim_years: Tuple[float, float] = (1e3, 1e7)
    # Fig 5: the replicate whose curves are shown, the mask of the example
    # panel, and the masks whose bias over time is shown.
    e4_example_replicate: int = 1
    e4_example_mask: str = "dispersed30"
    e4_bias_masks: Tuple[str, ...] = ("clustered30", "dispersed30")
    # Fig 6: the bootstrapped dataset shown with its bands.
    e5_example_dataset: int = 1
    # Runner jobs (scheduling only): one figure takes seconds and well under 1 GB.
    job_seconds: float = 20.0
    job_gb: float = 1.0


FIGURES = Figures()

__all__ += ["FIGURES", "Figures"]
