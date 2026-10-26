"""Every parameter of the manuscript's experiments, in one place.

No other module of the pipeline holds a number that shapes a result: the
experiments read their settings from here, and every job records the
settings it ran with in its provenance record. The file is organised as the
experimental design is:

1. inference settings shared by every PSMC fit (both tools);
2. scaling to years and individuals;
3. simulation settings and the demographic models;
4. the seed scheme;
5. the experiments E1-E7 (each with the settings of its steps);
6. the figures (what they show; their style is figures/style.py);
7. the runner's resource budget and cost model (scheduling only);
8. the profiles: ``production`` (the manuscript), ``smoke`` (a small
   end-to-end check of the same code paths) and ``tiny`` (unit tests).

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
    ``psmc -N25 -t15 -r5 -p "4+25*2+4+6"`` (psmc-rs: ``N_ITER 25 --t-max 15
    --tr-ratio 5 --pattern 4+25*2+4+6``), plus ``--smooth-lambda 0`` (C
    psmc's objective has no penalty) and ``--mstep-iters 1000``.

    The M-step: C psmc's Hooke-Jeeves search runs each M-step to
    convergence, and so does psmc-rs 0.2's L-BFGS, which stops by a
    convergence test far below the ``--mstep-iters`` cap (in E1 a median 76
    objective evaluations per M-step, at most 135; under a millisecond per
    EM iteration on one thread, 0.55 to 0.84 ms on E3's 100 Mb genome on an
    Apple M1 Pro). Over E1's 40 genomes the RMSE of log10 Ne between the two
    tools' curves on the informative window, interval by interval, has a
    median of 2.5e-5, as small as with C psmc's own Hooke-Jeeves search run
    inside psmc-rs's EM in a diagnostic build (median 2.6e-5;
    experiment/c_equivalence/). The default M-step is therefore the
    converged one, and no experiment fits a separate capped arm
    (:attr:`E1.capped_mstep_iters` and :attr:`E3.capped_mstep_iters` are
    None; a cap set there fits or times psmc-rs with M-steps capped at that
    many iterations too).
    """

    n_iter: int = 25  # EM iterations: C -N, psmc-rs N_ITER
    t_max: float = 15.0  # initial maximum time, in 2*N0 generations: C -t, --t-max
    tr_ratio: float = 5.0  # initial theta0/rho0: C -r, --tr-ratio
    pattern: str = "4+25*2+4+6"  # free-parameter pattern (64 intervals, 28 free lambdas): C -p, --pattern
    smooth_lambda: float = 0.0  # psmc-rs --smooth-lambda; 0 is C psmc's objective
    mstep_iters: int = 1000  # psmc-rs --mstep-iters: its default, a cap that converged M-steps stay below
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


# The models' settings, and so every simulation's truth record and job key,
# hold this text: where the ms -eN events come from, a script of commit
# 516daa27, the last commit that holds it.
_MS_SOURCE = (
    "ms-style model of experiment/scripts/simulate_msprime_to_psmcfa.py at commit 516daa27, converted exactly"
)

CONSTANT = ms_model("constant", 10_000, (), _MS_SOURCE)
BOTTLENECK = ms_model("bottleneck", 20_000, ((0.01, 0.05), (0.015, 0.5), (0.05, 0.25), (0.5, 0.5)), _MS_SOURCE)
EXPANSION = ms_model("expansion", 10_000, ((0.01, 0.1), (0.06, 1.0), (0.2, 0.5), (1.0, 1.0), (2.0, 2.0)), _MS_SOURCE)

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


# The dense grid (years: from, to, points) on which the true Ne(t) of a model
# is tabulated for the figures (E1_truth, E4_truth, E6_S2_curves).
TRUTH_TABLE_YEARS: Tuple[float, float, int] = (1e2, 1e8, 400)
# The quantiles psmc-rs writes in a bootstrap's summary.json (its format, not
# a setting): E2 and E5 recompute them from the replicates as a check.
PSMC_RS_SUMMARY_QUANTILES: Tuple[float, float, float] = (0.025, 0.5, 0.975)


@dataclass(frozen=True)
class E1:
    """Simulated accuracy and equivalence (the article's Fig 1 and Table 1, the supplement's Fig S1 and
    Table S1; fig2_simulated_accuracy).

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
    # An M-step cap to fit every genome with as well (job fit-rs-capped,
    # table E1_mstep.csv), against C psmc and the main fit; None: no such fit
    # (the default M-step converges, see Inference). A small cap (say 5)
    # shows what an M-step stopped short of convergence does.
    capped_mstep_iters: Optional[int] = None


@dataclass(frozen=True)
class RealGenome:
    """One empirical genome of E2 (a psmcfa, plain or gzip-compressed, in experiment/real_data)."""

    sample_id: str
    file: str
    species: str
    label: str  # how figures and tables name it
    scaling: Scaling
    scaling_source: str  # where mu and g come from, in full (E2_fit_summary.scaling_source)
    scaling_cite: str  # the same in a few words, for a figure panel (Fig 3)
    sequence_noun: str = "scaffolds"  # what its sequences are, in Fig 3's notes ("22 chromosomes split at 50 Mb")
    # The file's SHA-256 where its source records one (the tests check the
    # committed file against it); None: not pinned.
    sha256: Optional[str] = None
    # Another experiment that bootstraps this psmcfa too (E7 for NA12878).
    # E2's bootstrap of it at the default trunks is then that experiment's
    # run, with its seed, seed(<experiment>, sample_id, 0, "bootstrap"), and
    # E2 plans a run of the whole file only when that experiment's
    # replicates, trunks and inference (the replicates' iterations among it)
    # are E2's: one seed and one set of settings, so both bootstraps write the
    # same files. None: E2's own seed.
    bootstrap_shared_with: Optional[str] = None


# The three turtles are pleurodires (side-necked turtles). No pleurodire has
# a measured mutation rate; the only direct (pedigree) rate of any turtle is
# that of nine trios of the painted turtle Chrysemys picta, a cryptodire
# (Bergeron et al. 2023: 3.04e-9 to 6.56e-9 per bp per generation, mean
# 4.61e-9; the paper's 2.3e-10 per year divides it by an assumed g of 20).
# The psmc-rs 0.1 runs used 7.9e-9, a crocodilian rate: Green et al. 2014
# derived it from the alligator-crocodile divergence with a 20-year
# generation. g 15 years is shared by the three species: close to the values
# of Sozzoni et al. 2025 (Table S2) for P. castaneus (15.9) and E. subglobosa
# (12.4); H. tectifera has no published age at maturity or generation time.
# Ne and time scale as 1/mu, and time also as g (figures/README.md: Fig 3).
_TURTLE_SCALING = Scaling(4.6e-9, 15.0)
_TURTLE_SCALING_SOURCE = (
    "mu 4.6e-9 per bp per generation: the mean germline rate of 9 Chrysemys picta trios "
    "(Bergeron et al. 2023, Nature 615:285, Supp. Table 2; range 3.0e-9 to 6.6e-9), the only turtle with a "
    "pedigree estimate; C. picta is a cryptodire, and no pleurodire rate has been measured. "
    "g 15 y, shared by the three species: close to Sozzoni et al. 2025 (Mol Ecol Resour 25:e70040, Table S2: "
    "P. castaneus 15.9, E. subglobosa 12.4); H. tectifera has no published generation time. "
    "The crocodilian rate 7.9e-9 (Green et al. 2014, Science 346:1254449) is higher."
)
_TURTLE_SCALING_CITE = "painted turtle trios (Bergeron 2023)"
# The human scaling, shared with E7 (E7.scaling and scaling_source), which
# fits the same NA12878 psmcfa.
_HUMAN_SCALING = Scaling(1.25e-8, 29.0)
_HUMAN_SCALING_SOURCE = (
    "mu 1.25e-8: the pedigree-based human rate (Scally & Durbin 2012, Nat Rev Genet 13:745); g 29: the "
    "mean human generation interval of Fenner 2005 (Am J Phys Anthropol 128:415), as in Bergstrom et al. "
    "2020 (Science 367:eaay5012) and Cousins, Scally & Durbin 2025 (Nat Genet 57:856)"
)
_HUMAN_SCALING_CITE = "Scally & Durbin 2012; g: Fenner 2005"


def _turtle(sample_id: str, species: str) -> RealGenome:
    return RealGenome(sample_id, f"{sample_id}_diploid.psmcfa", species, f"{species} ({sample_id})",
                      _TURTLE_SCALING, _TURTLE_SCALING_SOURCE, _TURTLE_SCALING_CITE)


REAL_GENOMES: Tuple[RealGenome, ...] = (
    _turtle("HLemySub1", "Emydura subglobosa"),
    _turtle("HLhydTec1", "Hydromedusa tectifera"),
    _turtle("HLpelCas1", "Pelusios castaneus"),
    # NA12878 (CEU; GIAB's HG001): chr1-22 of the 1000 Genomes 30x NYGC
    # joint-called VCF, 22 sequences of 28,750,027 bins in all (GRCh38's
    # lengths / 100; 59 trunks of 50 Mb), one gzip member per chromosome.
    # It is E7's inputs/psmcfa/NA12878.psmcfa.gz byte for byte (kg1000.py:
    # the GRCh38 strict mask, every non-PASS position missing), and sha256 is
    # that file's entry in E7's MANIFEST.json. E7 bootstraps it too
    # (E7.bootstrap_samples), and E2's bootstrap of it is E7's run, seed and
    # settings (the same files). Its band is drawn from E2's tables
    # (fig3_real_genomes a, the article's Fig 1c); no figure draws E7's.
    RealGenome(
        "NA12878", "NA12878_1kGP30x_diploid.psmcfa.gz", "Homo sapiens",
        "Homo sapiens (NA12878, CEU)", _HUMAN_SCALING, _HUMAN_SCALING_SOURCE, _HUMAN_SCALING_CITE,
        sequence_noun="chromosomes",
        sha256="a9501bcc50825dd310c7e67547ce65139cd98a4ecc6f23fa9c3f69d14aadcd35",
        bootstrap_shared_with="E7",
    ),
)


@dataclass(frozen=True)
class E2:
    """Real genomes (the article's Fig 1 and Table 1; fig3_real_genomes, figS_turtle_patterns): both tools,
    psmc-rs bootstrap and qc on each.

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
    # A second bootstrap, on the trunks E5 also calibrates (50,000 bins,
    # 5 Mb), of every genome with fewer than ``few_trunks`` default trunks,
    # whose default band rests on that few resampling units. No production
    # genome has so few (NA12878 59, HLhydTec1 51, HLemySub1 897, HLpelCas1
    # 1,220), so production runs none; the smoke and tiny profiles' prefixes
    # do. E5 calibrates bootstraps of 50 Mb chromosomes cut into 50 Mb and
    # 5 Mb trunks; the turtles' default bootstraps resample whole scaffolds of
    # very different lengths (50%, 23% and 97% of the bins of HLemySub1,
    # HLhydTec1 and HLpelCas1 lie in scaffolds shorter than a trunk), a regime
    # E5 does not calibrate. None: no second bootstrap.
    fine_trunk_bins: Optional[int] = 50_000
    band: Tuple[float, float] = (0.025, 0.975)
    # The bootstrap fits replicates in parallel, one per thread; its output
    # does not depend on the threads (replicate i draws from
    # replicate_rng(seed, i)). 50: the turtles' bootstraps were E2's critical
    # path on 7 threads.
    rs_threads: int = 50
    # Seeds: seed("E2", sample_id, 0, "bootstrap") for the default trunks
    # (seed("E7", ...) for NA12878: RealGenome.bootstrap_shared_with),
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
    panel_order: Tuple[str, ...] = ("NA12878", "HLemySub1", "HLhydTec1", "HLpelCas1")  # fig3_real_genomes a-d
    # Below this many trunks psmc-rs warns that the bands are coarse
    # (FEW_TRUNKS in src/bin/psmc/run.rs); the tables and Fig 3 say so. Of
    # the production genomes HLhydTec1 forms the fewest trunks of the default
    # 500,000 bins, 51 (NA12878's chr1-22: 59).
    few_trunks: int = 20

    # -- The split-first-window check (E2_split_patterns, figS_turtle_patterns) --
    # Hilgers et al. 2025 (Curr Biol 35:927) found false recent peaks of Ne in
    # PSMC curves across many studies, in all their turtles among them, under
    # the default pattern 4+25*2+4+6: its first free lambda spans the first
    # four intervals, and splitting that window removed the peaks. psmc-rs
    # fits each genome of ``split_genomes`` again with each pattern of
    # ``split_patterns`` (name, pattern; job ``fit-rs-<name>``, every other
    # setting fit-rs's), and the table compares each fit with fit-rs's. Both
    # patterns keep the 64 intervals.
    split_patterns: Tuple[Tuple[str, str], ...] = (("split2", "2+2+25*2+4+6"), ("split1", "1+1+1+1+25*2+4+6"))
    split_genomes: Tuple[str, ...] = ("HLemySub1", "HLhydTec1", "HLpelCas1")
    # The recent intervals a false peak lies in: the default pattern's first
    # window (4 intervals) and the two 2-interval windows after it. The
    # plateau is the median Ne of the next as many intervals. A fit has a
    # recent peak when the largest Ne of the recent intervals lies after the
    # first interval and is at least ``split_peak_ratio`` times both the first
    # interval's Ne and the plateau.
    split_recent_intervals: int = 8
    split_peak_ratio: float = 1.25
    # Smoke and tiny profiles only: fit the first ``max_bins`` bins of each
    # genome (records in file order, the last one cut) instead of the whole
    # file. None (production) fits the packaged files themselves.
    max_bins: Optional[int] = None


# --- E3: how a run is timed, and the memory and time model of each tool ----


@dataclass(frozen=True)
class TimingProtocol:
    """How ``experiments/e3_run.py measure`` times one run of a tool.

    Before the run it waits ``settle_seconds``, then until the CPU use of the
    whole machine (all but idle and iowait time: processes, kernel threads,
    interrupts and steal) over ``quiet_window_seconds`` is at most
    ``quiet_cpu_percent`` of all cores and at most ``quiet_max_cores`` cores.
    During the run it samples the RSS of the tool's process tree every
    ``rss_sample_seconds`` and keeps the larger of that peak and the tool's
    own ``ru_maxrss`` (never below the peak, but on Linux it also counts the
    memory the process held when launched from the Python wrapper, 26 MB on
    Ibex in E3_runs.csv, so it is an upper bound for a tool that needs less);
    afterwards it records the CPU time other processes used while it ran.

    A run is kept only when it was clean: the machine became quiet within
    ``max_wait_seconds``, other processes used at most
    ``max_other_cpu_share`` of all cores and at most ``max_other_cores``
    cores on average during it, the machine did not sleep (wall-clock span
    within ``max_sleep_seconds`` of the monotonic run time) and, with
    ``require_ac_power``, it ran on AC power without Low Power Mode.
    Otherwise ``measure`` writes no record and exits with status 75, and the
    runner defers the run (starts it again later).

    "Other processes" are the user-space processes other than the tool's and
    the timing wrapper, each one's CPU time with that of the children it
    reaped, read before and after the run (Linux): a process that ended
    during the run counts through the one that waited for it, and only one
    that nothing waits for is missed (it shows in
    ``during.unaccounted_cores``, recorded, not limited). Steal time (another
    machine's, under a hypervisor) is limited with them. Kernel threads and
    interrupt time are not: part of it is the kernel's work for the tool (a
    run keeping 30 cores busy on a 128-core node adds about 0.2 core of it,
    0.17 to 0.20 more than the same run on one thread),
    the rest housekeeping that the quiet check before the run bounds, and
    neither is another workload. Where not every process can be read (other
    users' processes on macOS, where reaped children's CPU is not readable
    either), the machine's busy CPU time but the tool's is limited instead,
    kernel and interrupt time included, so that nothing is under-counted;
    each record says which (``during.other_method``).

    Each limit is the smaller of a share of all cores and a number of cores:
    on the 8 to 10 cores of a laptop the shares decide (0.8 to 1 core busy
    before a run, 0.4 to 0.5 core of other work during it); on a node of 128
    or 192 cores the shares would let 6 to 19 busy cores pass, and the core
    counts decide.
    """

    settle_seconds: float = 10.0
    quiet_cpu_percent: float = 10.0
    quiet_max_cores: float = 1.0
    quiet_window_seconds: float = 2.0
    max_wait_seconds: float = 180.0
    rss_sample_seconds: float = 0.5
    max_other_cpu_share: float = 0.05  # of all cores, over the run
    max_other_cores: float = 0.5  # core-equivalents (CPU seconds of other processes / wall seconds)
    max_sleep_seconds: float = 2.0
    require_ac_power: bool = True


# For the smoke and tiny profiles: no waiting, one short look at the machine,
# and every run kept whatever else the machine does.
QUICK_TIMING = TimingProtocol(
    settle_seconds=0.0, quiet_window_seconds=0.2, max_wait_seconds=0.0, rss_sample_seconds=0.05,
    quiet_cpu_percent=100.0, quiet_max_cores=1e9, max_other_cpu_share=1.0, max_other_cores=1e9,
    max_sleep_seconds=1e9, require_ac_power=False,
)


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
    # psmc-rs 0.2: the input (one byte per bin) and a few MB per sequence in
    # work: 12 MB for one 100 Mb sequence, 21 MB for ten on one thread,
    # 49 MB for ten on 8 threads.
    rs02_mb_base: float = 10.8
    rs02_mb_per_mb_input: float = 0.0103
    rs02_mb_per_extra_thread: float = 4.0
    # Run time per Mb and EM iteration on one thread, input and output
    # included (25 iterations: C psmc about 1 h for 3000 Mb). psmc-rs runs
    # min(threads, sequences) sequences at once, each extra one adding
    # parallel_efficiency of a thread.
    c_s_per_mb_iter: float = 0.05
    rs02_s_per_mb_iter: float = 0.006
    parallel_efficiency: float = 0.7
    # psmc-rs 0.2's M-step converges in under a millisecond per EM iteration
    # whatever the size (its cost follows the 64 states and the free
    # parameters, not the bins: 0.55 to 0.84 ms on E3's 100 Mb genome on an
    # Apple M1 Pro); rs02_mstep_s_per_iter is generous, and covers a capped arm.
    rs02_mstep_s_per_iter: float = 0.02


@dataclass(frozen=True)
class E3:
    """Performance (the article's Fig 2 and Table 2; fig4_performance, figS4_thread_scaling): run time and
    peak memory of C psmc and psmc-rs 0.2.

    One genome per size, made of ``chromosome_length`` chromosomes under the
    ``model``, simulated as in E1 with the seed key
    ``E3/<model>/<replicate>/...``: the chromosomes are the same for every
    size, so each genome is the first chromosomes of the next larger one.
    C psmc (one thread; it has no threads) and psmc-rs 0.2 (at each of
    ``rs02_threads``) fit every size with the inference recipe,
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
    repeats: int = 5
    inference: Inference = INFERENCE
    # Thread counts. psmc-rs works on one sequence per thread, so the 30
    # chromosomes of the largest genome make 30 threads the most that can
    # help; every run is exclusive, and gets as many cores as threads.
    rs02_threads: Tuple[int, ...] = (1, 8, 30)
    # psmc-rs 0.2 timed with its M-steps capped at capped_mstep_iters too
    # ("rs02-capped"), at these thread counts (those of rs02_threads); None:
    # no such runs. Every other run uses inference.mstep_iters, psmc-rs's
    # default, whose M-steps converge (see Inference).
    rs02_capped_threads: Tuple[int, ...] = (1, 8, 30)
    capped_mstep_iters: Optional[int] = None
    thread_scaling_size_mb: int = 3000
    thread_scaling_threads: Tuple[int, ...] = (1, 2, 4, 8, 16, 30)
    memory_limit_gb: float = 64.0  # skip any run whose estimated peak exceeds this (and say why)
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
    """Input preparation (the supplement's Fig S2, fig5_input_preparation): routes from a VCF and a
    callable mask to PSMC.

    fq2psmcfa's binning calls a bin missing only when more than 90% of its
    bases are uncallable, so missing data dispersed in short runs leave
    almost every bin called but holding only about (1 - f) of its bases.
    C psmc takes every called bin as ``bin_size`` bases, so theta per bin,
    N0 = theta / (4 mu s) and every time and size come out low, by
    somewhat more than the share of bases missing from the called bins
    (about 1.2 times it in the production run: theta -12% with 10% and
    -35% with 30% dispersed missing). psmc-rs's callable-length model
    (``--bin-callable auto``, its default; docs/design/bin-callable-model.md)
    weighs each bin by the callable bases the VCF and its mask say it holds.
    E4 fits both models on the same bins, the classic fq2psmcfa route,
    multihetsep (whose callable bases are approximate, so that
    ``--bin-callable auto`` fits its bins with C psmc's model) and the VCF
    without its mask, each against the truth and against the complete-data
    fit of the same genome.
    """

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
    # The routes fitted (below, with the --bin-callable of each in
    # E4_BIN_CALLABLE). mask_auto and mask_ignore are required: they are the
    # corrected route and the fq2psmcfa binning it is compared with, and the
    # control's mask_auto fit is the complete-data fit.
    routes: Tuple[str, ...] = ("mask_auto", "mask_ignore", "fq2psmcfa", "multihetsep", "nomask")
    multihetsep_crosscheck_replicates: int = 1
    grid: TimeGrid = TimeGrid()
    rs_threads: int = 1


# --- E4: how its steps run, beyond the design in E4 above --------------------
#
# The routes from one masked VCF to PSMC (experiments/e4.py, inputprep.py).
# "The exact bins" are psmc-rs's bins of the VCF with its mask (`psmc-rs
# convert`), which both mask routes fit:
#
#   mask_auto    psmc-rs on the VCF, mask mode, --vcf-mask callable.bed.gz,
#                --bin-callable auto (its default): each called bin weighted by
#                its callable bases, counted exactly from the mask
#   mask_ignore  the same, --bin-callable ignore: every called bin taken as
#                bin_size bases, as fq2psmcfa and C psmc take it
#   fq2psmcfa    consensus FASTA (hom: base, het: IUPAC, uncallable: N) streamed
#                into fq2psmcfa -s <bin_size> (its defaults otherwise: -g 10000,
#                and -q, which needs qualities, does nothing), then psmc-rs on
#                the psmcfa, which records no callable bases (C psmc's model
#                whatever --bin-callable says); its bins must equal the exact
#                bins (the bins job fails if not), and its fit mask_ignore's,
#                every fitted value of the JSON (the report fails if not)
#   multihetsep  `psmc-rs convert --multihetsep` of the mask routes' input
#                (nr_called = callable sites since the previous row, inclusive),
#                read back by psmc-rs with --bin-callable auto: multihetsep does
#                not say which sites of an interval are uncalled, so the reader
#                places them at its start, and the bins and their callable bases
#                follow that placement (approximate counts), which auto does not
#                fit (it fits exact counts only): every called bin is taken as
#                full, so the route measures the placement alone; replicate 1 of
#                every mask is also checked row for row against msmc-tools'
#                generate_multihetsep.py, run on each chromosome with its mask
#   nomask       psmc-rs on the same VCF without --vcf-mask (variants only):
#                uncallable sites count as called homozygous, every bin is full
#
# On the no-missing control every bin is full, so the callable-length model is
# C psmc's: the report fails unless mask_auto and mask_ignore fit identically,
# and that fit is the complete-data fit of the replicate.
#
# Masks (inputprep.py): an alternating renewal process per chromosome, of
# uncallable runs with mean length MaskSpec.mean_run_bp and an expected missing
# fraction f. Records at uncallable sites are removed from the VCF.
#
# Seeds (section 4), experiment "E4", model "bottleneck": the simulation's
# ancestry/<chrom> and mutations/<chrom>; mask/<mask>/<chrom> for each mask; and
# reference/<chrom> for the ancestral bases of the consensus FASTA at sites
# without a record (uniform over ACGT, JC69's stationary distribution; they
# do not change any bin).

E4_CONTROL_MASK = "nomissing"  # the no-missing control: every site callable

# psmc-rs --bin-callable of each E4 route, passed explicitly (auto is its
# default; see the routes above for what it does on each).
E4_BIN_CALLABLE: Dict[str, str] = {
    "mask_auto": "auto",
    "mask_ignore": "ignore",
    "fq2psmcfa": "auto",
    "multihetsep": "auto",
    "nomask": "auto",
}


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
    # psmc-rs convert: one pass over the input, 0.15 s and 30 MB per 50 Mb
    convert_s_per_mb: float = 0.005
    convert_gb: float = 0.3
    # the bins of three routes compared with the exact bins: 0.14 s per 50 Mb
    bins_s_per_mb: float = 0.005
    bins_gb: float = 0.5
    # generate_multihetsep.py, a Python loop over every site: 36 s per 50 Mb
    crosscheck_s_per_mb: float = 0.75
    crosscheck_gb: float = 0.5
    # psmc-rs fit (the VCF routes hold the merged mask intervals)
    fit_gb: float = 0.3


E4_COSTS = E4Costs()


@dataclass(frozen=True)
class E5:
    """Bootstrap calibration (the supplement's Fig S3, fig6_bootstrap_calibration).

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
    # a few CPU hours each share the cores without idle threads at their ends
    # (results do not depend on it).
    bootstrap_threads: int = 1
    example_dataset: int = 1  # the dataset whose bands fig6_bootstrap_calibration shows
    # Pointwise coverage over 10 datasets moves in steps of 0.1 (Clopper-Pearson
    # 95%: 0.55-1.0 at 9/10), and the coverage averaged over the window's
    # strongly correlated points has no interval of its own. The summary
    # therefore also resamples the datasets (with replacement, this many
    # times, seed stage "summary/<window>/trunk<B>") and gives 95% percentile
    # intervals of the window-averaged coverage and of the median SD ratio.
    dataset_resamples: int = 10_000


# --- E6: supplementary analyses S1, S2, S3 and S5 -----------------------------
# Every setting of E6 is in this block (experiments/e6.py reads nothing else
# of its own). E6 reuses what other experiments made: E1's genomes, fits and
# settings (S1-S3: models, inference, scaling, threads and time grid come from
# profile.e1) and E2's genomes and bootstrap trunk (S5); S4 is E3's own
# tables. Its costs only schedule the jobs; no result depends on them.


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
    table_s: float = 20.0  # S1, S2 and S5 tables
    table_gb: float = 1.0
    qc_s: float = 5.0  # psmc-rs qc of a real genome (0.1 s for 1.9 Gb)
    qc_gb: float = 0.5


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
    S4  Thread scaling and determinism: E3's own tables (E3_thread_scaling,
        E3_determinism); E6 adds nothing to them.
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

    # S5. psmc-rs qc of E2's genomes, with E2's bin size and bootstrap trunk
    # (--bootstrap-block-size), as E2 bootstraps them.
    qc_experiment: str = "E2"

    costs: E6Costs = E6Costs()


# --- E7: 1000 Genomes ----------------------------------------------------------
# Every setting of E7 is in this block: the sources, samples and processing of
# its data stage (kg1000.py, run outside the runner: it is network-bound) and
# the fits, comparisons and figures of its runner jobs (experiments/e7.py).
#
# Data: the joint-called NYGC 30x VCFs of the 3,202 samples (GATK
# HaplotypeCaller, GenotypeGVCFs, VQSR; Byrska-Bishop et al. 2022, Cell
# 185:3426), chr1-22, 1.68 TB in all: streamed by region over HTTPS, never
# stored. Samples: the 2,504 unrelated samples of the phase 3 panel (the
# sample column, 10, of the sequence index). Mask: the GRCh38 strict
# accessibility mask (2016-06-22) for every sample; non-PASS records are
# missing at their position, as psmc-rs reads them.

_KG_EBI = "https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/1000G_2504_high_coverage"
_KG_S3 = "https://1000genomes.s3.amazonaws.com/1000G_2504_high_coverage"
_KG_JOINT = ("working/20201028_3202_raw_GT_with_annot/"
             "20201028_CCDG_14151_B01_GRM_WGS_2020-08-05_{chrom}.recalibrated_variants.vcf.gz")
_KG_PANEL = ("working/20220422_3202_phased_SNV_INDEL_SV/"
             "1kGP_high_coverage_Illumina.{chrom}.filtered.SNV_INDEL_SV_phased_panel.vcf.gz")
_KG_MASK = ("https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/1000_genomes_project/working/"
            "20160622_genome_mask_GRCh38/StrictMask/20160622.allChr.mask.bed")
_GIAB = ("https://ftp-trace.ncbi.nlm.nih.gov/ReferenceSamples/giab/release/NA12878_HG001/NISTv4.2.1/GRCh38/"
         "HG001_GRCh38_1_22_v4.2.1_benchmark")


@dataclass(frozen=True)
class E7Costs:
    """Expected run time and peak memory of E7's runner jobs, for scheduling and ETA only.

    Fits use :class:`Costs`' rates on the bins of the stage's chunks (all
    of chr1-22: 2,875 Mb). The GIAB comparison reads two VCFs of a few
    million records in Python; the report reads every fit.
    """

    giab_s: float = 300.0
    giab_gb: float = 2.0
    report_s_base: float = 60.0
    report_s_per_fit: float = 0.05
    report_gb: float = 4.0
    vcf_fit_gb: float = 0.5  # psmc-rs on a VCF holds the merged mask intervals (4 million rows of the strict mask)


@dataclass(frozen=True)
class E7:
    """1000 Genomes (the article's Fig 3, fig7_1000_genomes; figS5_1000g_populations, figS6_1000g_checks):
    psmc-rs on the 30x NYGC genomes.

    The data stage (``python -m experiment.pipeline.kg1000``, outside the
    runner) reads the joint-called VCFs region by region (streamed, or from
    a chromosome downloaded to node-local disk), splits them into
    one het-or-missing VCF per sample and writes one psmcfa per sample of the
    cohort, as ``psmc-rs convert --vcf-sample`` would from the joint VCF
    (byte for byte; its pilot checks it), with the strict mask and every
    non-PASS position missing (``runs/E7/inputs``). The runner then fits:
    every sample's psmcfa with psmc-rs (``fit-rs-psmcfa``); the default set,
    ``per_population`` samples of every population (the ``pinned`` ones
    first, the others drawn by seed), also from its VCF with the mask and
    psmc-rs's callable bases per bin (``fit-rs-vcf``, the curves of Fig 7)
    and by C psmc from the psmcfa (``fit-c``); the ``bootstrap_samples``
    with psmc-rs's bootstrap (``boot-rs``); the first default sample of
    every population with the first time window split (``fit-rs-split``,
    ``split_pattern``; Hilgers et al. 2025, Curr Biol 35:927); and NA12878
    against GIAB HG001 v4.2.1 (``giab``). Curves are scaled with ``scaling``.

    Seeds: the default set's draw ranks each population's candidates by
    ``seed("E7", population, 0, "select/<sample>")``; the bootstraps
    ``seed("E7", sample, 0, "bootstrap")``.
    """

    # -- Sources (the data stage) ------------------------------------------
    # The joint VCF of a chromosome ({chrom}) on EBI and on its S3 mirror (the
    # same files, sizes checked); use_mirror streams from S3.
    vcf_url: str = f"{_KG_EBI}/{_KG_JOINT}"
    mirror_url: str = f"{_KG_S3}/{_KG_JOINT}"
    use_mirror: bool = True
    manifest_url: str = f"{_KG_EBI}/working/20201028_3202_raw_GT_with_annot/20201028_genotyped-manifest.tsv"
    # The fallback input (D4), only if streaming the joint VCFs is not
    # possible: the phased panel (EBI only), filtered to MAC >= 2 and without
    # missing genotypes, which lacks about 3.7% of each genome's PASS
    # heterozygous SNVs; a figure made from it must say so.
    input: str = "joint"  # "joint" or "panel"
    panel_url: str = f"{_KG_EBI}/{_KG_PANEL}"
    panel_manifest_url: str = f"{_KG_EBI}/working/20220422_3202_phased_SNV_INDEL_SV/20220804_manifest.txt"
    mask_url: str = _KG_MASK
    index_url: str = f"{_KG_EBI}/1000G_2504_high_coverage.sequence.index"
    ped_url: str = f"{_KG_EBI}/20130606_g1k_3202_samples_ped_population.txt"
    giab_vcf_url: str = f"{_GIAB}.vcf.gz"
    giab_bed_url: str = f"{_GIAB}.bed"
    # A directory of local files named as the URLs' last parts, read instead of
    # the network (relative: under <runs>/E7); the tiny profile's synthetic
    # cohort (kg1000 fixture). None: the URLs.
    local_source: Optional[str] = None

    # -- What the data stage makes ---------------------------------------------
    chromosomes: Tuple[str, ...] = tuple(f"chr{i}" for i in range(1, 23))
    # Smoke profile only: the stage reads only these bases of every chromosome
    # (1-based, inclusive), and the callable mask is cut to them. None: whole
    # chromosomes.
    window: Optional[Tuple[int, int]] = None
    chunk_bp: int = 15_000_000  # one streamed region (a SLURM array task): about 190 chunks for chr1-22
    # "unrelated2504": a psmcfa for every unrelated sample of the sequence
    # index; None: the default set only. One stream costs the same either way.
    cohort: Optional[str] = "unrelated2504"
    per_population: int = 2
    pinned: Tuple[str, ...] = ("NA18507", "NA19239", "NA12878")  # Li & Durbin 2011's samples among the 2,504
    admixed: Tuple[str, ...] = ("ACB", "ASW", "CLM", "MXL", "PEL", "PUR")
    superpopulations: Tuple[str, ...] = ("AFR", "AMR", "EAS", "EUR", "SAS")  # Fig 7 panels a-e
    split_batch_size: int = 500  # samples per bcftools +split call (one open file per sample)
    stream_threads: int = 4  # bcftools view --threads of the stream
    chunk_attempts: int = 3  # a broken stream is retried this many times in all
    # fetch-chrom: byte ranges downloaded at once (a slow cluster link: each
    # connection is slow, several at once are not; kg1000.fetch_ranges)
    fetch_parts: int = 16
    retry_wait_seconds: float = 60.0
    # psmc-rs convert's --min-good-fraction; None: its default (0.2). A window
    # leaves most of a chromosome uncallable, which the default would drop.
    min_good_fraction: Optional[float] = None
    raw_depth_genome_bp: float = 3.1e9  # raw depth = the sample's BASE_COUNT in the index / this
    # The pilot (kg1000 pilot): these samples' psmcfa of one chromosome made by
    # the split route must equal psmc-rs convert --vcf-sample on the stream.
    pilot_samples: Tuple[str, ...] = ("NA18507", "NA19239", "NA12878")

    # -- Fits (the runner) -------------------------------------------------------
    scaling: Scaling = _HUMAN_SCALING  # E2's, whose NA12878 is the same psmcfa
    scaling_source: str = _HUMAN_SCALING_SOURCE
    inference: Inference = INFERENCE
    fit_threads: int = 1
    # E2's bootstrap of NA12878 is this one (RealGenome.bootstrap_shared_with):
    # E2 plans it only with these replicates, trunks and inference.
    bootstrap_samples: Tuple[str, ...] = ("NA18507", "NA19239", "NA12878")
    bootstrap_replicates: int = 100
    bootstrap_trunk_bins: int = 500_000  # psmc-rs's default (splitfa's 50 Mb): about 60 trunks of chr1-22
    bootstrap_threads: int = 32
    band: Tuple[float, float] = (0.025, 0.975)  # the bootstrap's band
    split_pattern: str = "1+1+1+1+25*2+4+6"  # the first window of 4+25*2+4+6 split into its 4 intervals
    c_psmc: bool = True  # fit-c on the default set
    giab: bool = True  # the giab job, when NA12878 is in the default set
    grid: TimeGrid = TimeGrid()
    # Population summaries (E7_pop_bands): quantiles across individuals on a
    # dense log grid of years; the spread between individuals, not a CI.
    spread: Tuple[float, float] = (0.1, 0.9)
    display_years: Tuple[float, float, int] = (1e3, 1e8, 400)
    figure_years: Tuple[float, float] = (1e4, 1e7)  # the time axis of Figs 7 and S5
    costs: E7Costs = E7Costs()


# ---------------------------------------------------------------------------
# 6. Figures (experiment/pipeline/figures): what the figures show. Their
#    style (type, sizes, colours) is in figures/style.py; the tables they
#    read are described in figures/schemas.py. No result depends on these.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Figures:
    """Choices of the figures that are not style: grids, axis ranges, examples."""

    # Replicate curves are summarised (medians, percentile bands) at this many
    # log-spaced times across the axis.
    curve_grid_points: int = 600
    # Time axis (years) of the Ne(t) panels of fig2_simulated_accuracy and
    # figS2_format_consistency (simulations): the fitted curves of every
    # model start after 1 ky and the informative windows end before 2.5 My
    # (see TimeGrid).
    sim_years: Tuple[float, float] = (1e3, 1e7)
    # fig5_input_preparation (the supplement's Fig S2): the mask of panel a,
    # whose genome is the replicate of median RMSE (fig5.example_replicate).
    e4_example_mask: str = "dispersed30"
    # Runner jobs (scheduling only): one figure takes seconds and well under 1 GB.
    job_seconds: float = 20.0
    job_gb: float = 1.0


FIGURES = Figures()


# ---------------------------------------------------------------------------
# 7. Runner: resource budget and cost model
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
    # psmc-rs 0.2, one thread: 9.1 s and 11 MB for 50 Mb x 25 (63 s, 32 MB for
    # 500 Mb). Its M-step runs on one thread and converges in under a
    # millisecond per EM iteration whatever the size (ResourceModel);
    # rs02_mstep_s_per_iter is generous.
    rs02_s_per_mb_iter: float = 0.007
    rs02_mstep_s_per_iter: float = 0.02
    rs02_gb: float = 0.2
    # Tables and figures
    report_s: float = 30.0
    report_gb: float = 1.0

    def rs02_seconds(self, mb: float, inference: Inference, threads: int = 1) -> float:
        """One psmc-rs 0.2 fit: E-steps over ``mb`` Mb on ``threads`` threads, M-steps on one."""
        e_step = self.rs02_s_per_mb_iter * mb * inference.n_iter / max(threads, 1)
        return e_step + self.rs02_mstep_s_per_iter * inference.n_iter


# ---------------------------------------------------------------------------
# 8. Profiles
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
    e7: E7 = E7()
    budget: Budget = Budget()
    costs: Costs = Costs()


PRODUCTION = Profile(name="production")

# smoke: the same code paths at a size that runs in a few minutes, 3 EM
# iterations everywhere. E2 fits the first 20 Mb of each genome (job
# E2/subset/<sample>; the real genomes hold 1.85 to 2.88 Gb, and C psmc needs
# about 50 minutes for each turtle and 70 for NA12878) on 2 Mb trunks, 10 per
# genome, so each genome also gets the second bootstrap. E5's trunks make 8
# and 20 per genome, as production's make 6 and 60. E7 streams a 1 Mb window
# of chr21 in two chunks, the default set only (52 samples), and bootstraps on
# trunks of 1,000 bins (the window leaves the rest of chr21 uncallable, so the
# default trunks would mostly hold no data).
_SMOKE_GENOME = Genome(n_chromosomes=2, chromosome_length=2_000_000)
_SMOKE_INFERENCE = replace(INFERENCE, n_iter=3)
SMOKE = Profile(
    name="smoke",
    e1=E1(models=("constant", "bottleneck"), replicates=2, genome=_SMOKE_GENOME, inference=_SMOKE_INFERENCE),
    e2=E2(bootstrap_replicates=4, inference=_SMOKE_INFERENCE, rs_threads=2, max_bins=200_000,
          bootstrap_trunk_bins=20_000, fine_trunk_bins=5_000),
    e3=E3(genome_sizes_mb=(2, 4), chromosome_length=2_000_000, repeats=1, inference=_SMOKE_INFERENCE,
          rs02_threads=(1, 2), thread_scaling_size_mb=4,
          thread_scaling_threads=(1, 2), timing=QUICK_TIMING),
    e4=E4(replicates=1, genome=_SMOKE_GENOME, inference=_SMOKE_INFERENCE),
    e5=E5(genome=_SMOKE_GENOME, inference=_SMOKE_INFERENCE, truth_replicates=8, bootstrap_datasets=3,
          bootstrap_replicates=8, trunk_bins=(5_000, 2_000)),
    e7=E7(chromosomes=("chr21",), window=(30_000_001, 31_000_000), chunk_bp=500_000, cohort=None,
          min_good_fraction=0.0, inference=_SMOKE_INFERENCE, bootstrap_replicates=4, bootstrap_trunk_bins=1_000,
          bootstrap_threads=2),
)

# tiny: for the unit tests (seconds per experiment), 2 EM iterations. E1 and
# E3 also fit psmc-rs with M-steps capped at 5 iterations, to exercise that
# arm. E4 has two chromosomes (the multihetsep cross-check runs per
# chromosome), the control and one mask of each structure, its clustered runs
# short enough that a chromosome holds several. E7 reads a synthetic cohort
# from local files (kg1000 fixture: 24 msprime genomes of 7 populations on a
# 400 kb "chr21" and a 300 kb "chr22", with missing genotypes, non-PASS
# records, indels and a GIAB-like truth).
_TINY_GENOME = Genome(n_chromosomes=1, chromosome_length=300_000)
_TINY_INFERENCE = replace(INFERENCE, n_iter=2)
TINY = Profile(
    name="tiny",
    e1=E1(models=("bottleneck",), replicates=1, genome=_TINY_GENOME, inference=_TINY_INFERENCE,
          capped_mstep_iters=5),
    e2=E2(genomes=tuple(g for g in REAL_GENOMES if g.sample_id in ("NA12878", "HLhydTec1")), inference=_TINY_INFERENCE,
          bootstrap_replicates=2, bootstrap_trunk_bins=5_000, fine_trunk_bins=1_000, rs_threads=2, max_bins=20_000),
    e3=E3(genome_sizes_mb=(1,), chromosome_length=500_000, repeats=2, inference=_TINY_INFERENCE,
          rs02_threads=(1, 2), rs02_capped_threads=(1,), capped_mstep_iters=5,
          thread_scaling_size_mb=1, thread_scaling_threads=(1, 2), timing=QUICK_TIMING),
    e4=E4(replicates=1, genome=Genome(n_chromosomes=2, chromosome_length=300_000), inference=_TINY_INFERENCE,
          masks=(MaskSpec("clustered30", 0.3, "clustered", 5_000.0), MaskSpec("dispersed30", 0.3, "dispersed", 20.0))),
    e5=E5(genome=_TINY_GENOME, inference=_TINY_INFERENCE, truth_replicates=3, bootstrap_datasets=1,
          bootstrap_replicates=3, trunk_bins=(1_000, 500)),
    e7=E7(local_source="fixture", chromosomes=("chr21", "chr22"), chunk_bp=150_000, inference=_TINY_INFERENCE,
          bootstrap_replicates=2, bootstrap_trunk_bins=1_000, bootstrap_threads=2, split_batch_size=4,
          chunk_attempts=1, retry_wait_seconds=0.0),
)

PROFILES: Dict[str, Profile] = {p.name: p for p in (PRODUCTION, SMOKE, TINY)}


def get_profile(name: str) -> Profile:
    """The profile called ``name`` (``production``, ``smoke`` or ``tiny``)."""
    try:
        return PROFILES[name]
    except KeyError:
        raise KeyError(f"unknown profile {name!r}; known: {', '.join(PROFILES)}") from None


__all__ = [
    "BOTTLENECK", "Budget", "CONSTANT", "Costs", "DemographicModel", "E1", "E2", "E3", "E4", "E4Costs",
    "E4_BIN_CALLABLE", "E4_CONTROL_MASK", "E4_COSTS", "E5", "E6", "E6Costs", "E7", "E7Costs", "EXPANSION",
    "FIGURES", "Figures", "GENOME_500MB", "Genome", "INFERENCE", "Inference", "MODELS", "MODEL_ORDER", "MaskSpec",
    "PRODUCTION", "PROFILES", "PSMC_RS_SUMMARY_QUANTILES", "Profile", "QUICK_TIMING", "REAL_GENOMES",
    "RealGenome", "ResourceModel", "SEED_NAMESPACE", "SIMULATION", "SIMULATION_SCALING", "SMOKE", "Scaling",
    "Simulation", "SizeChange", "TINY", "TRUTH_TABLE_YEARS", "TimeGrid",
    "TimingProtocol", "ZIGZAG", "ZIGZAG_STDPOPSIM", "get_profile", "ms_model", "seed",
]
