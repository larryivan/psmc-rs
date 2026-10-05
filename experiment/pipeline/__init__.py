"""The experiment pipeline behind the psmc-rs manuscript.

Run it from the repository root with the experiment's Python environment::

    python -m experiment.pipeline --help

Modules:

* :mod:`.config`      every parameter of the experiments, the seed scheme and
                      the run profiles (one file);
* :mod:`.paths`       where the repository, the tools and the runs are;
* :mod:`.demography`  the demographic models as ``msprime.Demography`` and
                      their true Ne(t);
* :mod:`.simulate`    one simulated diploid genome: psmcfa, VCF, truth, TMRCA;
* :mod:`.tools`       psmc-rs 0.2 and 0.1 and C psmc: build, locate, identify,
                      and the exact command lines;
* :mod:`.results`     C ``.psmc`` and psmc-rs JSON outputs as Ne(t) curves;
* :mod:`.metrics`     time grids and RMSE of log10 Ne;
* :mod:`.provenance`  checksums and the provenance record of a job;
* :mod:`.runner`      the resumable, detached job runner;
* :mod:`.experiments` the experiments as runner jobs (E1, ...) and their tables.
"""

__all__ = ["__version__"]

__version__ = "1.0"
