"""The manuscript's figures, drawn from the experiments' tables (``runs/<EXP>/tables``).

Each figure is a module runnable as ``python -m experiment.pipeline.figures.<name>``
and a runner job of its experiment; it reads CSV tables only, so a figure can
be redrawn without re-running anything.
"""
