"""Tests for the figures' style, table schemas, readers and stand-in tables (experiment.pipeline.figures)."""

from __future__ import annotations

import csv
import math
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fixtures import ROOT, has_font, has_modules  # noqa: E402

from experiment.pipeline import config, demography  # noqa: E402
from experiment.pipeline.results import Curve  # noqa: E402

HAS_PLOTTING = has_modules("numpy", "pandas", "matplotlib")
if HAS_PLOTTING:  # the figures package, schemas too, needs matplotlib on import (style's colours)
    import numpy as np
    import pandas as pd

    from experiment.pipeline.figures import data, fig1, fig4, fig5, fig6, figs4, style
    from experiment.pipeline.figures.schemas import SCHEMAS


@unittest.skipUnless(HAS_PLOTTING, "numpy, pandas and matplotlib are needed")
class Style(unittest.TestCase):
    def test_text_sizes_and_fonts(self):
        rc = style.rc_params()
        self.assertEqual(rc["font.size"], 7.0)
        self.assertEqual(rc["xtick.labelsize"], 6.0)
        self.assertEqual(rc["font.sans-serif"][:2], ["Arial", "Helvetica"])
        self.assertEqual(rc["pdf.fonttype"], 42)
        self.assertEqual(rc["svg.fonttype"], "none")
        self.assertFalse(rc["axes.spines.top"])
        self.assertFalse(rc["axes.spines.right"])

    def test_neutral_chrome_and_line_weights(self):
        from matplotlib.colors import to_rgb

        greys = {"INK": "#1a1a1a", "AXIS": "#404040", "INK2": "#595959", "REF": "#8c8c8c", "CONTEXT": "#c4c4c4",
                 "FILL": "#e8e8e8", "GRID": "#e6e6e6", "UNDERLAY_GREY": "#d9d9d9"}
        for name, value in greys.items():
            self.assertEqual(getattr(style, name), value, name)
            r, g, b = to_rgb(value)
            self.assertTrue(r == g == b, name)  # neutral: colour belongs to the data
        self.assertEqual((style.LW_HAIR, style.LW_THIN, style.LW_DATA, style.LW_EMPH, style.UNDERLAY_LW),
                         (0.35, 0.5, 0.75, 1.0, 2.4))
        self.assertEqual((style.DASH, style.DOT), ((0, (2.5, 1.5)), (0, (0.8, 1.2))))
        rc = style.rc_params()
        # Dark tick labels, thinner darker chrome, no grid by default.
        self.assertEqual((rc["xtick.labelcolor"], rc["ytick.labelcolor"]), (style.INK, style.INK))
        self.assertEqual((rc["axes.edgecolor"], rc["xtick.color"]), (style.AXIS, style.AXIS))
        self.assertEqual((rc["axes.linewidth"], rc["xtick.major.width"], rc["xtick.minor.width"]), (0.5, 0.5, 0.35))
        self.assertEqual((rc["xtick.major.size"], rc["xtick.minor.size"], rc["xtick.major.pad"]), (2.0, 1.2, 1.5))
        self.assertEqual(rc["axes.labelpad"], 2.0)
        self.assertFalse(rc["axes.grid"])
        self.assertEqual((rc["grid.color"], rc["grid.linewidth"]), (style.GRID, style.LW_HAIR))
        self.assertEqual((rc["legend.handlelength"], rc["legend.columnspacing"]), (1.8, 1.5))

    def test_entity_colours_are_fixed(self):
        self.assertEqual(style.RS02.color, "#2a78d6")
        self.assertEqual(style.RS02.label, "psmc-rs")
        self.assertEqual(style.C_PSMC.color, "#eb6834")
        self.assertEqual(style.TRUTH.color, "#1a1a1a")
        self.assertEqual(style.TRUTH.linestyle, style.DASH)
        self.assertEqual(style.C_HALO, "#f6bba4")
        self.assertEqual(style.RS_BAND, "#dde9f8")
        # Routes and formats reuse the same colour for the same thing.
        self.assertEqual(style.ROUTES["multihetsep"].color, style.FORMATS["multihetsep"].color)
        # psmc-rs's default (the callable-length model) is psmc-rs's blue; every called bin taken
        # as full is C psmc's orange through fq2psmcfa and psmc-rs's darker blue from the VCF.
        self.assertEqual(style.ROUTES["mask_auto"].color, style.RS02.color)
        self.assertEqual(style.ROUTES["mask_ignore"].color, style.BLUE_650)
        self.assertEqual(style.ROUTES["fq2psmcfa"].color, style.C_PSMC.color)
        # Colour alone carries a route: no dashes, one marker shape, no route below 3:1 on white.
        for name, ent in style.ROUTES.items():
            self.assertEqual((ent.linestyle, ent.marker, ent.filled), ("-", "o", True), name)
        self.assertEqual((style.ROUTES["multihetsep"].color, style.ROUTES["nomask"].color), ("#b07800", "#c94f80"))
        self.assertEqual(style.SUPERPOP_COLORS["SAS"], "#a0522d")
        self.assertEqual(set(style.SUPERPOP_COLORS), set(style.SUPERPOP_TEXT))

    def test_underlay_rule_truth_and_bands(self):
        under = style.underlay_kw(style.C_PSMC)
        over = style.overlay_kw(style.RS02)
        self.assertEqual((under["color"], under["linewidth"]), (style.C_HALO, style.UNDERLAY_LW))
        self.assertEqual((over["color"], over["linewidth"]), (style.BLUE, style.LW_DATA))
        self.assertLess(under["zorder"], over["zorder"])
        self.assertEqual(under["linestyle"], over["linestyle"])  # never dashes over solid
        # A reference variant of any other colour is its 30% tint.
        self.assertEqual(style.underlay_kw(style.BLUE_650)["color"], style.tint(style.BLUE_650, 0.30))
        truth = style.truth_kw()
        self.assertEqual((truth["color"], truth["linewidth"], truth["linestyle"]), (style.INK, 0.75, style.DASH))
        self.assertGreater(truth["zorder"], over["zorder"])
        band = style.band_kw()
        self.assertEqual(band["color"], style.RS_BAND)
        self.assertNotIn("alpha", band)  # opaque tints, never alpha
        self.assertLess(band["zorder"], under["zorder"])
        # Legend keys are drawn exactly as the lines are.
        key = style.handle(style.C_PSMC, kind="underlay")
        self.assertEqual((key.get_color(), key.get_linewidth()), (style.C_HALO, style.UNDERLAY_LW))
        self.assertEqual(style.handle(style.RS02).get_linewidth(), style.LW_DATA)
        pts = style.RS02.points()
        self.assertEqual((pts["markersize"], pts["markeredgewidth"]), (style.MARKER_MS, 0.0))
        self.assertEqual(style.C_PSMC.marker, "s")

    def test_thousands_log_axes(self):
        fig = style.figure("double", 50)
        gs = style.grid(fig, 1, 3)
        a, b, c = (fig.add_subplot(gs[0, i]) for i in range(3))
        style.years_axis(a, (3e3, 3e6))
        style.ne_axis(a, (500.0, 1e5))
        style.ne_axis(b, (1.5e3, 6e4))  # under two powers of ten: 1, 2 and 5
        style.log_axis(c, "x", fmt="plain", lim=(8.0, 400.0))
        fig.canvas.draw()

        def labels(axis):
            lo, hi = sorted(axis.get_view_interval())
            return [t.get_text() for t, v in zip(axis.get_ticklabels(), axis.get_ticklocs()) if lo <= v <= hi]

        self.assertEqual(labels(a.xaxis), ["10", "100", "1,000"])
        self.assertEqual(labels(a.yaxis), ["1", "10", "100"])
        self.assertEqual(labels(b.yaxis), ["2", "5", "10", "20", "50"])
        self.assertEqual(labels(c.xaxis), ["10", "100"])
        self.assertEqual((a.get_xlabel(), a.get_ylabel()), (style.YEARS_LABEL, style.NE_LABEL))
        self.assertEqual(style.NE_LABEL, r"$\mathit{N}_\mathrm{e}$ (thousands)")
        self.assertEqual(style.YEARS_LABEL, "Thousands of years ago")
        self.assertFalse(any(line.get_visible() for line in a.get_xgridlines() + a.get_ygridlines()))
        self.assertEqual([style.fmt_num(v) for v in (0.5, 20.0, 1e4, -0.05)], ["0.5", "20", "10,000", "−0.05"])
        self.assertTrue(style.narrow_log_span(1e3, 2e4))
        self.assertFalse(style.narrow_log_span(3e3, 3e6))

    @unittest.skipUnless(has_font("Arial"), "the margins are measured in Arial, the house font")
    def test_layout_constants(self):
        self.assertEqual((style.MARGIN_MM, style.LEFT_MM, style.RIGHT_MM), (1.0, 11.0, 1.5))
        self.assertEqual((style.GUTTER_MM, style.GUTTER_SHARED_MM), (6.0, 3.0))
        self.assertEqual((style.ROW_GAP_MM, style.ROW_GAP_TITLED_MM), (9.0, 12.0))
        self.assertEqual((style.LETTER_X_MM, style.TITLE_RISE_MM), (1.0, 1.5))
        # The house layout keeps every edge of the ink 1 mm (right: 1.5 mm) from the figure's.
        fig = style.figure("double", 60)
        gs = style.grid(fig, 1, 2)
        axes = [fig.add_subplot(gs[0, i]) for i in range(2)]
        for ax in axes:
            ax.plot([2e3, 3e6], [1e3, 1e5])
            style.years_axis(ax, (2e3, 3e6))  # no tick label at the right end: it would overhang
            style.ne_axis(ax, (1e3, 1e5))
            style.panel_title(ax, "Bottleneck")
        style.panel_letters(fig, axes)
        fig.canvas.draw()
        box = fig.get_tightbbox(fig.canvas.get_renderer())
        w, h = fig.get_figwidth() / style.MM, fig.get_figheight() / style.MM
        self.assertAlmostEqual(box.x0 / style.MM, style.MARGIN_MM, delta=0.05)
        self.assertAlmostEqual(h - box.y1 / style.MM, style.MARGIN_MM, delta=0.15)
        self.assertAlmostEqual(box.y0 / style.MM, style.MARGIN_MM, delta=0.15)
        self.assertAlmostEqual(w - box.x1 / style.MM, style.RIGHT_MM, delta=0.05)

    def test_every_e4_route_has_a_style(self):
        from experiment.pipeline.experiments import e4

        self.assertEqual(set(style.ROUTES), set(e4.ROUTES))
        self.assertEqual(set(style.ROUTE_ORDER), set(e4.ROUTES))
        self.assertEqual(set(config.E4_BIN_CALLABLE), set(e4.ROUTES))
        self.assertTrue(set(fig5.DRAWN) <= set(e4.ROUTES))
        # Weighted bins, and bins taken as full as C psmc takes them (fq2psmcfa); mask_ignore fits as fq2psmcfa
        # does (asserted by the report) and is drawn in its place only where a table has no fq2psmcfa rows.
        self.assertEqual(fig5.DRAWN, ("mask_auto", "fq2psmcfa"))
        self.assertEqual(fig5.FULL_FALLBACK, "mask_ignore")

    def test_figure_sizes(self):
        fig = style.figure("double", 100)
        self.assertAlmostEqual(fig.get_figwidth() * 25.4, 183.0)
        self.assertAlmostEqual(fig.get_figheight() * 25.4, 100.0)
        fig = style.figure("single", 50)
        self.assertAlmostEqual(fig.get_figwidth() * 25.4, 89.0)

    def test_step_coordinates(self):
        x, y = style.step_xy([0.0, 10.0, 100.0, 1000.0], [1.0, 2.0, 3.0, 4.0], (50.0, 500.0))
        self.assertEqual(list(x), [50.0, 100.0, 500.0])
        self.assertEqual(list(y), [2.0, 3.0, 3.0])
        x, y = style.step_xy([0.0, 10.0], [1.0, 2.0], (1.0, 5.0))
        self.assertEqual(list(x), [1.0, 5.0])
        self.assertEqual(list(y), [1.0, 1.0])

    def test_save_writes_three_formats_reproducibly(self):
        def draw():
            fig = style.figure("single", 40)
            ax = fig.add_subplot(111)
            ax.plot([1, 2, 3], [1, 4, 9], color=style.BLUE)
            ax.set_xlabel("x")
            style.panel_letters(fig, [ax])
            return fig

        with tempfile.TemporaryDirectory() as d:
            first = [p.read_bytes() for p in style.save(draw(), Path(d) / "a", "f", dpi=72)]
            second = [p.read_bytes() for p in style.save(draw(), Path(d) / "b", "f", dpi=72)]
            self.assertEqual([p.suffix for p in sorted((Path(d) / "a").iterdir())], [".pdf", ".png", ".svg"])
            self.assertFalse(list(Path(d).rglob("*.partial")))
        self.assertEqual(first, second)
        self.assertNotIn(b"CreationDate", first[0])
        self.assertIn(b"<text", first[1])  # text stays text in the SVG

    def test_three_type_sizes(self):
        fig = style.figure("single", 40)
        ax = fig.add_subplot(111)
        note = style.note(ax, 0.5, 0.5, "a note")
        self.assertEqual((note.get_fontsize(), note.get_color()), (6.0, style.INK2))
        label = style.direct_label(ax, 0.5, 0.5, "8.6×")
        self.assertEqual((label.get_fontsize(), label.get_color()), (6.0, style.INK))
        leg = style.legend(ax, [ax.plot([0, 1], [0, 1], label="a line")[0]])
        self.assertEqual(leg.get_texts()[0].get_fontsize(), 7.0)
        title = style.panel_title(ax, "Constant")
        self.assertEqual((title.get_fontsize(), title.get_color()), (7.0, style.INK))
        self.assertEqual((style.PANEL_PT, style.TEXT_PT, style.TICK_PT, style.LABEL_PT, style.NOTE_PT),
                         (8.0, 7.0, 6.0, 6.0, 6.0))
        self.assertIn("three type sizes", " ".join(style.__doc__.split()))

    def test_no_text_below_5pt_superscripts_included(self):
        import matplotlib._mathtext as mt

        fig = style.figure("single", 40)
        self.assertEqual(mt.SHRINK_FACTOR, style.MATH_SHRINK)
        ax = fig.add_subplot(111)
        ax.plot([1e3, 1e6], [1e3, 1e5])
        style.log_axis(ax, "x")
        style.log_axis(ax, "y")
        ax.set_ylabel(style.NE_LABEL)
        style.note(ax, 0.1, 0.9, r"max 1.9$\times$10$^{-5}$")
        with tempfile.TemporaryDirectory() as d:
            svg = next(p for p in style.save(fig, Path(d), "f", dpi=72) if p.suffix == ".svg")
            self.assertEqual(style.small_text(svg), [])
            text = svg.read_text()
            # A 6 pt tick label's exponent at 0.84 of it, not mathtext's 0.7 (4.2 pt).
            self.assertIn("font-size: 5.04px", text)
            self.assertNotIn("font-size: 4.2px", text)
            bad = Path(d) / "bad.svg"
            bad.write_text('<svg><text style="font-size: 6px">10</text>'
                           '<tspan x="1" style="font-size: 4.2px; font-family: Arial">3</tspan></svg>')
            self.assertEqual(style.small_text(bad), [(4.2, "3")])
        self.assertIn("no text is below :data:`MIN_TEXT_PT` (5 pt)", " ".join(style.__doc__.split()))

    def test_tint_is_the_colour_at_replicate_opacity_over_white(self):
        self.assertEqual(style.tint("#000000", 0.25), "#bfbfbf")
        self.assertEqual(style.tint(style.BLUE, 1.0), style.BLUE)
        self.assertEqual(style.tint(style.BLUE, 0.0), style.SURFACE)

    def test_panel_letters_align_columns(self):
        fig = style.figure("double", 80)
        gs = style.grid(fig, 2, 2, left_mm=20.0)
        axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(2)]
        axes[1].plot([0, 1], [0, 100000])  # wide tick labels in the second column's top panel
        for ax in axes:
            ax.set_ylabel("y")
        fw, fh = fig.get_figwidth() / style.MM, fig.get_figheight() / style.MM
        a, b, c, d = style.panel_letters(fig, axes, align_cols=False)
        # First-column letters sit at 1 mm from the figure's edge, whatever the y label's width.
        self.assertAlmostEqual(a.get_position()[0] * fw, style.LETTER_X_MM)
        self.assertAlmostEqual(c.get_position()[0] * fw, style.LETTER_X_MM)
        self.assertNotAlmostEqual(b.get_position()[0], d.get_position()[0], places=3)
        # Every letter's baseline is 1.5 mm above its axes' top.
        for t, ax in zip((a, b, c, d), axes):
            self.assertEqual(t.get_verticalalignment(), "baseline")
            self.assertAlmostEqual((t.get_position()[1] - ax.get_position().y1) * fh, style.TITLE_RISE_MM)
        for t in fig.texts[:]:
            t.remove()
        a, b, c, d = style.panel_letters(fig, axes)  # aligned in columns by default
        self.assertAlmostEqual(b.get_position()[0], d.get_position()[0])
        self.assertLess(b.get_position()[0], axes[1].get_position().x0)

    def test_panel_title_shares_its_letters_baseline(self):
        fig = style.figure("double", 50)
        gs = style.grid(fig, 1, 2, wspace_mm=style.GUTTER_SHARED_MM)
        left, right = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
        right.tick_params(axis="y", labelleft=False)  # a shared, hidden y axis
        for ax in (left, right):
            style.panel_title(ax, "Constant")
        a, b = style.panel_letters(fig, [left, right])
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        for t, ax in ((a, left), (b, right)):
            title = ax._left_title
            self.assertEqual(title.get_text(), "Constant")
            self.assertEqual((title.get_verticalalignment(), t.get_verticalalignment()), ("baseline", "baseline"))
            # Same baseline: both texts are anchored at their baselines, at the same height.
            y_title = title.get_transform().transform(title.get_position())[1]
            y_letter = t.get_transform().transform(t.get_position())[1]
            self.assertAlmostEqual(y_title, y_letter, delta=0.01)
            self.assertAlmostEqual(title.get_window_extent(renderer).x0, ax.get_window_extent(renderer).x0, delta=0.5)
        # Beside a hidden y axis the letter keeps clear of the title instead of sitting on it.
        gap = right._left_title.get_window_extent(renderer).x0 - b.get_window_extent(renderer).x1
        self.assertGreaterEqual(gap / fig.dpi / style.MM, style.LETTER_TITLE_GAP_MM - 0.01)

    def test_top_legend_is_one_line_at_the_first_spine(self):
        fig = style.figure("double", 50)
        gs = style.grid(fig, 1, 2, top_mm=style.TOP_LEGEND_MM)
        axes = [fig.add_subplot(gs[0, i]) for i in range(2)]
        handles = [style.handle(style.TRUTH), style.handle(style.C_PSMC, kind="underlay"), style.handle(style.RS02)]
        leg = style.top_legend(fig, handles)
        fig.canvas.draw()
        box = leg.get_window_extent(fig.canvas.get_renderer())
        self.assertAlmostEqual(box.x0, axes[0].get_window_extent().x0, delta=0.5)
        self.assertAlmostEqual((fig.bbox.y1 - box.y1) / fig.dpi / style.MM, style.MARGIN_MM, delta=0.05)
        self.assertEqual([t.get_text() for t in leg.get_texts()], ["Truth", "C psmc", "psmc-rs"])
        ys = {round(t.get_window_extent().y0, 1) for t in leg.get_texts()}
        self.assertEqual(len(ys), 1)  # one line


@unittest.skipUnless(HAS_PLOTTING, "numpy, pandas and matplotlib are needed")
class Data(unittest.TestCase):
    def test_truth_line_has_exact_change_times(self):
        x, y = data.truth_xy("bottleneck", 25.0, (1e3, 1e7))
        changes = [t * 4 * 20_000 * 25.0 for t, _ in config.BOTTLENECK.ms_events]
        for t in changes:
            self.assertIn(t, list(x))
        eps = demography.epochs(config.BOTTLENECK)
        for xi, yi in zip(x, y):
            if xi not in changes:
                self.assertAlmostEqual(yi, demography.ne_at(eps, xi / 25.0))

    def test_truth_line_samples_growth(self):
        x, y = data.truth_xy("zigzag", 25.0, (1e3, 1e7))
        self.assertGreater(len(x), 100)
        self.assertTrue(all(b >= a for a, b in zip(x, x[1:])))

    def test_truth_table_check(self):
        years = np.logspace(3, 7, 20)
        good = pd.DataFrame({"model": "expansion", "years": years, "ne": data.truth_at("expansion", years, 25.0)})
        data.check_truth_table(good, 25.0)
        bad = good.assign(ne=good["ne"] * 1.01)
        with self.assertRaises(ValueError):
            data.check_truth_table(bad, 25.0)

    def test_median_of_curves(self):
        cs = [Curve((0.0, 10.0), (float(k), float(k))) for k in (1, 100, 1000, 10)]
        self.assertTrue(np.allclose(data.median(cs, np.array([1.0, 20.0])), 10 ** 1.5))  # geometric: of log10 Ne

    def test_windows_of_a_table(self):
        df = pd.DataFrame({"model": ["a", "b"], "window_lo_years": [1, 2], "window_hi_years": [10, 20]})
        self.assertEqual(data.table_windows(df, "model"), {"a": (1.0, 10.0), "b": (2.0, 20.0)})

    def test_curves_from_a_table(self):
        df = pd.DataFrame({"model": ["a", "a", "b", "b"], "interval": [1, 0, 0, 1],
                           "years": [5.0, 0.0, 0.0, 7.0], "ne": [2.0, 1.0, 3.0, 4.0]})
        cs = data.curves(df, ["model"])
        self.assertEqual(cs[("a",)].years, (0.0, 5.0))
        self.assertEqual(cs[("a",)].ne, (1.0, 2.0))

    def test_formatting(self):
        self.assertEqual(data.fmt_bp(50_000_000), "50 Mb")
        self.assertEqual(data.fmt_bp(500_000), "500 kb")
        self.assertEqual(data.fmt_bp(2_500_000), "2.5 Mb")
        self.assertEqual(data.fmt_sig(0.0381), "0.038")
        self.assertIn("10$^{-4}$", data.fmt_sig(9.0e-4))
        self.assertEqual(data.decades(1.5e3, 6e5), (1e3, 1e6))

    def test_decades_never_add_an_empty_decade(self):
        # More than a third of a decade from a power of ten: the next 1, 2 or 5.
        self.assertEqual(data.decades(3e3, 2e5), (2e3, 2e5))
        self.assertEqual(data.decades(769.2, 6.3e4), (500.0, 1e5))  # Fig 5d: not down to 10^2
        self.assertEqual(data.decades(521.5, 1.073e5), (500.0, 2e5))
        self.assertEqual(data.decades(1e-6, 2.032), (1e-6, 5.0))
        self.assertEqual(data.decades(2.449e-6, 0.4902), (2e-6, 1.0))
        # Close to a power of ten, or on one: that power.
        self.assertEqual(data.decades(1.561e4, 3.0e6), (1e4, 5e6))
        self.assertEqual(data.decades(1e3, 1e5), (1e3, 1e5))
        self.assertEqual(data.decades(1e3 * (1 + 1e-12), 1e5 * (1 - 1e-12)), (1e3, 1e5))
        for lo, hi in ((1.1, 9.9), (3.3e-5, 4.4e-2), (7.7e2, 1.2e7)):
            a, b = data.decades(lo, hi)
            self.assertLessEqual(a, lo)
            self.assertGreaterEqual(b, hi)
            self.assertLess(math.log10(lo / a), 0.5)  # under half a decade added at either end
            self.assertLess(math.log10(b / hi), 0.5)


@unittest.skipUnless(HAS_PLOTTING, "numpy, pandas and matplotlib are needed")
class Schemas(unittest.TestCase):
    """The schemas against real tables are checked in test_figures_draw (the experiments' tiny runs)."""

    def setUp(self):
        self.runs = Path(self.enterContext(tempfile.TemporaryDirectory()))

    def test_missing_table_and_columns_are_named(self):
        with self.assertRaises(data.MissingTable):
            data.read(self.runs, "E2_rs_vs_c")
        path = SCHEMAS["E2_rs_vs_c"].path(self.runs)
        path.parent.mkdir(parents=True)
        with open(path, "w", newline="") as f:
            csv.writer(f).writerows([["sample_id", "window_lo_years"], ["NA12878", "1000"]])
        with self.assertRaises(data.SchemaError) as cm:
            data.read(self.runs, "E2_rs_vs_c")
        self.assertIn("rmse_log10_ne_rs_vs_c", str(cm.exception))

    def test_every_schema_names_an_experiments_table(self):
        from experiment.pipeline.experiments import e1, e2, e3, e4, e5, e6, e7

        written = {f"{m.NAME}/{name}" for m in (e1, e2, e3, e4, e5, e7) for name in m.TABLES.values()}
        written |= {f"E6/{name}" for part in e6.TABLES.values() for name in part.values()}
        for key, s in SCHEMAS.items():
            self.assertIn(f"{s.experiment}/{s.file}", written, key)


@unittest.skipUnless(HAS_PLOTTING, "numpy, pandas and matplotlib are needed")
class FigureContent(unittest.TestCase):
    """What Figs 1, 5 and 6 draw, without the experiments' tables."""

    def test_fig1_shows_the_callable_length_model_and_the_fit_feeds_the_outputs(self):
        from matplotlib.colors import to_hex
        from matplotlib.patches import FancyBboxPatch

        fig = fig1.draw()
        ax = fig.axes[0]
        text = "\n".join(t.get_text() for t in ax.texts)
        self.assertIn("--bin-callable", text)
        self.assertIn(r"e_T(k)^{\,c/s}", text)
        self.assertIn("callable", text.split("Route B")[1].split("Inputs made")[0])
        self.assertIn("C psmc's model and EM", text)
        bullets = [t for t in ax.texts if t.get_text() == "•"]
        self.assertEqual(len(bullets), len(fig1.FIT_ITEMS))
        # One line leaves the fit box (the lowest of psmc-rs's boxes) for a bus beside the five outputs.
        boxes = [p for p in ax.patches if isinstance(p, FancyBboxPatch)]
        rs_boxes = [p for p in boxes if to_hex(p.get_facecolor()) == fig1.RS_FILL]
        fit = min(rs_boxes, key=lambda p: p.get_y())
        outputs = [p for p in boxes if p.get_x() > fit.get_x() + fit.get_width() + 1.0]
        self.assertEqual(len(outputs), 5)
        lines = [ln for ln in ax.lines if ln.get_color() == fig1.style.BLUE]
        self.assertEqual(len(lines), 2)
        out, bus = lines
        self.assertAlmostEqual(out.get_xdata()[0], fit.get_x() + fit.get_width())
        self.assertTrue(fit.get_y() < out.get_ydata()[0] < fit.get_y() + fit.get_height())
        self.assertEqual(out.get_xdata()[1], bus.get_xdata()[0])
        centres = [p.get_y() + p.get_height() / 2 for p in outputs]
        self.assertLessEqual(min(bus.get_ydata()), min(centres))
        self.assertGreaterEqual(max(bus.get_ydata()), max(centres))

    def test_fig5_draws_the_bins_taken_as_full_once(self):
        rows = [("dispersed30", r, rep, theta) for r, theta in (("fq2psmcfa", -0.35), ("mask_ignore", -0.35),
                                                                ("mask_auto", -0.02)) for rep in (1, 2)]
        routes = pd.DataFrame(rows, columns=["mask", "route", "replicate", "theta_rel_complete"])
        # fq2psmcfa and the VCF read with --bin-callable ignore fit the same bins: fq2psmcfa is drawn, once.
        self.assertEqual(fig5.full_route(routes), "fq2psmcfa")
        self.assertEqual(fig5.full_route(routes[routes.route != "fq2psmcfa"]), "mask_ignore")
        self.assertEqual(fig5.full_route(routes[routes.route != "mask_ignore"]), "fq2psmcfa")
        # If they do not agree, the figure is not drawn.
        bad = routes.copy()
        bad.loc[(bad.route == "mask_ignore") & (bad.replicate == 2), "theta_rel_complete"] = -0.30
        with self.assertRaises(ValueError):
            fig5.full_route(bad)

    def test_fig5_shows_the_median_genome_and_each_structure_from_the_control(self):
        rmse = {1: 0.30, 2: 0.10, 3: 0.20, 4: 0.40}
        routes = pd.DataFrame([("dispersed30", "mask_auto", r, v) for r, v in rmse.items()],
                              columns=["mask", "route", "replicate", "rmse_log10_ne_vs_complete"])
        # The lower median of four by RMSE against the complete-data fit: 0.10, 0.20, 0.30, 0.40.
        self.assertEqual(fig5.example_replicate(routes, "dispersed30"), 3)
        self.assertIsNone(fig5.example_replicate(routes, "clustered30"))
        cfg = config.PRODUCTION.e4
        for structure in fig5.STRUCTURES:
            masks = fig5.facet_masks(cfg, structure)
            self.assertEqual(masks[0], (fig5.COMPLETE, 0.0))  # every facet starts at the control
            xs = [x for _, x in masks]
            self.assertEqual(xs, sorted(xs))
            self.assertEqual({m for m, _ in masks[1:]}, {m.name for m in cfg.masks if m.structure == structure})
        self.assertEqual(fig5.example_mask(cfg, pd.DataFrame({"mask": [m.name for m in cfg.masks]})),
                         config.FIGURES.e4_example_mask)

    def test_fig6_recomputes_each_genomes_coverage_and_checks_it(self):
        def summary(**kw):  # an E5_summary row, as fig6 reads it
            row = {"trunk_bins": 100, "trunks": 6, "sd_ratio_median": 0.934, "coverage_of_fit_mean_mean": 0.5,
                   "coverage_mean": 0.5}
            return next(pd.DataFrame([{**row, **kw}]).itertuples())

        sampling = pd.DataFrame({"point": [0, 1], "mean_log10_ne": [3.0, 4.0]})  # the fits' mean: 1,000 and 10,000
        bands = pd.DataFrame(
            [(100, 1, 0, 500.0, 1000.0, 1), (100, 1, 1, 2e4, 3e4, 1),  # bounds included: 1,000 is held
             (100, 2, 0, 2e3, 3e3, 0), (100, 2, 1, 5e3, 2e4, 0)],
            columns=["trunk_bins", "dataset", "point", "boot_lo_ne", "boot_hi_ne", "covers_truth"])
        per = fig6.genome_coverage(bands, sampling)
        self.assertEqual(per.loc[(100, 1)].tolist(), [0.5, 1.0])  # (fits' mean, truth)
        self.assertEqual(per.loc[(100, 2)].tolist(), [0.5, 0.0])
        cov = pd.DataFrame({"trunk_bins": [100, 100], "coverage_fit_mean": [0.5, 0.5], "coverage": [0.5, 0.5]})
        fig6._check_coverage(per, cov, {100: summary()}, [100])  # E5_bands, E5_coverage and E5_summary agree
        # A coverage the tables disagree on raises rather than drawing a number the text does not cite.
        with self.assertRaises(ValueError):
            fig6._check_coverage(per, cov.assign(coverage=[0.5, 1.0]), {100: summary()}, [100])
        with self.assertRaises(ValueError):
            fig6._check_coverage(per, cov, {100: summary(coverage_mean=0.6)}, [100])
        # b's key: counts right-aligned with figure spaces, so that the ratios form one column.
        labels = fig6._key_labels_b([10, 100], {10: summary(trunk_bins=10, trunks=60, sd_ratio_median=1.306),
                                                100: summary()})
        self.assertEqual(labels, ["60 blocks" + fig6.RATIO_SEP + "1.31×",
                                  fig6.FIGURE_SPACE + "6 blocks" + fig6.RATIO_SEP + "0.93×"])
        self.assertEqual((fig6.FIGURE_SPACE, fig6.RATIO_SEP), ("\u2007", "\u00a0\u00a0"))


@unittest.skipUnless(HAS_PLOTTING, "numpy, pandas and matplotlib are needed")
class Readme(unittest.TestCase):
    def test_lists_every_table_and_column(self):
        text = (ROOT / "experiment" / "pipeline" / "figures" / "README.md").read_text()
        for s in SCHEMAS.values():
            rows = [line for line in text.splitlines() if line.startswith(f"| `{s.experiment}/tables/{s.file}`")]
            self.assertEqual(len(rows), 1, s.file)
            for col in s.names:
                self.assertIn(f"`{col}`", rows[0], f"{s.file}: {col}")


@unittest.skipUnless(HAS_PLOTTING, "numpy, pandas and matplotlib are needed")
class Performance(unittest.TestCase):
    """Figs 4 and S4 (E3): thread counts and the ceiling of the thread scaling."""

    def test_fig4_draws_each_thread_count_of_a_tool_in_its_own_style(self):
        cfg = config.PRODUCTION.e3
        # Table 2's rows: every size at 1, 8 and 30 threads, the thread-scaling size at 2, 4 and 16 too.
        rows = [("c", 1)] + [("rs02", t) for t in (1, 8, 30)] + [("rs02", t) for t in (2, 4, 16)]
        table2 = pd.DataFrame(rows, columns=["tool", "threads"])
        series = fig4.series(table2, fig4.tool_threads(cfg))
        self.assertEqual([(tool, t) for tool, t, _ in series], rows[:4])  # not the thread-scaling counts
        ents = [e for name, _, e in series if name == "rs02"]
        self.assertEqual(len({str(e.linestyle) for e in ents}), 3)
        self.assertEqual([e.filled for e in ents], [True, False, False])
        self.assertEqual(ents[0].linestyle, "-")
        self.assertEqual(len(fig4.series(table2)), 7)  # without the profile's counts: every one in the table

    def test_fig4_gives_each_size_its_chromosomes(self):
        table2 = pd.DataFrame({"size_mb": [100, 100, 300, 3000], "chromosomes": [1, 1, 3, 30]})
        self.assertEqual(fig4.chromosome_counts(table2), {100: 1, 300: 3, 3000: 30})
        with self.assertRaises(ValueError):
            fig4.chromosome_counts(pd.DataFrame({"size_mb": [100, 100], "chromosomes": [1, 2]}))
        self.assertIn("ceil(n / t)", " ".join(fig4.__doc__.split()))

    def test_figs4_names_the_runs_behind_more_than_one_output(self):
        rows = []
        for size in (100, 300, 1000):
            for t in (1, 8, 30):
                for r in (1, 2):
                    sha = "same" if size == 100 or t > 1 else f"one-{size}"
                    rows.append(("rs02", size, t, r, sha, 2.5e-5 if sha.startswith("one") else 0.0))
            rows.append(("c", size, 1, 1, f"c-{size}", 0.0))
        det = pd.DataFrame(rows, columns=["tool", "size_mb", "threads", "repeat", "sha256", "max_rel_diff_params"])
        (n, text), = figs4.output_differences(det)
        nb = figs4.NBSP
        self.assertEqual(n, 2)
        self.assertTrue(text.startswith(f"psmc-rs 0.2 at 300{nb}Mb and above: 1{nb}thread wrote different bytes "
                                        f"from 8 and 30{nb}threads"), text)
        self.assertIn("2.5$\\times$10$^{-5}$ relative", text)
        self.assertIn("repeats at a thread count are identical", text)
        # Repeats that disagree are named as such.
        det.loc[(det.size_mb == 1000) & (det.threads == 8) & (det.repeat == 2), "sha256"] = "odd"
        (n, text), = figs4.output_differences(det)
        self.assertIn("runs differ", text)
        self.assertEqual(figs4.output_differences(det[det.tool == "c"]), [])

    def test_figs4_ceiling_of_whole_sequences_per_thread(self):
        speedup = figs4.ceiling_speedup([1, 2, 4, 8, 16, 30, 60], 30)
        self.assertEqual([round(float(x), 6) for x in speedup], [1.0, 2.0, 3.75, 7.5, 15.0, 30.0, 30.0])
        self.assertEqual(float(figs4.ceiling_speedup(1, 30)), 1.0)


class FiguresConfig(unittest.TestCase):
    def test_settings(self):
        f = config.FIGURES
        self.assertLess(f.sim_years[0], f.sim_years[1])
        self.assertIn(f.e4_example_mask, {m.name for m in config.PRODUCTION.e4.masks})
        self.assertTrue(set(f.e4_bias_masks) <= {m.name for m in config.PRODUCTION.e4.masks})
        self.assertLessEqual(config.PRODUCTION.e5.example_dataset, config.PRODUCTION.e5.bootstrap_datasets)
        for name in config.MODEL_ORDER:
            lo = demography.coalescence_quantile(demography.epochs(config.MODELS[name]), 0.01) * 25.0
            hi = demography.coalescence_quantile(demography.epochs(config.MODELS[name]), 0.99) * 25.0
            self.assertTrue(f.sim_years[0] < lo and hi < f.sim_years[1], name)
            self.assertTrue(math.isfinite(lo))


if __name__ == "__main__":
    unittest.main()
