"""Tests for experiment.pipeline.figures.e4 (Fig 5 from E4's tables)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment.pipeline.figures import e4 as fig5  # noqa: E402

SUMMARY = [
    {"mask": m, "route": r, "structure": s, "missing_fraction_target": f}
    for m, s, f in (("nomissing", "none", "0.0"), ("clustered10", "clustered", "0.1"),
                    ("clustered30", "clustered", "0.3"), ("dispersed10", "dispersed", "0.1"))
    for r in ("exact", "nomask")
]


class Helpers(unittest.TestCase):
    def test_masks_keep_table_order_once_each(self):
        self.assertEqual(fig5.mask_order(SUMMARY), ["nomissing", "clustered10", "clustered30", "dispersed10"])

    def test_curve_panels_show_the_largest_missing_fraction_of_each_structure(self):
        info = fig5.mask_info(SUMMARY)
        self.assertEqual(fig5.worst_masks(info), {"clustered": "clustered30", "dispersed": "dispersed10"})
        self.assertEqual(fig5.worst_masks({"nomissing": ("none", 0.0)}), {"clustered": None, "dispersed": None})

    def test_tick_labels(self):
        self.assertEqual(fig5.mask_label("none", 0.0), "none")
        self.assertEqual(fig5.mask_label("dispersed", 0.3), "30%")

    def test_every_drawn_route_has_a_colour_and_a_marker(self):
        markers = {s["marker"] for s in fig5.ROUTE_STYLE.values()}
        colours = {s["color"] for s in fig5.ROUTE_STYLE.values()}
        self.assertEqual(len(markers), len(fig5.ROUTE_STYLE))
        self.assertEqual(len(colours), len(fig5.ROUTE_STYLE))


if __name__ == "__main__":
    unittest.main()
