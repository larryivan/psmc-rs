"""Every variant against C psmc's reference outputs, round by round: ref_variants.json.

Usage: compare_ref.py   (reads ref/, written by run_ref.sh, and C psmc's outputs in tests/data)
For each run: the relative difference from C psmc of the log-likelihood (LK, which C prints to 6 decimals) in
every round, its largest over rounds 2 to 5, of the parameters (PA, 9 decimals), of the RI line, and the M-step's
evaluations (IT) of both programs.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from psmcfile import rounds  # noqa: E402

C = Path(__file__).resolve().parent.parent
REPO = C.parents[1]
CASES = {
    "p23": REPO / "tests/data/c_reference/c_4+5x3+4_N5.psmc",
    "p64": REPO / "tests/data/c_reference/c_4+25x2+4+6_N5.psmc",
    "f23": REPO / "tests/data/psmc_out/c_23x1_N5.psmc",
    "frag": REPO / "tests/data/psmc_out/c_4+5x3+4_fragmented_N1.psmc",
}
VARIANTS = ["none", "lastbin", "hj", "both", "prec"]


def rel(a, b):
    return abs(a - b) / abs(b)


res = {}
for case, cfile in CASES.items():
    c = rounds(cfile)
    for v in VARIANTS:
        r = rounds(C / "ref" / f"{v}_{case}.psmc")
        assert len(r) == len(c), (case, v, len(r), len(c))
        n = len(c) - 1
        lk = [rel(r[i]["lk"], c[i]["lk"]) for i in range(1, n + 1)]
        res[f"{v}_{case}"] = {
            "lk_rel_by_round": lk,
            "lk_rel_max_rounds2_5": max(lk[1:]) if n > 1 else None,
            "lk_rel_round1": lk[0],
            "params_rel_by_round": [max(rel(a, b) for a, b in zip(r[i]["pa"], c[i]["pa"])) for i in range(1, n + 1)],
            "theta_rho_tmax_rel_by_round": [max(rel(a, b) for a, b in zip(r[i]["pa"][:3], c[i]["pa"][:3]))
                                            for i in range(1, n + 1)],
            "lam_rel_by_round": [max(rel(a, b) for a, b in zip(r[i]["pa"][3:], c[i]["pa"][3:])) for i in range(1, n + 1)],
            "ri_rel_by_round": [rel(r[i]["ri"], c[i]["ri"]) for i in range(1, n + 1)],
            "it_rs_c_by_round": [(r[i]["it"], c[i]["it"]) for i in range(1, n + 1)],
        }
(C / "ref_variants.json").write_text(json.dumps(res, indent=1) + "\n")
print(f"{'run':14s} {'LK r1':>8s} {'LK r2-5 max':>11s}  {'params max (r1..)':40s} IT rs/c")
for k, x in res.items():
    lk25 = x["lk_rel_max_rounds2_5"]
    print(f"{k:14s} {x['lk_rel_round1']:8.1e} {'-' if lk25 is None else format(lk25, '11.2e'):>11}  "
          f"{' '.join(f'{p:.1e}' for p in x['params_rel_by_round']):40s} {x['it_rs_c_by_round']}")
