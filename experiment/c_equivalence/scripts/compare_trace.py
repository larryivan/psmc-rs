"""The sequence of points C psmc's Hooke-Jeeves search evaluates (traced by c_trace.patch) against the port's.

Usage: compare_trace.py C_TRACE PORT_TRACE   (the HJ lines of each: count, objective, parameters)
"""
import sys


def load(path):
    out = []
    for line in open(path):
        if line.startswith("HJ\t"):
            f = line.split("\t")
            out.append((int(f[1]), float(f[2]), [float(x) for x in f[3:]]))
    return out


c, r = load(sys.argv[1]), load(sys.argv[2])
c0, r0 = c[0][1], r[0][1]
first_div, max_df, rejected = None, 0.0, 0
for i, (a, b) in enumerate(zip(c, r)):
    rel = max(abs(x - y) / abs(y) if y else abs(x - y) for x, y in zip(a[2], b[2]))
    if rel > 0:
        first_div = i
        break
    if abs(a[1]) < 1e299 and abs(b[1]) < 1e299:
        max_df = max(max_df, abs((a[1] - c0) - (b[1] - r0)))
    else:
        assert abs(a[1]) >= 1e299 and abs(b[1]) >= 1e299, (i, a[1], b[1])
        rejected += 1
print(f"evaluations: C {len(c)}, port {len(r)}")
print(f"identical parameter sequence for the first {first_div if first_div is not None else min(len(c), len(r))} evaluations")
if first_div is not None:
    i = first_div
    print(f"first difference at evaluation {i + 1}: C f-f0 {c[i][1] - c0:.6e}, port {r[i][1] - r0:.6e}")
    print(f"  previous: C f-f0 {c[i - 1][1] - c0:.17e}, port {r[i - 1][1] - r0:.17e}")
fin = max(abs(x - y) / abs(y) for x, y in zip(c[-1][2], r[-1][2]))
print(f"last evaluated points: largest relative parameter difference {fin:.2e}")
print(f"f - f0 at the end: C {c[-1][1] - c0:.10e}, port {r[-1][1] - r0:.10e}")
print(f"points both reject (C 1e300, port inf) in the common prefix: {rejected}")
print(f"largest absolute difference of f - f0 over the common prefix: {max_df:.2e}")
