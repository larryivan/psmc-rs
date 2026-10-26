"""The rounds of a .psmc file, C psmc's or psmc-rs's."""


def rounds(path):
    """One dict per round 0..N: lk, ri, pa (theta, rho, t_max, then the lambdas) and it, the objective evaluations
    of the M-step that produced the round (its IT line; None for round 0)."""
    out, cur, pending_it = [], None, None
    for line in open(path):
        f = line.rstrip("\n").split("\t")
        tag = f[0]
        if tag == "RD":
            cur = {"rd": int(f[1]), "it": pending_it}
            out.append(cur)
            pending_it = None
        elif tag == "IT":
            pending_it = int(f[1])
        elif cur is None:
            continue
        elif tag == "LK":
            cur["lk"] = float(f[1])
        elif tag == "RI":
            cur["ri"] = float(f[1])
        elif tag == "PA":
            v = f[1].split()
            cur["pattern"], cur["pa"] = v[0], [float(x) for x in v[1:]]
    return out
