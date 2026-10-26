"""alternatives -- cost estimates of the two standard constructions of the
multiplexed preparation of c(k)/||c(k)||, against which the CORDIC circuit
(poisson_isometry) is compared in Sec. 5 of the paper.

    multiplexor  standard state preparation multiplexed over k (Mottonen et al.):
                 two uniformly controlled R_y gates, N^2 and 2 N^2 rotations at
                 classically tabulated angles; no Toffoli gates.
    lookup       QROM lookup of the two tabulated angles to b bits each, with
                 clean ancillas (Berry et al. 2019, Sec. 3.2): ceil(d/k1) + M (k1 - 1)
                 Toffoli gates to compute, ceil(d/k2) + k2 to uncompute by
                 measurement, d = N^2 entries of M = 2b bits, (k1 - 1) M ancillas;
                 then rotations by phase-gradient addition, estimated as 3b Toffoli
                 gates (one b-bit addition per rotation and one selection of the
                 omega register by t_0). The phase-gradient state is prepared once.

T gates are estimated as T_PER_TOFFOLI per Toffoli (Jones 2013) and
3 log2(1/eps) per rotation synthesized to accuracy eps (Ross and Selinger), with
the synthesis errors of all R rotations together at most delta. These are
estimates from published cost formulas, not gate-level circuits; the CORDIC
counts come from the gate lists of poisson_isometry.
"""
from __future__ import annotations

import numpy as np

from .poisson import poisson_isometry

T_PER_TOFFOLI = 4


def t_per_rotation(eps):
    """T gates of one single-qubit rotation to accuracy eps (Ross and Selinger, leading term)."""
    return 3 * np.log2(1 / eps)


def _qroam(d, M):
    """(k1, compute cost, k2, uncompute cost) with k1, k2 powers of two."""
    ks = [2 ** j for j in range(int(np.log2(d)) + 1)]
    c1 = [int(np.ceil(d / k)) + M * (k - 1) for k in ks]
    c2 = [int(np.ceil(d / k)) + k for k in ks]
    i1, i2 = int(np.argmin(c1)), int(np.argmin(c2))
    return ks[i1], c1[i1], ks[i2], c2[i2]


def fractional_bits(N, delta, c=6):
    """F with c N 2^-F <= delta (the measured bound of Theorem 1)."""
    return int(np.ceil(np.log2(c * N / delta)))


def options(N, delta):
    """Estimated cost of the three constructions at the accuracy delta, for N = 2^n."""
    n = int(np.log2(N))
    out = {}
    R = 3 * N * N
    out["multiplexor"] = dict(toffoli=0, rotations=R, work=0)
    b = int(np.ceil(np.log2(2 * np.pi / delta)))       # two angles, each rounded to pi 2^-b
    d, M = N * N, 2 * b
    k1, c1, k2, c2 = _qroam(d, M)
    out["lookup"] = dict(toffoli=c1 + c2 + 3 * b, rotations=0, work=k1 * M + 2 * b, b=b, k1=k1)
    F = fractional_bits(N, delta)
    sv = poisson_isometry(n, F)
    r = sv.tree.rotation_counts()
    out["arithmetic"] = dict(toffoli=sum(sv.toffoli().values()) + r["toffoli"],
                             rotations=r["controlled_ry"] + r["fixed_ry"],
                             work=sv.c.n - 2 * n + r["temporaries"], F=F)
    for v in out.values():
        v["T"] = T_PER_TOFFOLI * v["toffoli"] + (v["rotations"] * t_per_rotation(delta / v["rotations"])
                                                 if v["rotations"] else 0.0)
    return out
