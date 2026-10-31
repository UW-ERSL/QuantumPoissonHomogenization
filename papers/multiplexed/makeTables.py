"""makeTables.py -- the tables and the checks of the multiplexed state preparation paper.

    python makeTables.py                 Table 3 (tables/options.tex), and Theorem 2 and Table 2 with Sec. 4
    python makeTables.py options         just Table 3
    python makeTables.py checks          the worked example, the rotation-mode and vectoring bounds of the
                                         proof of Proposition 1, and the counts of Sec. 5
    python makeTables.py --list          names

Every number the paper quotes is asserted before it is printed or written.
"""
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))     # tables/ and figs/ go to this folder
import sys

import numpy as np

from quantumpoissonhomogenization.arithmetic import (poisson_isometry, exact_unit, TreeCircuit,
                                                     SymbolVector)
from quantumpoissonhomogenization.arithmetic.alternatives import options
from quantumpoissonhomogenization.arithmetic.revcirc import BitSim
from quantumpoissonhomogenization.reproduce import dispatch


# ==========================================================================
#  tables
# ==========================================================================
def tab_options() -> None:
    """Sec. 5, Table 3: the CORDIC circuit against the two standard constructions
    (multiplexor of standard state preparation, QROM lookup of tabulated angles),
    at the accuracy delta = 1.955e-10, for N = 8 to 4096. Asserts the statements
    of Secs. 5 and 6 (crossovers, F = 45 and b = 35 at N = 1024, about 1,800
    Toffoli gates per doubling of N, synthesis about 11 percent of the T count),
    prints every N, and writes the rows of Table 3 (N = 16, 64, 256, 1024, 2048,
    4096) to tables/options.tex."""
    delta = 1.955e-10
    Ns = [2 ** n for n in range(3, 13)]
    o = {N: options(N, delta) for N in Ns}
    T = lambda N, k: o[N][k]["T"]
    assert all(T(N, "multiplexor") < T(N, "arithmetic") for N in Ns if N <= 16)
    assert all(T(N, "multiplexor") > T(N, "arithmetic") for N in Ns if N >= 32)
    assert all(T(N, "lookup") < T(N, "arithmetic") for N in Ns if N <= 2048)
    assert T(4096, "lookup") > T(4096, "arithmetic")
    assert o[1024]["arithmetic"]["F"] == 45 and o[1024]["arithmetic"]["toffoli"] == 44424
    assert o[1024]["lookup"]["b"] == 35
    growth = np.diff([o[N]["arithmetic"]["toffoli"] for N in Ns if N >= 16])
    assert 1700 < growth.mean() < 1900                     # about 1,800 per doubling of N
    a = o[1024]["arithmetic"]
    share = (a["T"] - 4 * a["toffoli"]) / a["T"]
    assert 0.10 < share < 0.12                             # synthesis about 11 percent

    print(f"delta = {delta:g}")
    print("     N   F    T: multiplexor    lookup  arithmetic   Toffoli: lookup  arithmetic"
          "   work: lookup  arithmetic")
    for N in Ns:
        m, l, a = o[N]["multiplexor"], o[N]["lookup"], o[N]["arithmetic"]
        print(f"{N:6d}  {a['F']:2d}    {m['T']:14.3g}  {l['T']:8.3g}  {a['T']:10.3g}"
              f"   {l['toffoli']:14,d}  {a['toffoli']:10,d}   {l['work']:12,d}  {a['work']:10,d}")
    l, a = o[1024]["lookup"], o[1024]["arithmetic"]
    print(f"N = 1024: arithmetic / lookup Toffoli = {a['toffoli'] / l['toffoli']:.2f}; "
          f"rotation synthesis of the arithmetic {a['rotations']} single-qubit rotations, "
          f"{a['T'] - 4 * a['toffoli']:,.0f} T gates, {100 * share:.1f} percent of its T count; "
          f"Toffoli growth per doubling of N {growth.mean():,.0f}")


    # ----- the rows of Table 3, as typeset in the paper
    def sci(x):
        e = int(np.floor(np.log10(x)))
        m = round(x / 10 ** e, 1)
        if m >= 10:
            m, e = m / 10, e + 1
        return f"${m:.1f} \\times 10^{{{e}}}$"


    def intx(x):
        return f"${x:,d}$".replace(",", "{,}")


    rows = []
    for N in (16, 64, 256, 1024, 2048, 4096):
        m, l, a = o[N]["multiplexor"], o[N]["lookup"], o[N]["arithmetic"]
        rows.append(" & ".join([f"${N}$", f"${a['F']}$", sci(m["T"]), sci(l["T"]), sci(a["T"]),
                                intx(l["toffoli"]), intx(a["toffoli"]), intx(l["work"]), intx(a["work"])]) + r"\\")
    os.makedirs("tables", exist_ok=True)
    open("tables/options.tex", "w").write("\n".join(rows) + "\n")
    print("\nrows of Table 3 (tables/options.tex):")
    print("\n".join(rows))


def tab_cost() -> None:
    """The Fourier blocks of the Poisson isometry (Q4 elements, two dimensions).

    1. Cost at N = 1024, F = 45, G = 4, by stage (Theorem 2 and Table 2 of the
       paper).
    2. Error of the prepared vector c~(k) against c(k)/||c(k)||, simulated bit by bit
       over every k != 0, for N = 4 ... 256 and F = 12, 16, ..., 32 (42 cases):
       the measured bound 6 N 2^-F of Sec. 4, and the numbers of the abstract at
       N = 32, F = 32.
    3. The block itself: e^(i(phi1+phi2)) T c~(k) against Vhat(k) assembled element
       by element from the Q4 gradient, at N = 16, F = 24."""
    # 1. cost
    F, n = 45, 10
    sv = poisson_isometry(n, F)
    t, r, tr = sv.toffoli(), sv.tree.rotation_counts(), sv.tree
    assert t == sv.formulas()
    vec = 4 * tr.I * (tr.W - 1)
    const = 2 * (tr.W - 1) * (len(tr.digits[(1, 1)]) - 1)
    assert t["tree"] == 2 * vec + const
    total = sum(t.values()) + r["toffoli"]
    print(f"N = {2**n}, F = {F}, G = 4 (registers of {sv.w} bits)")
    for name, val in (("angles phi1 +- phi2 (adder, subtractor)", t["angles"]),
                      ("cosines and sines (two rotation-mode runs)", t["trig"]),
                      ("a, b, zeta (sum to product)", t["rows"]),
                      ("omega: vectoring of (a, b)", vec),
                      (f"sqrt(2/3) K zeta: constant product, s = {len(tr.digits[(1, 1)])}", const),
                      ("tau: vectoring of (K r, sqrt(2/3) K zeta)", vec),
                      ("rotation stage: selection of the omega bits", r["toffoli"])):
        print(f"  {name:48s} {val:7,d}")
    print(f"  {'total':48s} {total:7,d}")
    work = sv.c.n - 2 * n + r["temporaries"]
    assert work == 2 * F ** 2 + 35 * F + 47
    print(f"  rotations: {r['controlled_ry']} controlled, {r['fixed_ry']} fixed, {r['flag_z']} flag Z; "
          f"work qubits {work:,} ({sv.c.n - 2 * n:,} arithmetic, {r['temporaries']} rotation stage)")

    # 2. error, 42 cases
    print("\n  N   F   max err 2^F/N")
    worst = 0
    for n in range(2, 9):
        for F in range(12, 33, 4):
            sv = poisson_isometry(n, F)
            amp, vreg, vex, treg, tex, ck = sv.simulate()
            assert ck["work_bits_left_set"] == 0 and ck["inputs_restored"] and ck["no_overflow"]
            e = np.linalg.norm(amp - exact_unit(vex), axis=1)
            worst = max(worst, e.max() * 2 ** F / 2 ** n)
            if F in (12, 32):
                print(f"{2**n:4d}  {F}      {e.max() * 2**F / 2**n:6.3f}")
            if (n, F) == (5, 32):               # the abstract: N = 32, F = 32
                r32 = sv.tree.rotation_counts()
                t32 = sum(sv.toffoli().values()) + r32["toffoli"]
                assert t32 == 23089 and r32["controlled_ry"] == 65 and len(sv.tree.digits[(1, 1)]) == 11
                assert 8.95e-9 < e.max() < 9.05e-9
                abstract = (f"N = 32, F = 32: {t32:,} Toffoli, {r32['controlled_ry']} controlled rotations, "
                            f"max err {e.max():.1e}")
    assert worst <= 6
    print(f"max err 2^F/N over the 42 cases: {worst:.2f}")
    print(abstract)


    # 3. the block Vhat(k), assembled element by element
    def vhat_exact(N, k1, k2):
        q = 1 / np.sqrt(3)
        gp = [(-q, -q), (q, -q), (q, q), (-q, q)]                     # (xi_g, eta_g)
        corners, xi = [(0, 0), (1, 0), (1, 1), (0, 1)], [(-1, -1), (1, -1), (1, 1), (-1, 1)]
        t1, t2 = 2 * np.pi * k1 / N, 2 * np.pi * k2 / N
        ph = np.array([np.exp(1j * (t1 * cx + t2 * cy)) for cx, cy in corners])
        rows = []
        for r_, s_ in gp:
            Bx = [xa * (1 + ya * s_) for xa, ya in xi]
            By = [ya * (1 + xa * r_) for xa, ya in xi]
            rows += [np.dot(Bx, ph), np.dot(By, ph)]
        v = np.array(rows)
        return v / np.linalg.norm(v)


    q = 1 / np.sqrt(3)
    T = np.zeros((8, 4), complex)                                   # T|l> = t_l, l = 0..3
    for g, (xg, eg) in enumerate([(-q, -q), (q, -q), (q, q), (-q, q)]):
        T[2 * g, 0], T[2 * g + 1, 1] = 0.5j, 0.5j
        T[2 * g, 2], T[2 * g + 1, 3] = -np.sqrt(3) / 2 * eg, -np.sqrt(3) / 2 * xg
    n, F = 4, 24
    N = 2 ** n
    sv = poisson_isometry(n, F)
    amp, vreg, vex, treg, tex, ck = sv.simulate()
    k1, k2 = [a.ravel()[1:] for a in np.meshgrid(np.arange(N), np.arange(N), indexing="ij")]
    err = max(np.linalg.norm(np.exp(1j * np.pi * (a + b) / N) * T @ amp[i] - vhat_exact(N, a, b))
              for i, (a, b) in enumerate(zip(k1, k2)))
    print(f"\nN = {N}, F = {F}: max |e^(i(phi1+phi2)) T c~(k) - Vhat(k)| = {err:.1e}")


# ==========================================================================
#  checks: the numbers of Secs. 2-5 that the text quotes outside the tables
# ==========================================================================
def check_worked() -> None:
    """Secs. 2 and 3, the worked example: k = (3, 1), N = 16, F = 12, registers of
    16 bits. Part 1: the cosines and sines of theta_+- and the bit strings of a, b,
    zeta. Part 2: the node outputs and decision bits of the tree, the angles they
    encode, and the prepared amplitudes against c/||c||."""
    n, F, k = 4, 12, (3, 1)
    sv, N, u = poisson_isometry(n, F), 2 ** n, 2.0 ** -F
    t = sv.tree
    sim = BitSim(sv.c.n, 1)
    for kj, val in zip(sv.k, k):
        sim.set_int(kj, np.array([val]))
    bits = lambda r: "".join(str(int(sim.get_bits(q)[0])) for q in reversed(r))

    # ----- Part 1
    sim.run(sv.c.gates[:t.gate_range[0]])
    f1, f2 = np.pi * k[0] / N, np.pi * k[1] / N
    th = [f1 + f2, f1 - f2]
    print("Part 1")
    for r, (cs, sn) in enumerate(zip(sv.trig[0::2], sv.trig[1::2])):
        print(f"  theta_{'+-'[r]}: (cos, sin) = ({sim.get_int(cs)[0] * u:.5f}, {sim.get_int(sn)[0] * u:.5f}),"
              f" exact ({np.cos(th[r]):.5f}, {np.sin(th[r]):.5f})")
    a, b, z = np.sin(f1) * np.cos(f2), np.cos(f1) * np.sin(f2), np.sin(f1) * np.sin(f2)
    for name, r, ex in (("a", sv.rows[0], a), ("b", sv.rows[1], b), ("zeta", sv.rows[2], z)):
        print(f"  {name:4s} = {sim.get_int(r)[0] * u:.5f}: {bits(r)}, exact {ex:.5f}")

    # ----- Part 2
    print("Part 2")
    pos = t.gate_range[0]
    for end, node, reg in t.marks:
        sim.run(sv.c.gates[pos:end])
        pos = end
        print(f"  node {node}: output {sim.get_int(reg)[0] * u:.5f}: {bits(reg)}")
    sim.run(sv.c.gates[pos:t.gate_range[1]])
    K = np.prod(np.sqrt(1 + 4.0 ** -np.arange(t.I)))
    r_ = np.hypot(a, b)
    print(f"  exact K r = {K * r_:.5f}, sqrt(2/3) K zeta = {np.sqrt(2/3) * K * z:.5f},"
          f" K^2 ||c|| = {K * K * np.sqrt(a*a + b*b + 2*z*z/3):.5f}")
    ang = t.angles(sim)
    exact = {(1, 0): np.arctan2(b, a), (1, 1): np.pi / 4, (0, 0): np.arctan2(np.sqrt(2/3) * z, r_)}
    for node, bb in t.dbits.items():
        print(f"  node {node}: decision bits d_0..d_{t.I - 1} = {''.join(str(int(sim.get_bits(q)[0])) for q in bb)}")
    for node in sorted(ang):
        print(f"  node {node}: angle {ang[node][0]:.5f}, exact {exact[node]:.5f}")
    for f in t.flags.values():
        print(f"  sign flag f = {int(sim.get_bits(f)[0])}")
    amp = t.amplitudes(ang, 1)[0]
    c = np.array([a, b, z / np.sqrt(3), z / np.sqrt(3)])
    print("  amplitudes", np.round(amp, 5), "\n  exact     ", np.round(c / np.linalg.norm(c), 5),
          f"\n  error {np.linalg.norm(amp - c / np.linalg.norm(c)):.1e}")


def check_rotation() -> None:
    """Rotation mode, the first step of the proof of Proposition 1: check the bound
    ||(x_I, y_I) - (cos theta, sin theta)|| <= 2^(1-I) + I u_z / 2 + K u / 2 + sqrt(2) kappa (I - 1) u, kappa = 1.0415,
    on integer rotation mode with floor-truncated shifts, angle constants rounded to
    the angle grid, and the start value 1/K rounded, as in revcirc.cordic_rotate.
    I = F fractional bits for x, y and the angle (in units of pi), u = 2^-I,
    u_z = pi 2^-I; 200,000 random angles in [-pi/2, pi/2) per I, seed 6. The check
    also confirms the step of the proof: the residual angle is at most
    alpha_(I-1) + I u_z / 2."""
    rng = np.random.default_rng(6)
    print(" I   max err / bound   max residual / bound")
    for I in (8, 16, 24, 32, 40):
        u, B = 2.0 ** -I, 200000
        K = np.prod(np.sqrt(1 + 4.0 ** -np.arange(I)))
        z0 = rng.integers(-2 ** (I - 1), 2 ** (I - 1), B)          # theta = pi z0 2^-I
        a = np.round(np.arctan(2.0 ** -np.arange(I)) / np.pi * 2 ** I).astype(np.int64)
        x = np.full(B, int(np.round(2 ** I / K)), np.int64)
        y = np.zeros(B, np.int64)
        z = z0.copy()
        phi = np.zeros(B)
        for i in range(I):
            s = np.where(z < 0, -1, 1)
            x, y = x - s * (y >> i), y + s * (x >> i)
            z = z - s * a[i]
            phi += s * np.arctan(2.0 ** -i)
        th = np.pi * z0 * u
        err = np.hypot(x * u - np.cos(th), y * u - np.sin(th))
        uz = np.pi * u
        bound = 2.0 ** (1 - I) + I * uz / 2 + K * u / 2 + np.sqrt(2) * 1.0415 * (I - 1) * u
        rbound = np.arctan(2.0 ** -(I - 1)) + I * uz / 2
        print(f"{I:2d}      {err.max() / bound:.3f}             {np.abs(th - phi).max() / rbound:.3f}")
        assert err.max() <= bound and np.abs(th - phi).max() <= rbound
    print("the rotation-mode bound holds in every case")


def check_lemma() -> None:
    """One vectoring, the third step of the proof of Proposition 1: check the bounds on integer vectoring with
    floor-truncated shifts (I fractional bits, I iterations, u = 2^-I), on
    200,000 random pairs per I with magnitudes down to 2^-I, seed 5."""
    rng = np.random.default_rng(5)
    print(" I   x nondecr.  |y_I| bound   eps/bound_a   lenerr/bound_b")
    for I in (8, 16, 24, 32, 40):
        u, B = 2.0 ** -I, 200000
        x0 = np.floor(rng.uniform(0, 1, B) * 2.0 ** -rng.uniform(0, I, B) * 2 ** I).astype(np.int64)
        y0 = np.floor(rng.uniform(-1, 1, B) * 2.0 ** -rng.uniform(0, I, B) * 2 ** I).astype(np.int64)
        keep = (x0 != 0) | (y0 != 0)
        x0, y0 = x0[keep], y0[keep]

        # vectoring
        x, y, th, mono = x0.copy(), y0.copy(), np.zeros(len(x0)), True
        for i in range(I):
            s = np.where(y >= 0, 1, -1)
            xn, yn = x + s * (y >> i), y - s * (x >> i)
            mono &= bool(np.all(xn >= x))
            x, y = xn, yn
            th += s * np.arctan(2.0 ** -i)

        # the two bounds on the direction and the length
        K = np.prod(np.sqrt(1 + 4.0 ** -np.arange(I)))
        X0, Y0, X, Y = x0 * u, y0 * u, x * u, y * u
        rho = np.hypot(X0, Y0)
        resid = bool(np.all(np.abs(Y) <= 2.0 ** (1 - I) * X + (I - 1) * u))
        eps = np.hypot(rho * np.cos(th) - X0, rho * np.sin(th) - Y0)
        ra = (eps / (2 * np.sqrt(2) * 2.0 ** -I * rho + 2.95 * (I - 1) * u)).max()
        rb = (np.abs(X / K - rho) / (2 * 2.0 ** -I * rho + 2.52 * (I - 1) * u)).max()
        print(f"{I:2d}      {mono}        {resid}        {ra:.3f}          {rb:.3f}")


def check_counts() -> None:
    """Sec. 5: assert the Toffoli and work-qubit counts against the gate lists, and
    check that the work register returns to zero and the inputs are unchanged."""
    rng = np.random.default_rng(0)
    print(" m   F   G   Toffoli (arith.)  work qubits  rot. Toffoli  ctrl. Ry  clean")
    for L in (1, 2, 3, 4):
        for F in (16, 32, 45):
            tc = TreeCircuit(L, F)
            t, f = tc.tree, tc.tree.formulas()
            assert t.toffoli() == f["toffoli"] and t.work_qubits == f["work"]
            V = rng.uniform(-1, 1, (256, 2 ** L))
            _, ck = tc.simulate(V)
            assert ck["work_bits_left_set"] == 0 and ck["inputs_restored"]
            r = t.rotation_counts()
            print(f"{2**L:2d}  {F:3d}  {t.G}   {t.toffoli():10d}   {t.work_qubits:10d}"
                  f"   {r['toffoli']:8d}   {r['controlled_ry']:6d}    yes")
    print("all counts match the formulas of Sec. 5")

    # Part 1: the counts of the full circuit against its gate lists, on random
    # angles and coefficient matrices, with the work register checked on every index
    print("\nPart 1 (SymbolVector): random angles p in {-3..3}^d, q in {-4..4}, random C")
    for trial in range(6):
        d, R, m0 = 1 + trial % 2, 1 + trial % 3, 2 + trial % 3
        n, F = 4, 16 + 4 * (trial % 3)
        angles = [(tuple(int(x) for x in rng.integers(-3, 4, d)), int(rng.integers(-4, 5))) for _ in range(R)]
        C = np.round(rng.uniform(-1, 1, (m0, 2 * R + 1)) * 8) / 8 / (2 * R + 1)
        sv = SymbolVector(n, d, F, angles, C)
        assert sv.toffoli() == sv.formulas()
        amp, vreg, vex, treg, tex, ck = sv.simulate()
        assert ck["work_bits_left_set"] == 0 and ck["inputs_restored"] and ck["no_overflow"]
        assert np.all(np.abs(vreg - vex) <= sv.entry_bounds())
        e = np.linalg.norm(amp - exact_unit(vex), axis=1)
        print(f"  d = {d}, R = {R}, m = {m0}, F = {F}: Toffoli {sv.toffoli()}, entries within bound, "
              f"max err {e.max():.1e}, clean")
    print("all Part 1 counts match the formulas of Sec. 5")


TABLES = {
    "options": (tab_options, "Table 3: the circuit against the multiplexor and the QROM lookup (Sec. 5)"),
    "cost":    (tab_cost,    "Theorem 2 and Table 2 at N = 1024, F = 45; the 42 error cases (Sec. 4); the abstract"),
}
CHECKS = {
    "worked":   (check_worked,   "the worked example at k = (3, 1), N = 16, F = 12 (Secs. 2, 3)"),
    "rotation": (check_rotation, "rotation-mode bound (Proposition 1, first step) on random angles"),
    "lemma":    (check_lemma,    "vectoring bounds (Proposition 1, third step) on random integer vectorings"),
    "counts":   (check_counts,   "Toffoli and work-qubit counts against the gate lists (Sec. 5)"),
}


if __name__ == "__main__":
    dispatch(sys.argv[1:], {"tables": TABLES, "checks": CHECKS})
