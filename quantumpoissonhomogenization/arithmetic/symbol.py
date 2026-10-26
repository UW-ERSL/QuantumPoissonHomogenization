"""symbol -- the full circuit for vectors whose entries are trigonometric
polynomials of the index (the scalar case of the note).

The vector is v(j) = C t(j), with

    t(j) = (1, cos theta_1(j), sin theta_1(j), ..., cos theta_R(j), sin theta_R(j)),
    theta_r(j) = pi (p_r . j + q_r) / N,

j = (j_1, ..., j_d) on d registers of n qubits (N = 2^n), integer vectors p_r,
integers q_r, and a fixed real matrix C of m0 rows and 2R + 1 columns.

Part 1 (index bits to bit strings) has three stages:
  angles   z_r = (p_r . j + q_r) / N in units of pi, modulo 2, on Fz + 1 bits:
           the bits of j are wired into place (no gates), the constant q_r is
           written by X gates, and each further signed digit of p_r is one adder;
  trig     CORDIC rotation mode on each z_r, after the range reduction
           |z| >= 1/2 -> z - 1 with start (-1/K, 0), gives cos and sin;
  rows     v_l = C_l0 + sum of C_l,c times the trig registers, by shifts and
           additions over the canonical signed digits of each coefficient; a
           row equal to one trig register with coefficient 1 uses it directly.
Part 2 (bit strings to amplitudes) is the tree R of tree.py, on the same
F fractional bits.

The whole arithmetic is built and simulated at gate level and then reversed,
so the work register returns to zero. The rotation stage of R is counted, not
simulated at gate level (see tree.py).
"""
from __future__ import annotations

import numpy as np

from .revcirc import (Circuit, BitSim, ZERO, copy_view, shift, add_view, csd,
                      load_select, cordic_rotate, cordic_gain, to_int, reverse)
from .tree import Tree, exact_unit, _toffoli, _dirty_bits


def _int_digits(p):
    """Canonical signed digits of an integer: [(e, +-1)], p = sum d 2^e."""
    return csd(p, 0)


class SymbolVector:
    """Part 1 and Part 2 for v(j) = C t(j).

    n, d     index: d registers of n qubits, N = 2^n
    F        fractional bits of every register; I = F CORDIC iterations
    angles   list of (p, q): theta = pi (p . j + q) / N, p a tuple of d integers
    C        array (m0, 2R + 1), columns (1, cos theta_1, sin theta_1, ...)
    GA       integer bits, sign included, of the trig and row registers
    G, known, signed   options of the tree (tree.py)
    """

    def __init__(self, n, d, F, angles, C, G=None, GA=4, known=None, signed=()):
        C = np.asarray(C, float)
        R = len(angles)
        assert C.shape[1] == 2 * R + 1
        c = Circuit()
        self.n, self.N, self.d, self.F, self.angles, self.C = n, 2 ** n, d, F, angles, C
        Fi = F
        self.Fi = self.Fz = self.I = Fi
        wz, w = Fi + 1, Fi + GA
        self.wz, self.w = wz, w
        K = cordic_gain(self.I)
        assert Fi >= n
        self.k = [c.alloc(n) for _ in range(d)]
        self.n_add = dict(angles=0, rows=0)
        seg = [0]

        # -- Part 1a: angle registers, z = (p . j + q)/N in units of pi, mod 2
        zs = []
        for p, q in angles:
            z = c.alloc(wz)
            Zq = int(q) * 2 ** (Fi - n) % 2 ** wz
            for b in range(wz):
                if (Zq >> b) & 1:
                    c.x(z[b])
            first = (Zq == 0)
            for i, pi in enumerate(p):
                for e, dg in _int_digits(int(pi)):
                    off = Fi - n + e
                    if off >= wz:
                        continue                  # multiple of 2 pi
                    view = ([ZERO] * off + self.k[i] + [ZERO] * wz)[:wz]
                    if first and dg > 0:
                        copy_view(c, view, z)
                    else:
                        add_view(c, z, view, neg=None if dg > 0 else True)
                        self.n_add["angles"] += 1
                    first = False
            zs.append(z)
        seg.append(len(c.gates))

        # -- Part 1b: cosine and sine by CORDIC rotation mode with range reduction
        self.trig = []
        for z in zs:
            f = c.alloc(1)[0]
            c.cx(z[-1], f); c.cx(z[-2], f)        # |z| >= 1/2
            c.cx(f, z[-1])                        # z <- z + 1 (mod 2)
            x, y = c.alloc(w), c.alloc(w)
            load_select(c, x, to_int(1 / K, Fi, w), to_int(-1 / K, Fi, w), f)
            cordic_rotate(c, x, y, z, self.I, Fi, self.Fz)
            self.trig += [x, y]
        seg.append(len(c.gates))

        # -- Part 1c: rows v_l = C_l0 + sum_c C_lc t_c. For a known-angle leaf
        # node (theta_c, scale), the left row is computed divided by scale and
        # the right row is not computed (the tree does not read it).
        m = 2 ** int(np.ceil(np.log2(max(C.shape[0], 2))))
        L = int(np.log2(m))
        self.row_scale = np.ones(C.shape[0])
        unread = set()
        for (q, p), val in dict(known or {}).items():
            if q == L - 1:
                unread.add(2 * p + 1)
                if isinstance(val, tuple) and 2 * p < C.shape[0]:
                    self.row_scale[2 * p] = val[1]
        self.rows, self.row_digits = [], []
        for l in range(C.shape[0]):
            row = C[l] / self.row_scale[l]
            if l in unread:
                self.rows.append([])                       # not read by the tree
                self.row_digits.append(0)
                continue
            nz = [(cc, row[cc]) for cc in range(1, 2 * R + 1) if row[cc] != 0]
            if row[0] == 0 and len(nz) == 1 and nz[0][1] == 1.0:
                self.rows.append(self.trig[nz[0][0] - 1])      # alias, no gates
                self.row_digits.append(0)
                continue
            out = c.alloc(w)
            C0 = to_int(row[0], Fi, w)
            for b in range(w):
                if (C0 >> b) & 1:
                    c.x(out[b])
            terms = [(self.trig[cc - 1], e, dg) for cc, coef in nz for e, dg in csd(coef, Fi)]
            pos = [i for i, t in enumerate(terms) if t[2] > 0]
            if C0 == 0 and pos:                  # a positive digit first, as a copy
                terms.insert(0, terms.pop(pos[0]))
            first, ndig = (C0 == 0), 0
            for src, e, dg in terms:
                ndig += 1
                if first and dg > 0:
                    copy_view(c, shift(src, e), out)
                else:
                    add_view(c, out, shift(src, e), neg=None if dg > 0 else True)
                    self.n_add["rows"] += 1
                first = False
            self.rows.append(out)
            self.row_digits.append(ndig)
        seg.append(len(c.gates))
        self.m0 = C.shape[0]
        self.unread = unread
        regs = self.rows + [c.alloc(w) for _ in range(m - self.m0)]   # zero padding

        # -- Part 2: the tree on the same registers
        self.tree = Tree(c, regs, Fi, G, known, signed)
        seg.append(len(c.gates))
        self.seg = seg
        self.n_arith = len(c.gates)
        c.gates += reverse(c.gates)
        self.c = c

    # -- resources ----------------------------------------------------------------
    def toffoli(self):
        """Compute plus uncompute, counted from the gate list, by part."""
        g = self.c.gates
        names = ("angles", "trig", "rows", "tree")
        return {k: _toffoli(g[self.seg[i]:self.seg[i + 1]]) for i, k in enumerate(names)}

    def formulas(self):
        """Closed forms (compute plus uncompute) for Part 1:
        angles 2 a Fz, trig 2 R I (2(w - 1) + Fz), rows 2 r (w - 1),
        with a and r the adder counts of the angle and row stages."""
        R, I, w, Fz = len(self.angles), self.I, self.w, self.Fz
        return dict(angles=2 * self.n_add["angles"] * Fz,
                    trig=2 * R * I * (2 * (w - 1) + Fz),
                    rows=2 * self.n_add["rows"] * (w - 1),
                    tree=self.tree.formulas()["toffoli"])

    # -- exact values -------------------------------------------------------------
    def index_values(self):
        N, d = self.N, self.d
        idx = np.arange(N ** d)
        return [(idx // N ** (d - 1 - i)) % N for i in range(d)]

    def exact(self, js):
        N = self.N
        cols = [np.ones(len(js[0]))]
        for p, q in self.angles:
            th = np.pi * (sum(pi * jj for pi, jj in zip(p, js)) + q) / N
            cols += [np.cos(th), np.sin(th)]
        return np.stack(cols, 1) @ self.C.T

    # -- simulation -----------------------------------------------------------------
    def simulate(self, skip_zero=True):
        """Every index j (skipping those with v(j) = 0 if skip_zero). Returns the
        prepared amplitudes, the register values of v, the exact v, the trig
        register values with their exact values, and checks."""
        js = self.index_values()
        vex = self.exact(js)
        keep = np.linalg.norm(vex, axis=1) > 1e-12 if skip_zero else np.ones(len(vex), bool)
        js = [jj[keep] for jj in js]
        vex = vex[keep]
        B = len(vex)
        sim = BitSim(self.c.n, B)
        for kj, val in zip(self.k, js):
            sim.set_int(kj, val)
        sim.run(self.c.gates[:self.tree.gate_range[0]])
        u = 2.0 ** -self.Fi                         # read Part 1 before the tree runs in place
        vreg = np.stack([sim.get_int(r) * u if r else np.zeros(B) for r in self.rows], 1) * self.row_scale
        for l in self.unread:                       # declared relation of a known leaf node
            th = self.tree.known[(self.tree.L - 1, l // 2)]
            vreg[:, l] = np.tan(th) * vreg[:, l - 1]
        treg = np.stack([sim.get_int(r) * u for r in self.trig], 1)
        ok = self.tree.run_checked(sim, self.c.gates, self.tree.gate_range[0])
        sim.run(self.c.gates[self.tree.gate_range[1]:self.n_arith])
        tex = []
        for p, q in self.angles:
            th = np.pi * (sum(pi * jj for pi, jj in zip(p, js)) + q) / self.N
            tex += [np.cos(th), np.sin(th)]
        tex = np.stack(tex, 1)
        amp = self.tree.amplitudes(self.tree.angles(sim), B)[:, :self.m0]
        sim.run(self.c.gates[self.n_arith:])
        dirty = _dirty_bits(sim, self.c.n, {q for r in self.k for q in r})
        restored = all(np.all(sim.get_int(kj, False) == val) for kj, val in zip(self.k, js))
        return amp, vreg, vex, treg, tex, dict(work_bits_left_set=dirty,
                                               inputs_restored=restored, no_overflow=ok)

    # -- bounds of the note -------------------------------------------------------------
    def eps_trig(self):
        """Lemma (rotation mode): ||(x_I, y_I) - (cos, sin)|| <= this."""
        I, u = self.I, 2.0 ** -self.Fi
        uz = np.pi * 2.0 ** -self.Fz
        return 2.0 ** (1 - I) + I * uz / 2 + cordic_gain(I) * u / 2 + np.sqrt(2) * 1.0415 * (I - 1) * u

    def entry_bounds(self):
        """Proposition (rows): |v~_l - v_l| <= ||C_l, trig part||_1 eps_trig
        + (S_l + n_l + 1) u, with S_l signed digits and n_l nonzero
        trigonometric coefficients of row l (0 for an aliased row's rounding)."""
        u, e = 2.0 ** -self.Fi, self.eps_trig()
        out = []
        for l in range(self.m0):
            row = self.C[l] / self.row_scale[l]
            if l in self.unread:
                out.append(np.tan(self.tree.known[(self.tree.L - 1, l // 2)]) * out[-1])
                continue
            if self.row_digits[l] == 0 and row[0] == 0:
                out.append(self.row_scale[l] * e)
                continue
            nz = np.count_nonzero(row[1:])
            out.append(self.row_scale[l] * (np.abs(row[1:]).sum() * e + (self.row_digits[l] + nz + 1) * u))
        return np.array(out)

    def end_to_end_bound(self, vex):
        """Corollary (end to end): for each index,
        ||prepared - v/||v|| || <= 2 delta/||v|| + T/(||v|| - delta), with
        delta = ||entry_bounds||_2 and T = m (a + 16.5 b + 6.9 eta) from
        Propositions 4, 5 and Lemma 3 at I = F, u = 2^-F, Lambda = ||v|| + delta + 0.007."""
        m, I, u = self.tree.m, self.I, 2.0 ** -self.Fi
        delta = np.linalg.norm(self.entry_bounds())
        nv = np.linalg.norm(vex, axis=1)
        lam = nv + delta + 0.007
        a = 2 * np.sqrt(2) * lam * u + 2.95 * (I - 1) * u
        b = 2 * lam * u + 2.52 * (I - 1) * u
        T = m * (a + 16.5 * b + 6.9 * np.sqrt(2) * u)
        val = 2 * delta / np.maximum(nv, 1e-300) + T / np.maximum(nv - delta, 1e-300)
        return np.where(nv > delta, np.minimum(val, 2.0), 2.0)   # two unit vectors differ by <= 2
