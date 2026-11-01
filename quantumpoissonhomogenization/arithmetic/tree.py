"""tree -- conversion of an m-entry vector into amplitudes by a tree of
CORDIC vectorings (the circuit R of the paper).

Given m = 2^L value registers v_0..v_{m-1} (two's complement, F fractional
bits), the arithmetic stage copies them into working registers of
W = F + G bits, or vectors in place when a register already has W bits, and
runs m - 1 vectorings of I = F iterations from the leaves to the root. Node
(q, p) at level q covers leaves p 2^(L-q) to
(p+1) 2^(L-q) - 1; its decision bits encode atan2(norm of right half, norm of
left half). A leaf pair with v_2p < 0 is negated under a flag, which adds pi to
its angle. The uncomputation is the exact reverse.

The arithmetic stage is built and simulated at gate level. The rotation stage
(target qubits t_0 .. t_{L-1}, t_0 most significant) is counted by
rotation_counts() but not simulated at gate level: the simulation applies the
angles it encodes, theta = A - 2 sum_i d_i alpha_i (+ pi f at the leaves).
"""
from __future__ import annotations

import numpy as np

from .revcirc import (Circuit, BitSim, copy_view, shift, add_view, csd,
                      load_select, cordic_rotate, cordic_vector, cordic_gain,
                      to_int, reverse)



def range_bits(L, vmax=1.0):
    """Integer bits G with K^L sqrt(m) vmax < 2^(G-1) (root register range)."""
    bound = cordic_gain(64) ** L * np.sqrt(2 ** L) * vmax
    return int(np.ceil(np.log2(bound))) + 2


def _toffoli(gates):
    """Toffoli count, compute plus uncompute, of a one-way gate list."""
    k = {t: sum(1 for q in gates if q[0] == t) for t in ("ccx", "and", "andu")}
    return 2 * k["ccx"] + k["and"] + k["andu"]


# ==========================================================================
# the circuit R (arithmetic stage), on an existing circuit
# ==========================================================================
class Tree:
    """The circuit R on caller registers `regs` (m = 2^L of them).

    Options:
      G      integer bits of the working registers, sign included, default
             range_bits(L); the simulation checks that no value overflows;
      known  {(q, p): theta_c} declares node (q, p) a known-angle node: the
             caller guarantees rho_right = tan(theta_c) rho_left for every
             index, with cos(theta_c) >= 1/sqrt 2. The node has no vectoring;
             its output is (K / cos theta_c) x_left, by shifts and additions
             over the signed digits of the constant, and its rotation is the
             fixed R(theta_c). At the leaf level the right register is not read.
             A value may also be a pair (theta_c, scale) for a leaf node whose
             left register holds the entry divided by scale; the constant
             product then uses K scale / cos theta_c.
      signed nodes in `known` at the leaf level whose left entry may be
             negative; they get the sign flag of a leaf pair.
    """

    def __init__(self, c, regs, F, G=None, known=None, signed=()):
        m = len(regs)
        L = int(np.log2(m))
        assert 2 ** L == m, "pad with zero registers to a power of 2"
        self.c, self.regs, self.F, self.m, self.L = c, regs, F, m, L
        self.Fi = F
        self.I = self.Fi
        self.G = range_bits(L) if G is None else G
        self.W = W = self.Fi + self.G
        self.known, self.scale = {}, {}
        for node, val in dict(known or {}).items():
            th, sc = (val if isinstance(val, tuple) else (val, 1.0))
            assert sc == 1.0 or node[0] == L - 1, "a scale applies to leaf nodes only"
            self.known[node], self.scale[node] = float(th), float(sc)
        for (q, p), th in self.known.items():
            assert np.cos(th) >= 1 / np.sqrt(2) - 1e-12, "order children so cos >= 1/sqrt 2"
            assert q == L - 1 or th >= 0, "partial norms are nonnegative: theta_c in [0, pi/4]"
        self.signed = set(signed)
        K = cordic_gain(self.I)
        n0, g0 = c.n, len(c.gates)

        # working copies, sign-extended (CNOT only); a known leaf node reads its left entry only
        skip = {2 * p + 1 for (q, p) in self.known if q == L - 1}
        xs, self.n_copies, used = [], 0, set()
        for l, r in enumerate(regs):
            if l in skip:
                xs.append(None)
                continue
            if len(r) == W and r[0] not in used:
                xs.append(r)                     # no shift needed: vector in place,
                used.add(r[0])                   # restored by the uncomputation
                continue
            x = c.alloc(W)
            copy_view(c, r + [r[-1]] * (W - len(r)), x)
            xs.append(x)
            self.n_copies += 1

        self.flags, self.dbits, self.digits, self.marks = {}, {}, {}, []
        cur = xs
        for q in range(L - 1, -1, -1):
            nxt = []
            for p in range(2 ** q):
                x, y = cur[2 * p], cur[2 * p + 1]
                leaf = q == L - 1
                if leaf and ((q, p) not in self.known or (q, p) in self.signed):
                    f = c.alloc(1)[0]            # sign flag: (x, y) <- ~(x, y) = -(x, y) - ulp
                    c.cx(x[-1], f)
                    for j in range(W):
                        c.cx(f, x[j])
                        if y is not None:
                            c.cx(f, y[j])
                    self.flags[p] = f
                if (q, p) in self.known:         # known-angle node: constant product
                    M = K * self.scale[(q, p)] / np.cos(self.known[(q, p)])
                    out, self.digits[(q, p)] = _const_product(c, x, M, self.Fi)
                    nxt.append(out)
                else:                            # vectoring, in place on the children's x registers
                    self.dbits[(q, p)] = cordic_vector(c, x, y, self.I)
                    nxt.append(x)
                self.marks.append((len(c.gates), (q, p), nxt[-1]))
            cur = nxt
        self.root = cur[0]
        self.gate_range = (g0, len(c.gates))
        self.work_qubits = c.n - n0

    # -- decoding ---------------------------------------------------------------
    def angles(self, sim):
        """Node angles: A - 2 sum_i d_i alpha_i at vectoring nodes, theta_c at
        known nodes, plus pi f where a leaf sign flag is set."""
        al = np.arctan(2.0 ** -np.arange(self.I))
        th = {}
        for (q, p), bits in self.dbits.items():
            th[(q, p)] = sum((1 - 2 * sim.get_bits(d)) * a for d, a in zip(bits, al))
        for node, t in self.known.items():
            th[node] = np.full(sim.B, float(t))
        for p, f in self.flags.items():
            th[(self.L - 1, p)] = th[(self.L - 1, p)] + np.pi * sim.get_bits(f)
        return th

    def run_checked(self, sim, gates, start):
        """Run the arithmetic gates[start:] and check, when each node finishes,
        that its output lies in (-1, 2^(G-1)), so no computed value wrapped.
        Returns True if no output overflowed."""
        ok, pos = True, start
        for end, _, reg in self.marks:
            sim.run(gates[pos:end])
            pos = end
            v = sim.get_int(reg)
            # a wrapped value would read below -1; a known node may output -O(ulp)
            ok &= bool(np.all((v > -2 ** self.Fi) & (v < 2 ** (self.G - 1 + self.Fi))))
        sim.run(gates[pos:self.gate_range[1]])
        return ok

    def amplitudes(self, th, B):
        """Prepared vector: product of cos/sin of the node angles along each path."""
        L, m = self.L, self.m
        amp = np.ones((B, m))
        for l in range(m):
            for q in range(L):
                t = th[(q, l >> (L - q))]
                amp[:, l] *= np.sin(t) if (l >> (L - 1 - q)) & 1 else np.cos(t)
        return amp

    # -- resources --------------------------------------------------------------
    def toffoli(self):
        """Arithmetic stage, compute plus uncompute, counted from the gate list."""
        return _toffoli(self.c.gates[slice(*self.gate_range)])

    def formulas(self):
        """Closed forms of Sec. 5 for the arithmetic stage: vectoring nodes cost
        4 I (W - 1) each, known nodes 2 (W - 1)(s - 1) for s signed digits."""
        I, W, m = self.I, self.W, self.m
        n_vec = len(self.dbits)
        n_copy = self.n_copies
        s = [len(d) for d in self.digits.values()]
        return dict(toffoli=4 * I * (W - 1) * n_vec + 2 * (W - 1) * sum(k - 1 for k in s),
                    work=n_copy * W + len(self.flags) + 2 * W + W * len(s)
                    + n_vec * (I + I * W - I * (I - 1) // 2))

    def rotation_counts(self):
        """Rotation stage (counted, not simulated at gate level).

        Level q has vectoring nodes V_q and known nodes C_q. The decision bits
        are selected over V_q only: D_i = XOR_{p in V_q} [t_0..t_{q-1} = p]
        d_i^(q, p), by unary iteration over the 2^q prefixes (2^q - 1
        temporary ANDs, repeated for the measurement-based uncompute) and one
        Toffoli per (p, i); at level 0 the root bits are used directly. The
        leaf sign flags are selected the same way and take a Z gate. Each level
        applies R(A) to t_q if V_q is not empty and I rotations R(-2 alpha_i)
        controlled by D_i; each known node adds the rotation R(theta_c) (or
        R(theta_c - A) when V_q is not empty) by a fixed angle, controlled by
        its prefix (uncontrolled at the root). `temporaries` counts the bits
        D_i, the copies of the flags, and the ANDs of the unary iteration,
        which are live during the stage only.
        """
        I, L = self.I, self.L
        tof = ctrl = fixed = tmp = 0
        for q in range(L):
            V = [p for p in range(2 ** q) if (q, p) in self.dbits]
            C = [p for p in range(2 ** q) if (q, p) in self.known]
            nflag = len(self.flags) if q == L - 1 else 0
            if q >= 1 and (V or nflag):
                tof += 2 * (2 ** q - 1) + I * len(V) + nflag
                tmp = max(tmp, (2 ** q - 1) + I * len(V) + nflag)
            ctrl += (I if V else 0) + (len(C) if q >= 1 else 0)
            fixed += (1 if V else 0) + (len(C) if q == 0 else 0)
        return dict(toffoli=tof, controlled_ry=ctrl, fixed_ry=fixed,
                    flag_z=len(self.flags), temporaries=tmp)


def _const_product(c, src, M, F):
    """Fresh register <- M * src by shifts and additions over the canonical
    signed digits of M: the leading digit is a copy, the others additions.
    Returns the register and the digits."""
    digits = csd(M, F)
    out = c.alloc(len(src))
    j0, d0 = digits[-1]
    assert d0 > 0
    copy_view(c, shift(src, j0), out)
    for j, d in digits[:-1]:
        add_view(c, out, shift(src, j), neg=None if d > 0 else True)
    return out, digits


# ==========================================================================
# drivers
# ==========================================================================
def exact_unit(V):
    V = np.asarray(V, float)
    return V / np.linalg.norm(V, axis=1, keepdims=True)


def _dirty_bits(sim, n, protected):
    return int(sum(sim.get_bits(q).sum() for q in range(n) if q not in protected))


class TreeCircuit:
    """R alone, on caller registers loaded with given values (rounded to F bits)."""

    def __init__(self, L, F, G=None, w_in=None, known=None, signed=()):
        c = Circuit()
        self.F = F
        self.w_in = F + 2 if w_in is None else w_in       # |v_i| < 2
        self.v = [c.alloc(self.w_in) for _ in range(2 ** L)]
        self.tree = Tree(c, self.v, F, G, known, signed)
        self.n_arith = len(c.gates)
        c.gates += reverse(c.gates)
        self.c = c

    def simulate(self, V):
        """V: (B, m). Returns prepared amplitudes (B, m) and checks."""
        V = np.asarray(V, float)
        B = V.shape[0]
        Z = np.round(V * 2 ** self.F).astype(np.int64)
        sim = BitSim(self.c.n, B)
        for l, r in enumerate(self.v):
            sim.set_int(r, Z[:, l] & ((1 << self.w_in) - 1))
        sim.run(self.c.gates[:self.tree.gate_range[0]])
        ok = self.tree.run_checked(sim, self.c.gates, self.tree.gate_range[0])
        th = self.tree.angles(sim)
        amp = self.tree.amplitudes(th, B)
        sim.run(self.c.gates[self.n_arith:])
        dirty = _dirty_bits(sim, self.c.n, {w for r in self.v for w in r})
        restored = all(np.all(sim.get_int(r) == Z[:, l]) for l, r in enumerate(self.v))
        return amp, dict(work_bits_left_set=dirty, inputs_restored=restored,
                         no_overflow=ok)
