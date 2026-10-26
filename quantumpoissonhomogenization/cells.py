"""cells -- the three periodic two-phase test cells, applied matrix-free, shared by all papers.

    square     centered square inclusion of side N/2, v_f = 1/4 (isolated conductor; worked example)
    cross      two orthogonal bands of width N/8 through the center, v_f = 15/64 (connected)
    laminate   chi = 1 for x < 1/2, v_f = 1/2 (layers normal to x)

Each cell has a material oracle controlled(-R_chi) of O(1) gates, checked on every element. Every
operator is applied matrix-free (FFT for functions of K, sparse A = W^{1/2} G), so the cells reach
N = 1024 (N_dof ~ 10^6). sigma_max = 1 throughout.
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .fourier_symbol import symbol_function, apply_mode_function
from .isometry import _B, _CORNERS

ALPHA = 16 / 3            # subnormalization of the encoding of K^chi / sigma_max (inversion paper)
GEOMETRIES = ("square", "cross", "laminate")


def k_function(f, N):
    """x -> f(K) x by FFT, zero on the constant field."""
    T = symbol_function(f, N)
    return lambda x: apply_mode_function(T, x)


def square_chi(N):
    """Centered square of side N/2: v_f = 1/4. chi[e] = 1 in the inclusion."""
    i, j = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    return ((i >= N // 4) & (i < 3 * N // 4) & (j >= N // 4) & (j < 3 * N // 4)).ravel().astype(float)


def assemble(N):
    """Sparse A = W^{1/2} G (8 N^2 x N^2), row 8 e + 2 g + c, element e = i N + j."""
    h = 1 / N
    Bs = [_B(g, h) for g in range(4)]
    rows, cols, vals = [], [], []
    for i in range(N):
        for j in range(N):
            e = i * N + j
            for g in range(4):
                for a, (cx, cy) in enumerate(_CORNERS):
                    n = ((i + cx) % N) * N + (j + cy) % N
                    for c in range(2):
                        rows.append(8 * e + 2 * g + c); cols.append(n); vals.append(Bs[g][c, a])
    G = sp.csr_matrix((vals, (rows, cols)), shape=(8 * N * N, N * N))
    return (h / 2) * G                                         # sqrt(w) = h / 2


def e_hat(N, a):
    """W^{1/2} ebar_a: sqrt(w) = 1/(2N) at rows 8 e + 2 g + (a - 1)."""
    v = np.zeros(8 * N * N)
    v[(a - 1)::2] = 1 / (2 * N)
    return v


# ==========================================================================
# 1. geometries and their material oracles
# ==========================================================================
def chi(name, N):
    """Phase indicator per element e = i N + j (i along x): 1 in the inclusion."""
    i, j = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    if name == "square":
        c = (i >= N // 4) & (i < 3 * N // 4) & (j >= N // 4) & (j < 3 * N // 4)
    elif name == "cross":
        band = lambda k: (k >= 7 * N // 16) & (k < 9 * N // 16)
        c = band(i) | band(j)
    elif name == "laminate":
        c = i < N // 2
    else:
        raise ValueError(name)
    return c.ravel().astype(float)


def oracle_network(name):
    """Gate list of controlled(-R_chi) on (s, i bits, j bits, work bits).

    -R_chi is +1 in the inclusion and -1 in the matrix, so its controlled form
    is the phase (-1)^{s (1 - chi)} = (-1)^s (-1)^{s chi}. Only the leading
    bits of each coordinate enter: b0 = most significant, b1, b2, b3. Gates:
    ("x", t), ("z", t), ("cx", c, t), ("cz", a, b), ("ccx", a, b, t, pa, pb)
    with control polarities pa, pb (1 = filled, 0 = open), ("ccz", a, b, c).
    """
    if name == "laminate":                      # chi = not i0: (-1)^{s (1 - chi)} = CZ(s, i0)
        return [("cz", "s", "i0")]
    g = [("z", "s")]
    if name == "square":                        # chi = (i0 xor i1) and (j0 xor j1)
        comp = [("cx", "i0", "i1"), ("cx", "j0", "j1"), ("ccx", "i1", "j1", "t", 1, 1)]
        return g + comp + [("cz", "s", "t")] + comp[::-1]
    if name == "cross":                         # band: top four bits in {0111, 1000}
        comp = []
        for c in "ij":
            comp += [("cx", f"{c}2", f"{c}3"), ("cx", f"{c}1", f"{c}2"), ("cx", f"{c}0", f"{c}1")]
            # now c1 = b0^b1 (x), c2 = b1^b2 (y), c3 = b2^b3 (z): band = x & !y & !z
            comp += [("ccx", f"{c}1", f"{c}2", f"u{c}", 1, 0),
                     ("ccx", f"u{c}", f"{c}3", f"a{c}", 1, 0)]
        # (-1)^{s (A or B)} = CZ(s,A) CZ(s,B) CCZ(s,A,B)
        mid = [("cz", "s", "ai"), ("cz", "s", "aj"), ("ccz", "s", "ai", "aj")]
        return g + comp + mid + comp[::-1]
    raise ValueError(name)


def oracle_phase(name, n):
    """Simulate oracle_network on every (s, position): returns the phase table and counts."""
    N = 2 ** n
    i, j = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i, j = i.ravel(), j.ravel()
    out = {}
    for s in (0, 1):
        bits = {"s": np.full(N * N, s)}
        for k in range(4):                       # leading bits, b0 most significant
            bits[f"i{k}"] = (i >> (n - 1 - k)) & 1 if k < n else np.zeros_like(i)
            bits[f"j{k}"] = (j >> (n - 1 - k)) & 1 if k < n else np.zeros_like(j)
        phase = np.ones(N * N)
        for gate in oracle_network(name):
            for w in gate[1:4]:
                if isinstance(w, str):
                    bits.setdefault(w, np.zeros(N * N, int))
            op = gate[0]
            if op == "z":
                phase *= (-1.0) ** bits[gate[1]]
            elif op == "cz":
                phase *= (-1.0) ** (bits[gate[1]] & bits[gate[2]])
            elif op == "ccz":
                phase *= (-1.0) ** (bits[gate[1]] & bits[gate[2]] & bits[gate[3]])
            elif op == "cx":
                bits[gate[2]] = bits[gate[2]] ^ bits[gate[1]]
            elif op == "ccx":
                a, b, t, pa, pb = gate[1:]
                bits[t] = bits[t] ^ ((bits[a] == pa) & (bits[b] == pb)).astype(int)
        work = [w for w in bits if w[0] in "uat" and w != "s"]
        assert all(not bits[w].any() for w in work), name                  # work bits clean
        for k in range(min(4, n)):                                           # inputs restored
            assert np.array_equal(bits[f"i{k}"], (i >> (n - 1 - k)) & 1)
        out[s] = phase
    net = oracle_network(name)
    counts = dict(toffoli=sum(g[0] in ("ccx", "ccz") for g in net),
                  cnot=sum(g[0] in ("cx", "cz") for g in net),
                  single=sum(g[0] in ("x", "z") for g in net))
    return out, counts


_ORC = {}


def check_oracle(name):
    """controlled(-R_chi) acts as I for s = 0 and 2 chi - 1 for s = 1, at N = 16 ... 256."""
    if name in _ORC:
        return _ORC[name]
    for n in range(4, 9):
        ph, counts = oracle_phase(name, n)
        assert np.all(ph[0] == 1), name
        assert np.array_equal(ph[1], 2 * chi(name, 2 ** n) - 1), (name, n)
    _ORC[name] = counts
    return counts


# ==========================================================================
# 2. operators, matrix-free
# ==========================================================================
_A = {}


def A_matrix(N):
    """A = W^{1/2} G (8 N^2 x N^2), vectorized; equals verify_square.assemble."""
    if N in _A:
        return _A[N]
    h = 1 / N
    i, j = np.divmod(np.arange(N * N), N)
    rows, cols, vals = [], [], []
    for g in range(4):
        Bg = _B(g, h)
        for a, (cx, cy) in enumerate(_CORNERS):
            node = ((i + cx) % N) * N + (j + cy) % N
            for c in range(2):
                rows.append(8 * np.arange(N * N) + 2 * g + c)
                cols.append(node)
                vals.append(np.full(N * N, Bg[c, a]))
    A = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(8 * N * N, N * N)) * (h / 2)
    _A[N] = A
    return A


class Cell:
    """Two-phase cell: N, rho, geometry; Mtilde, K^chi and the cell-problem source, matrix-free."""

    def __init__(self, name, N, rho):
        self.name, self.N, self.rho = name, N, rho
        self.A = A_matrix(N)
        self.chi = chi(name, N)
        self.vf = self.chi.mean()
        self.St = np.repeat(np.where(self.chi == 1, 1.0, 1 / rho), 8)       # Sigma / sigma_max
        self.sbar = self.vf + (1 - self.vf) / rho                          # Voigt bound
        self.reuss = 1 / (self.vf + (1 - self.vf) * rho)                   # Reuss bound
        self.Kih = k_function(lambda e: e ** -0.5, N)
        self.Kinv = k_function(lambda e: 1 / e, N)
        self.n = N * N

    # Mtilde = M + sbar Pi_0, M = V^T Sigmatilde V, V = A (K^+)^{1/2}
    def Mt(self, v):
        y = self.Kih(v)
        out = self.Kih(self.A.T @ (self.St * (self.A @ y)))
        return out + self.sbar * v.mean() * np.ones_like(v)

    def Mt_op(self):
        return spla.LinearOperator((self.n, self.n), matvec=lambda v: self.Mt(np.ravel(v)), dtype=float)

    def Kc(self, u):                                   # K^chi / sigma_max
        return self.A.T @ (self.St * (self.A @ u))

    def ehat(self, a):
        v = np.zeros(8 * self.n)
        v[(a - 1)::2] = 1 / (2 * self.N)
        return v


def pcg(Aop, Pop, b, tol, x_ref=None, energy_tol=None, maxiter=5000):
    """Preconditioned CG on the complement of the constants. Returns (x, iterations).
    With x_ref and energy_tol, stops when the relative A-norm error reaches energy_tol."""
    b = b - b.mean()
    x = np.zeros_like(b)
    r = b.copy()
    z = Pop(r); z -= z.mean()
    p = z.copy()
    rz = r @ z
    nb = np.sqrt(b @ Pop(b))
    ref_norm = None if x_ref is None else np.sqrt(x_ref @ Aop(x_ref))
    for it in range(1, maxiter + 1):
        Ap = Aop(p)
        al = rz / (p @ Ap)
        x += al * p
        r -= al * Ap
        if x_ref is not None:
            e = x - x_ref
            if np.sqrt(e @ Aop(e)) <= energy_tol * ref_norm:
                return x, it
        z = Pop(r); z -= z.mean()
        rz_new = r @ z
        if x_ref is None and np.sqrt(abs(rz_new)) <= tol * nb:
            return x, it
        p = z + (rz_new / rz) * p
        rz = rz_new
    raise RuntimeError("pcg did not converge")


def extreme_eigs(cell, tol=1e-9):
    """Smallest and largest eigenvalues of Mtilde (Lanczos)."""
    op = cell.Mt_op()
    lo = spla.eigsh(op, k=1, which="SA", tol=tol, return_eigenvectors=False)[0]
    hi = spla.eigsh(op, k=1, which="LA", tol=tol, return_eigenvectors=False)[0]
    return lo, hi


def lambda_min_Kc(cell, tol=1e-9, seed=0):
    """Smallest nonzero eigenvalue of K^chi / sigma_max by LOBPCG, preconditioned by K^+,
    with the constant field as constraint."""
    n = cell.n
    Aop = spla.LinearOperator((n, n), matvec=lambda v: cell.Kc(np.ravel(v)), dtype=float)
    P = spla.LinearOperator((n, n), matvec=lambda v: cell.Kinv(np.ravel(v)), dtype=float)
    X = np.random.default_rng(seed).standard_normal((n, 1))
    X -= X.mean()
    lam, _ = spla.lobpcg(Aop, X, M=P, Y=np.ones((n, 1)), tol=tol, maxiter=500, largest=False)
    return float(lam[0])


def sigma_fe(cell, a=1):
    """sigma^H_aa / sigma_max from the finite element cell problem (PCG preconditioned by K^+)."""
    e = cell.ehat(a)
    u, _ = pcg(cell.Kc, cell.Kinv, -(cell.A.T @ (cell.St * e)), 1e-13)
    r = e + cell.A @ u
    return float(r @ (cell.St * r))
