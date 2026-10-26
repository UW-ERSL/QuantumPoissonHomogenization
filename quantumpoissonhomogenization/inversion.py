"""inversion -- the inverse of the preconditioned operator (inversion paper): matrix-free, the resources
of the gate-level circuit, the dense references of its checks, and the inverse as one Qiskit circuit.

The odd QSVT polynomial q_t at kappa = rho, t = ceil(rho/2 ln(2/eps_p)), degree d = 2t - 1, applied
to Mtilde by the Clenshaw recurrence on its Chebyshev interpolant; the choice of F; the resource
counts; and the cost estimates of the three constructions of the wavenumber-controlled stage.
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from numpy.polynomial import chebyshev as Cheb
from qiskit.quantum_info import Operator, Statevector

from .fourier_symbol import symbol
from .isometry import IsometryCircuit, T_matrix, vhat_exact, end_to_end_error
from .polynomials import q_poly, scaled_poly, wx_phases, reflection_phases
from .cells import pcg, check_oracle, square_chi, assemble, k_function, e_hat
from .circuits import q_mux, inverse_circuit, counts
from .emulation import _plane_rotation_minus_identity

DELTA_CONST = 6           # delta <= 6 N 2^-F (measured, inversion paper Sec. 3.8)


# ==========================================================================
# 3. the QSVT inverse polynomial applied matrix-free (Sec. 3.7)
# ==========================================================================
# q_t is the odd polynomial of Sec. 2.4 at kappa = rho with t = ceil(rho/2 ln(2/eps_p)) (Sec. 3.7),
# degree d = 2t - 1. It is applied to Mtilde by the Clenshaw recurrence on its Chebyshev interpolant of
# degree d, which equals q_t up to rounding, with d applications of Mtilde.
_PLAN = {}


def inverse_plan(rho, eps_p):
    """q_t, its degree d, its Chebyshev coefficients, and C = max_[-1,1] |q_t| (Eq. qsvt-norm)."""
    if (rho, eps_p) not in _PLAN:
        q, d = q_poly(rho, eps_p)
        x = np.linspace(1e-9, 1.0, 2_000_001)
        C = float(np.abs(q(x)).max())
        _PLAN[rho, eps_p] = (q, d, Cheb.chebinterpolate(q, d), C)
    return _PLAN[rho, eps_p]


def encoding_bits(N, rho, eps_p):
    """Sec. 3.8: delta = eps_p / (2 d C), so that the encoding error adds at most eps_p to the energy-norm
    error, and F = ceil(log2(6 N / delta)) by Eq. (delta-bound)."""
    _, d, _, C = inverse_plan(rho, eps_p)
    delta = eps_p / (2 * d * C)
    return delta, int(np.ceil(np.log2(DELTA_CONST * N / delta)))


def apply_q(cell, coef, b):
    """q_t(Mtilde) b by the Clenshaw recurrence."""
    b1 = np.zeros_like(b); b2 = np.zeros_like(b)
    for c in coef[:0:-1]:
        b1, b2 = 2 * cell.Mt(b1) - b2 + c * b, b1
    return cell.Mt(b1) - b2 + coef[0] * b


def inverse_error(cell, eps_p, a=1):
    """Relative energy-norm error of u_Q = (K^+)^{1/2} q_t(Mtilde) (K^+)^{1/2} f against the cell solution,
    for the cell-problem source f of the macroscopic gradient e_a (Eq. energy-error), sigma_max = 1."""
    _, d, coef, _ = inverse_plan(cell.rho, eps_p)
    f = -(cell.A.T @ (cell.St * cell.ehat(a)))
    u, _ = pcg(cell.Kc, cell.Kinv, f, 1e-13)
    uq = cell.Kih(apply_q(cell, coef, cell.Kih(f)))
    e = uq - u
    return float(np.sqrt((e @ cell.Kc(e)) / (u @ cell.Kc(u)))), d


# ==========================================================================
# 4. resource counts
# ==========================================================================
_UV = {}


def uv_counts(N, F):
    """Toffoli count, controlled rotations and qubits of one application of U_V (gate level, cached)."""
    if (N, F) not in _UV:
        n = int(np.log2(N))
        r = IsometryCircuit(n=n, F=F).resources()
        _UV[N, F] = dict(toffoli=r["toffoli_total"],
                         cordic_rot=r["controlled_rotations"] + r["fixed_rotations"],
                         qft_rot=2 * 2 * n * (n - 1) // 2, phase_rot=r["phase_gates"],
                         qubits=r["wires"])
    return _UV[N, F]


def inverse_resources(name, N, rho, eps_p):
    """Sec. 3.7: one query to U_M is U_V, U_Sigma (one controlled R_chi), U_V^dag; d queries."""
    _, d, _, C = inverse_plan(rho, eps_p)
    delta, F = encoding_bits(N, rho, eps_p)
    uv = uv_counts(N, F)
    t_rchi = check_oracle(name)["toffoli"]
    per_query = 2 * uv["toffoli"] + t_rchi
    # two U_V (CORDIC rotations, QFT controlled phases, the 2n phase gates of
    # e^{i(phi1+phi2)}), and the two Ry of U_Sigma
    rot = 2 * (uv["cordic_rot"] + uv["qft_rot"] + uv["phase_rot"]) + 2
    return dict(name=name, N=N, rho=rho, eps_p=eps_p, d=d, C=C, delta=delta, F=F,
                toffoli_uv=uv["toffoli"], toffoli_per_query=per_query, rot_per_query=rot,
                toffoli_total=float(d) * per_query,
                qubits=uv["qubits"] + 3)    # + the U_Sigma ancilla, the flag qubit, the QSVT ancilla


# ==========================================================================
# 5. three constructions of the wavenumber-controlled stage (Sec. 3.8, tab:options)
# ==========================================================================
T_PER_TOFFOLI = 4         # Jones (2013)


def t_per_rotation(eps):
    """T gates of one single-qubit rotation to accuracy eps (Ross and Selinger, leading term)."""
    return 3 * np.log2(1 / eps)


def _best_k(d, M):
    """Power of two k minimizing ceil(d / k) + M (k - 1), and that minimum (Berry et al. 2019, Sec. 3.2)."""
    ks = [2 ** j for j in range(int(np.log2(d)) + 1)]
    cost = [int(np.ceil(d / k)) + M * (k - 1) for k in ks]
    i = int(np.argmin(cost))
    return ks[i], cost[i]


def stage_options(N, rho, eps_p):
    """Estimated cost of one application of the wavenumber-controlled stage for the three
    constructions of Sec. 3.8, at the encoding error delta = eps_p / (2 d C) of Eq. (F-choice).

    mux     standard state preparation multiplexed over k (Mottonen et al.): two uniformly
            controlled R_y gates, N^2 and 2 N^2 rotations at angles tabulated classically.
    lookup  QROM lookup of the two tabulated angles to b bits each (Berry et al. 2019: compute
            ceil(d/k1) + M (k1 - 1), measurement-based uncompute ceil(d/k2) + k2, d = N^2,
            M = 2b, (k1 - 1) M clean ancillas), then the rotations by phase-gradient addition,
            estimated as 3b Toffoli gates (one b-bit addition per rotation and one selection of
            the omega register by t_0); the phase-gradient state is prepared once.
    arith   the CORDIC circuit of Prop. gate (gate-level counts).
    All three share the zero-mode flag, 2 (2n - 1) Toffoli gates. T gates: T_PER_TOFFOLI per
    Toffoli and t_per_rotation(delta / R) for each of R synthesized rotations, so that the
    synthesis adds at most delta. Counts other than `arith` are estimates from published cost
    formulas, not gate-level circuits."""
    n = int(np.log2(N))
    delta, F = encoding_bits(N, rho, eps_p)
    zero = 2 * (2 * n - 1)
    out = {}
    R = 3 * N * N
    out["mux"] = dict(toffoli=zero, rotations=R, work=2 * n - 1)
    b = int(np.ceil(np.log2(2 * np.pi / delta)))           # two angles, each rounded to pi 2^-b
    d, M = N * N, 2 * b
    k1, c1 = _best_k(d, M)
    k2 = 2 ** int(np.argmin([np.ceil(d / 2 ** j) + 2 ** j for j in range(n + n + 1)]))
    c2 = int(np.ceil(d / k2)) + k2
    out["lookup"] = dict(toffoli=zero + c1 + c2 + 3 * b, rotations=0, work=k1 * M + 2 * b, b=b, k1=k1)
    r = IsometryCircuit(n=n, F=F).resources()
    out["arith"] = dict(toffoli=r["toffoli_total"],
                        rotations=r["controlled_rotations"] + r["fixed_rotations"], work=r["work"])
    for v in out.values():
        v["T"] = T_PER_TOFFOLI * v["toffoli"] + (v["rotations"] * t_per_rotation(delta / v["rotations"])
                                                 if v["rotations"] else 0.0)
    return dict(N=N, F=F, delta=delta, **out)


# ==========================================================================
# 6. dense references for the checks of the inversion paper (N <= 64)
# ==========================================================================
def circulant(f, N):
    """Dense circulant matrix f(K), zero on the constant field; node n = i N + j."""
    S = symbol(N)
    T = np.where(S > 1e-12, f(np.clip(S, 1e-300, None)), 0.0)
    ker = np.fft.ifft2(T).real
    i, j = np.divmod(np.arange(N * N), N)
    return ker[(i[:, None] - i[None, :]) % N, (j[:, None] - j[None, :]) % N]


def operators(N, rho):
    A = assemble(N).tocsr()
    chi = square_chi(N)
    St = np.repeat(np.where(chi == 1, 1.0, 1 / rho), 8)       # Sigma / sigma_max
    K = (A.T @ A).toarray()
    Kc = (A.T @ sp.diags(St) @ A).toarray()                    # K^chi / sigma_max
    return A, St, K, Kc, chi


_KIH = {}


def spectra(N, rho):
    A, St, K, Kc, chi = operators(N, rho)
    if N not in _KIH:
        _KIH[N] = circulant(lambda e: e ** -0.5, N)
        assert np.abs(circulant(lambda e: e, N) - K).max() < 1e-10 * np.abs(K).max()
    Kih = _KIH[N]
    lm = np.linalg.eigvalsh(Kih @ Kc @ Kih)
    lm = lm[np.abs(lm) > 1e-9]
    lk = np.linalg.eigvalsh(Kc)
    lk = lk[lk > 1e-9 * lk.max()]
    return dict(N=N, rho=rho, Mmin=float(lm.min()), Mmax=float(lm.max()),
                kappaM=float(lm.max() / lm.min()),
                Kmax=float(lk.max()), Kmin=float(lk.min()), kappaK=float(lk.max() / lk.min()))


def qsvt_degree(kappa, eps):
    """Degree 2t - 1 of the odd QSVT polynomial with relative residual eps on [1/kappa, 1]."""
    a = 1 / kappa ** 2
    t = int(np.ceil(np.arccosh(1 / eps) / np.arccosh((1 + a) / (1 - a))))
    return 2 * t - 1


def circuit_block(N, rho, F):
    """Block of U_V^dag U_Sigma U_V on the input subspace, from the gate-level circuit."""
    chi = square_chi(N)
    circ = IsometryCircuit(int(np.log2(N)), F=F)
    Q, chk = circ.simulate()
    F1 = np.exp(-2j * np.pi * np.outer(np.arange(N), np.arange(N)) / N) / np.sqrt(N)
    Fm = np.kron(F1, F1)
    BD = np.zeros((16 * N * N, N * N), complex)
    for k in range(N * N):
        BD[16 * k:16 * k + 16, k] = Q[k, :, 0]                 # all 16 rows
    X = np.kron(Fm.conj().T, np.eye(16)) @ BD @ Fm              # U_V on the input subspace
    Se = np.repeat(np.where(chi == 1, 1.0, 1 / rho), 16)        # U_Sigma block: Sigma_e x I_16
    return X.conj().T @ (Se[:, None] * X), chk


def prop4_check(N, rho, F):
    """Block from the gate-level circuit against Mtilde = M + sbar Pi_0."""
    A, St, K, Kc, chi = operators(N, rho)
    Kih = circulant(lambda e: e ** -0.5, N)
    M = Kih @ Kc @ Kih
    sbar = chi.mean() + (1 - chi.mean()) / rho
    Mt = M + sbar * np.ones((N * N, N * N)) / N ** 2
    Mc, chk = circuit_block(N, rho, F)
    lm = np.linalg.eigvalsh((Mc + Mc.conj().T) / 2)
    return dict(N=N, rho=rho, F=F, err=float(np.linalg.norm(Mc - Mt, 2)),
                spec_min=float(lm.min()), spec_max=float(lm.max()), sbar=float(sbar),
                clean=chk["work_bits_left_set"] == 0)


def alpha_check():
    """Cross-check ALPHA against pyblockencode, when it is installed."""
    try:
        from pyblockencode import POISSON2D_2PHASE
    except ImportError:
        return None
    return POISSON2D_2PHASE(vf=0.25, E1=1.0, E2=0.1).alpha()


# ---- sources, solutions, errors ---------------------------------------------
def zero_mean(f):
    return f - f.mean()


def sources(N, A, St, Kh, lam, U):
    n = N * N
    i, j = np.divmod(np.arange(n), N)
    x, y = i / N, j / N
    wd = np.tile(np.array([1.0, 0.0]) / (2 * N), 4 * n)        # sqrt(w) e_x at every Gauss point
    nz = lam > 1e-9
    f = {"smooth": np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y),
         "random": np.random.default_rng(0).standard_normal(n),
         "cell": -A.T @ (St * wd),                              # macroscopic gradient (1, 0)
         "worst": Kh(U[:, nz][:, np.argmin(lam[nz])])}          # eigenvector at 1/rho
    return {k: zero_mean(v) for k, v in f.items()}


def fe_solve(Kc, f):
    keep = np.arange(1, Kc.shape[0])                            # pin node 0, then remove the mean
    u = np.zeros_like(f)
    u[keep] = spla.spsolve(Kc[keep][:, keep].tocsc(), f[keep])
    return zero_mean(u)


def energy_error(Kc, u, x):
    e = x - u
    return np.sqrt(e @ (Kc @ e) / (u @ (Kc @ u)))


def apply_inverse(q, Kih, lam, U, sbar, f):
    lt = np.where(np.abs(lam) > 1e-9, lam, sbar)                # spectrum of Mtilde
    return Kih(U @ (q(lt) * (U.conj().T @ Kih(f))))


LOADS = ("smooth", "random", "cell", "worst")


def inverse_case(N, rho, eps, F_bits=()):
    """Energy-norm errors of u_Q for the four sources, the QSVT degree d, and
    the errors with Mtilde from the gate-level circuit for each F in F_bits."""
    q, d = q_poly(rho, eps)
    A = assemble(N).tocsr()
    chi = square_chi(N)
    St = np.repeat(np.where(chi == 1, 1.0, 1 / rho), 8)
    Kc = (A.T @ sp.diags(St) @ A).tocsr()                       # K^chi / sigma_max
    Kih, Kh = (k_function(g, N) for g in (lambda e: e ** -0.5, np.sqrt))
    M = Kih(Kc @ Kih(np.eye(N * N)))
    lam, U = np.linalg.eigh((M + M.T) / 2)
    sbar = chi.mean() + (1 - chi.mean()) / rho
    F_ = sources(N, A, St, Kh, lam, U)
    sols = {k: fe_solve(Kc, f) for k, f in F_.items()}
    errs = {k: energy_error(Kc, sols[k], apply_inverse(q, Kih, lam, U, sbar, F_[k])) for k in LOADS}
    circ = {}
    for Fb in F_bits:
        Mc, chk = circuit_block(N, rho, Fb)
        lc, Uc = np.linalg.eigh((Mc + Mc.conj().T) / 2)
        circ[Fb] = {k: energy_error(Kc, sols[k], apply_inverse(q, Kih, lc, Uc, sbar,
                                                               F_[k].astype(complex)).real) for k in LOADS}
    return dict(d=d, errs=errs, circuit=circ)


def residual(rho, eps):
    """Largest |x q_t(x) - 1| on [1/rho, 1], that is, 1/T_t(s)."""
    a = 1 / rho ** 2
    t = int(np.ceil(0.5 * rho * np.log(2 / eps)))
    return 1 / np.cosh(t * np.arccosh((1 + a) / (1 - a)))


def full_error(n, F, check_unitary=False):
    """(delta, checks) of U_V at N = 2^n and F bits; checks the exact unitary if asked."""
    c = IsometryCircuit(n=n, F=F); N = c.N
    Q, chk = c.simulate()
    T = T_matrix()
    delta, gap = 0.0, 0.0
    for kk in range(1, N * N):
        k1, k2 = divmod(kk, N)
        v = np.concatenate([vhat_exact(N, k1, k2), np.zeros(8)])
        q = Q[kk][:, 0]
        d = np.linalg.norm(q - v)
        delta = max(delta, d)
        if check_unitary:
            P = np.exp(1j * np.pi * (k1 + k2) / N) * T
            x, y = (P.conj().T @ q).real, (P.conj().T @ v).real
            D = P @ _plane_rotation_minus_identity(x, y) @ P.conj().T @ Q[kk]   # Q(k) - Q~(k)
            Qx = Q[kk] + D
            gap = max(gap, abs(np.linalg.norm(D, 2) - d) / max(d, 1e-300),
                      np.linalg.norm(Qx[:, 0] - v), np.abs(Qx.conj().T @ Qx - np.eye(16)).max())
    chk["zero_mode_outside"] = bool(np.linalg.norm(Q[0][:8, 0]) < 1e-14)
    chk["exact_unitary_gap"] = gap
    return delta, chk


C_BOUND = 6          # delta <= C_BOUND * N * 2^-F in every case below (largest measured: 5.73)


# ---- the gate-level U_V: factorization, circuit, scaling (Secs. 3.5, 3.8) ----------------
def factorization():
    T, worst = T_matrix(), 0.0
    for N in (8, 64):
        for k1 in range(N):
            for k2 in range(N):
                if k1 == k2 == 0:
                    continue
                f1, f2 = np.pi * k1 / N, np.pi * k2 / N
                a, b, z = np.sin(f1) * np.cos(f2), np.cos(f1) * np.sin(f2), np.sin(f1) * np.sin(f2)
                c = np.zeros(16); c[:4] = a, b, z / np.sqrt(3), z / np.sqrt(3)
                v = np.exp(1j * (f1 + f2)) * (T @ (c / np.linalg.norm(c)))
                worst = max(worst, np.abs(v[:8] - vhat_exact(N, k1, k2)).max(), np.abs(v[8:]).max())
    return worst


def circuit_checks():
    out = []
    for n in (3, 4):
        c = IsometryCircuit(n, F=24)
        Q, chk = c.simulate()
        q, e, _ = end_to_end_error(c, Q)
        zero = np.linalg.norm(Q[0][:8, 0])                 # Q(0)|0> in the states 8..15
        out.append((c.N, chk["work_bits_left_set"], chk["inputs_restored"], zero, q, e))
    return out


def scaling():
    rows = []
    for n in (4, 5, 6, 7):
        N = 2 ** n
        vh = np.array([vhat_exact(N, k1, k2) for k1 in range(N) for k2 in range(N)])
        for F in (16, 24, 32):
            c = IsometryCircuit(n, F=F)
            Q, chk = c.simulate()
            d = np.linalg.norm(Q[:, :8, 0] - vh, axis=1).max()
            r = c.resources()
            rows.append((N, F, d, chk["work_bits_left_set"] == 0, r["toffoli_total"], r["wires"]))
    return rows


# ==========================================================================
# 7. the inverse as one Qiskit circuit (Sec. 5.5), square, state vector
# ==========================================================================

def qiskit_references(N, rho):
    """Mtilde, M, psi = K^{+1/2} f, and K^{1/2} u for the finite element solution u."""
    chi = square_chi(N)
    A = assemble(N).tocsr()
    St = np.repeat(np.where(chi == 1, 1.0, 1 / rho), 8)
    sbar = chi.mean() + (1 - chi.mean()) / rho
    Kih = k_function(lambda e: e ** -0.5, N)
    Kh = k_function(lambda e: e ** 0.5, N)
    M = Kih((A.T @ sp.diags(St) @ A) @ Kih(np.eye(N * N)))
    M = (M + M.T) / 2
    Mt = M + sbar * np.full((N * N, N * N), 1 / (N * N))
    Kc = (A.T @ sp.diags(St) @ A).tocsc()                             # sigma_max = 1
    f = -A.T @ (St * e_hat(N, 1))                                     # cell problem, gradient (1, 0)
    keep = np.arange(1, N * N)                                        # pin node 0, then remove mean
    u = np.zeros(N * N); u[keep] = spla.spsolve(Kc[keep][:, keep], f[keep]); u -= u.mean()
    return Mt, M, Kih(f), Kh(u)


_PHASES = {}


def odd_phases(rho, eps_p):
    if (rho, eps_p) not in _PHASES:
        q, d, alpha, p = scaled_poly(rho, eps_p)
        _PHASES[rho, eps_p] = q, d, alpha, reflection_phases(wx_phases(p))
    return _PHASES[rho, eps_p]


def qiskit_inverse(N, rho=10.0, eps_p=1e-3):
    """Build the inverse at N, simulate it by state vector, and check it."""
    q, d, alpha, phi = odd_phases(rho, eps_p)
    qc, UV, UM, Uq, R = inverse_circuit(N, rho, phi)
    Mt, M, psi, v_fe = qiskit_references(N, rho)
    nb = N * N

    Q = Operator(q_mux(N)).data                                       # Q(k)|0> vs Vhat(k)
    err_Q = max(np.linalg.norm(Q[nb * np.arange(8) + k1 * N + k2, k1 * N + k2]
                               - vhat_exact(N, k1, k2))
                for k1 in range(N) for k2 in range(N) if k1 or k2)
    err_M = np.abs(Operator(UM).data[:nb, :nb] - Mt).max()

    a = Statevector(R).data[:nb]                                      # postselected rhs
    y = Statevector(qc).data[:nb]                                     # postselected solution
    s = np.vdot(psi, a) / np.vdot(psi, psi)                           # a = s psi
    rhs_err = np.linalg.norm(a - s * psi) / np.linalg.norm(a)
    lam, U = np.linalg.eigh(Mt)
    err_q = np.abs(y - (U * (q(lam) / alpha)) @ U.T @ a).max()
    eM = lambda z: np.sqrt(abs(np.real(np.conj(z) @ M @ z)))          # energy norm of K^{-1/2} z
    energy = eM(alpha * y - s * v_fe) / eM(s * v_fe)
    r = dict(N=N, d=d, alpha=alpha, err_Q=err_Q, err_M=err_M, rhs_err=rhs_err, err_q=err_q,
             energy=energy, p_rhs=np.linalg.norm(a) ** 2, p_success=np.linalg.norm(y) ** 2)
    for name, c in (("U_V", UV), ("U_M", UM), ("rhs", R), ("U_q", Uq), ("inverse", qc)):
        r[name] = counts(c)
    r["qubits"] = r["inverse"][0]
    return r
