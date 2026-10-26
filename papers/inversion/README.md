# Preconditioned inverse (papers/inversion)

Scripts for the paper *A Preconditioned Quantum Linear Solver for Periodic Two-Phase Heat Conduction with
Mesh-Independent Condition Number*. The preconditioned operator `M = V^T Sigmatilde V` is block-encoded
with subnormalization 1 and condition number at most `rho` for every mesh size, and QSVT with the odd
polynomial `q_t` of degree `d_t = 2t - 1` block-encodes its inverse. The code is
`quantumpoissonhomogenization.inversion` and the modules it uses.

| command | reproduces | time |
|---|---|---|
| `python makeTables.py` | the six tables: `options`, `geometry`, `be_M`, `inverse`, `qiskit`, `resources` | about 3 minutes |
| `python makeTables.py inverse` | one table | |
| `python makeTables.py listings` | Listings 1 to 3 of Sec. 4, output compared with the paper | about 20 seconds |
| `python makeTables.py checks` | the seven checks below | about 10 minutes |
| `python makeTables.py symbol` | Sec. 2.2: the closed-form symbol of `K` against assembly | seconds |
| `python makeTables.py isometry` | Secs. 3.5, 3.8: factorization, the circuit gate by gate, error and Toffoli count against N and F | about 30 seconds |
| `python makeTables.py full_unitary` | Sec. 3.8: encoding error `delta` in 42 cases and the exact unitary of Prop. gate (iii) | a few minutes |
| `python makeTables.py qsvt_degree` | Sec. 3.7: degree and sup norm of `q_t` | seconds |
| `python makeTables.py square` | Sec. 3.6: condition numbers and the gate-level encoding of `Mtilde` | about 4 minutes |
| `python makeTables.py inverse_dense` | Sec. 3.7: the inverse on four heat sources, dense | about 1 minute |
| `python makeTables.py qiskit_check` | Sec. 5.5: the inverse as one Qiskit circuit, N = 4 and 8, every stage against its matrix | about 2.5 minutes |
| `python makeFigures.py` | `spectrum`, `cells`, `mesh` | about 10 minutes |
| `python makeFigures.py qsvt_poly angles` | figures the paper does not use | |

The baseline condition number of Sec. 5.2 uses the subnormalization `alpha = 16/3` of the encoding of
`K^chi / sigma_max` (`quantumpoissonhomogenization.cells.ALPHA`); when pyblockencode is installed,
`python makeTables.py square` cross-checks it.
