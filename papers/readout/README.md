# Least-squares readout (papers/readout)

Scripts for the paper *Quantum Readout of the Effective Conductivity of Periodic Two-Phase Media by a
Least-Squares Range Projection*. With `g = Sigmatilde^{1/2} ehat` and `Vs = Sigmatilde^{1/2} V`,
`sigma^H / sigma_max = min_w |g + Vs w|^2 = g^T (I - P) g`; the even polynomial `f_t` of degree
`d_f <= sqrt(rho) ln(2/eps_p) + 2` approximates `I - P`, and amplitude estimation reads out
`A = g^T f_t(Vs Vs^T) g`. The code is `quantumpoissonhomogenization.readout` and the modules it uses.

| command | reproduces | time |
|---|---|---|
| `python makeTables.py` | the six tables: `ls`, `mesh`, `qiskit`, `arith`, `ae`, `cost` | about 40 seconds |
| `python makeTables.py qiskit` | one table | |
| `python makeTables.py listings` | Listings 1 and 2 of Sec. 4, output compared with the paper | about 20 seconds |
| `python makeFigures.py` | `cells`, `poly` | seconds |
