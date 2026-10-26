# Multiplexed state preparation (papers/multiplexed)

Scripts for the paper *Multiplexed state preparation by CORDIC arithmetic for the periodic Poisson
isometry*. For every wavenumber k != 0, the circuit prepares c(k)/||c(k)||,
c = (a, b, zeta/sqrt3, zeta/sqrt3), a = sin(phi_1) cos(phi_2), b = cos(phi_1) sin(phi_2),
zeta = sin(phi_1) sin(phi_2), phi_j = pi k_j / N: CORDIC rotation mode computes the entries from the
bits of k (`SymbolVector`), and two CORDIC vectorings and one constant product convert them into
amplitudes (`Tree`). The code is `quantumpoissonhomogenization.arithmetic`.

| command | reproduces | time |
|---|---|---|
| `python makeTables.py` | both tables below | a few minutes |
| `python makeTables.py options` | Sec. 5, Table 3 (`tables/options.tex`): the circuit against the multiplexor and the QROM lookup, N = 8 to 4096, with the crossovers asserted | about 1 second |
| `python makeTables.py cost` | Theorem 1 and Table 1 (cost by stage at N = 1024, F = 45); Sec. 4, the error in the 42 cases and the bound of Corollary 6; the block against the Q4 assembly | a few minutes |
| `python makeTables.py checks` | the four checks below | about 30 seconds |
| `python makeTables.py worked` | Secs. 2 and 3, the worked example at k = (3, 1), N = 16, F = 12 | seconds |
| `python makeTables.py rotation` | Lemma 2 (rotation mode) on random angles | seconds |
| `python makeTables.py lemma` | Lemma 3 (one vectoring) on random integer vectorings | seconds |
| `python makeTables.py counts` | Sec. 5: Toffoli and work-qubit counts against the gate lists | about 10 seconds |
| `python makeFigures.py` | Sec. 4, Figure 4 (`figs/error.pdf`) | about 15 seconds |
