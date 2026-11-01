# Multiplexed state preparation (papers/multiplexed)

`tutorial.ipynb` walks through Listings 1 to 3 of Sec. 6, each with its purpose and the output
printed in the paper: the circuit used as a library, then one wavenumber in its registers, then its
cost against the two standard constructions. It is generated from the same constants the `listings`
check asserts, so run `makeTables.py tutorial` after changing a listing rather than editing the
notebook.

Scripts for the paper *Multiplexed state preparation by CORDIC arithmetic for the periodic Poisson
isometry*. For every wavenumber k != 0, the circuit prepares c(k)/||c(k)||,
c = (a, b, zeta/sqrt3, zeta/sqrt3), a = sin(phi_1) cos(phi_2), b = cos(phi_1) sin(phi_2),
zeta = sin(phi_1) sin(phi_2), phi_j = pi k_j / N: CORDIC rotation mode computes the entries from the
bits of k (`SymbolVector`), and two CORDIC vectorings and one constant product convert them into
amplitudes (`Tree`). The code is `quantumpoissonhomogenization.arithmetic`.

| command | reproduces | time |
|---|---|---|
| `python makeTables.py` | both tables below | about 15 seconds |
| `python makeTables.py options` | Sec. 5, Table 3 (`tables/options.tex`): the circuit against the multiplexor and the QROM lookup, N = 8 to 4096, with the crossovers, the growth per doubling of N, and the share of rotation synthesis asserted | about 1 second |
| `python makeTables.py cost` | Theorem 2 and Table 2 (cost by stage at N = 1024, F = 45); Sec. 4, the error in the 42 cases (at most 6 N 2^-F); the abstract at N = 32, F = 32 (23,089 Toffoli gates, 65 controlled rotations, error 9.0e-9); the block against the Q4 assembly | about 15 seconds |
| `python makeTables.py listings` | Listings 1 to 3 of Sec. 6, output compared with the paper | a few seconds |
| `python makeTables.py tutorial` | regenerates `tutorial.ipynb` from Listings 1 to 3 | seconds |
| `python makeTables.py checks` | the four checks below | about 30 seconds |
| `python makeTables.py worked` | Secs. 2 and 3, the worked example at k = (3, 1), N = 16, F = 12 | seconds |
| `python makeTables.py rotation` | Proposition 1, first step: the rotation-mode bound on random angles | seconds |
| `python makeTables.py lemma` | Proposition 1, third step: the vectoring bounds on random integer vectorings | seconds |
| `python makeTables.py counts` | Sec. 5: Toffoli and work-qubit counts against the gate lists | about 10 seconds |
| `python makeFigures.py` | Sec. 4, Figure 4 (`figs/error.pdf`) | about 15 seconds |
