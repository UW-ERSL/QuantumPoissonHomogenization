# QuantumPoissonHomogenization

Code for three papers by K. Suresh (UW-Madison) on quantum algorithms for periodic two-phase heat
conduction, discretized by bilinear (Q4) finite elements on a structured periodic mesh. One package,
`quantumpoissonhomogenization`, holds the code the papers share; each paper has its own folder under
`papers/` with the scripts that reproduce its tables, figures, and listings.

| folder | paper |
|---|---|
| `papers/multiplexed` | *Multiplexed state preparation by CORDIC arithmetic for the periodic Poisson isometry* |
| `papers/inversion` | *A Preconditioned Quantum Linear Solver for Periodic Two-Phase Heat Conduction with Mesh-Independent Condition Number* |
| `papers/readout` | *Quantum Readout of the Effective Conductivity of Periodic Two-Phase Media by a Least-Squares Range Projection* |

The three papers build on each other. The preconditioned cell operator
`M = (K^+)^{1/2} K^chi (K^+)^{1/2} / sigma_max` factors as `V^T Sigmatilde V`, where
`V = W^{1/2} G (K^+)^{1/2}` is a partial isometry; the multiplexed paper prepares the unit vector of
`V` at every wavenumber by fixed-point arithmetic, the inversion paper block-encodes `M` with
subnormalization 1 and condition number at most `rho` and inverts it by QSVT, and the readout paper
reads out the effective conductivity as a least-squares residual with `Vs = Sigmatilde^{1/2} V`.

## Install

    git clone https://github.com/UW-ERSL/QuantumPoissonHomogenization
    cd QuantumPoissonHomogenization
    python -m pip install -r requirements.txt       # installs the package in editable mode

The package needs numpy, scipy, and Qiskit; the paper scripts also need matplotlib and pyqsp (QSVT
phases). On Debian or Ubuntu system
Python, pyqsp fails to build (`AttributeError: install_layout`); use a virtual environment:

    python -m venv venv && . venv/bin/activate
    pip install -U setuptools wheel && pip install -r requirements.txt

## Reproduce a paper

Every script writes its `tables/` and `figs/` into its own paper folder, wherever it is started from
(a terminal in that folder, the repository root, or the Run button of an editor); these folders are not
committed. Install the package first, in the Python environment that runs the scripts
(`pip install -e .` from the repository root, or `pip install -r requirements.txt`).

    cd papers/multiplexed && python makeTables.py && python makeFigures.py
    cd papers/inversion   && python makeTables.py && python makeFigures.py
    cd papers/readout     && python makeTables.py && python makeFigures.py

Each folder holds only `makeTables.py` and `makeFigures.py`. Without arguments, `makeTables.py` writes the
tables; `python makeTables.py checks` runs the checks of the numbers the text quotes outside the tables,
`python makeTables.py listings` runs the code listings of the paper and compares their output with the
output printed in the paper, and `python makeTables.py --list` lists every name. Every script asserts
the numbers its paper quotes before it writes a table or figure. Each folder has a README with the
commands, the sections they reproduce, and their run times.

## Package

| module | contents | used by |
|---|---|---|
| `arithmetic/` | CORDIC fixed-point arithmetic, reversible circuits and a bit-packed simulator, multiplexed state preparation (`SymbolVector`, `Tree`, `poisson_isometry`), cost estimates of the alternatives | all |
| `fourier_symbol.py` | closed-form symbol of `K`, functions of `K` by FFT, `K` assembled element by element | inversion, readout |
| `isometry.py` | `Vhat(k)` in closed form, `V` assembled, the fixed unitary `T`, and the gate-level circuit `U_V` (`IsometryCircuit`) | inversion, readout |
| `cells.py` | the three test cells (square, cross, laminate), matrix-free operators, material oracles checked on every element, the finite element cell solution | inversion, readout |
| `polynomials.py` | the odd inverse polynomial `q_t`, the even range-projection polynomial `f_t`, and their QSVT phases | inversion, readout |
| `inversion.py` | the inverse applied matrix-free, the choice of `F`, resource counts, the three constructions of the wavenumber-controlled stage, the dense references of the checks, and the inverse as one Qiskit circuit | inversion, readout |
| `circuits.py` | Qiskit circuits for the square: `Q(k)` from uniformly controlled rotations, `U_V`, `U_Sigma`, the inverse, and the readout | inversion, readout |
| `emulation.py` | the readout amplitude with the gate-level `U_V`, exact unitaries within `delta`, amplitude-estimation sampling | readout |
| `readout.py` | the least-squares readout (matrix-free, Qiskit, gate-level arithmetic) and its cost against the readout through the inverse | readout |
| `reproduce.py` | the command line of the paper scripts, and the check of a listing against the paper | all |

## License

Apache 2.0.
