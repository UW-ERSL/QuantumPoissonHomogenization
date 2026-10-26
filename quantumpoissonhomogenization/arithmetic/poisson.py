"""poisson -- the circuit for the Fourier blocks of the isometry of the periodic
Poisson problem with bilinear (Q4) elements in two dimensions.

For a wavenumber k = (k1, k2) != 0 and phi_j = pi k_j / N, the block of the
isometry V = W^(1/2) G (K^+)^(1/2) is Vhat(k) = e^(i(phi1 + phi2)) T c(k)/||c(k)||
with a fixed unitary T and

    c(k) = (a, b, zeta/sqrt3, zeta/sqrt3),
    a = sin phi1 cos phi2,  b = cos phi1 sin phi2,  zeta = sin phi1 sin phi2.

By sum to product, with theta_+- = phi1 +- phi2,
    a = (sin theta_+ + sin theta_-)/2,  b = (sin theta_+ - sin theta_-)/2,
    zeta = (cos theta_- - cos theta_+)/2,
so c(k) = C t(k) with t = (1, cos theta_+, sin theta_+, cos theta_-, sin theta_-).
The two hourglass entries are equal, so their node is a known-angle node
(pi/4) whose left register holds zeta (scale 1/sqrt3).
"""
import numpy as np

from .symbol import SymbolVector

_R3 = 1 / np.sqrt(3)
ANGLES = [((1, 1), 0), ((1, -1), 0)]
C = np.array([[0, 0, 0.5, 0, 0.5],             # a
              [0, 0, 0.5, 0, -0.5],            # b
              [0, -0.5 * _R3, 0, 0.5 * _R3, 0],  # zeta / sqrt3
              [0, -0.5 * _R3, 0, 0.5 * _R3, 0]])  # zeta / sqrt3 (not computed)
KNOWN = {(1, 1): (np.pi / 4, _R3)}


def poisson_isometry(n, F, G=4):
    """The circuit for c(k)/||c(k)|| on the wavenumber register (2 x n qubits).
    Default G = 4: the configuration of the Poisson paper (registers of F + 4
    bits, F iterations)."""
    return SymbolVector(n, 2, F, ANGLES, C, G=G, known=KNOWN)
