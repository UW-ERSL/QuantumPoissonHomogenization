"""quantumpoissonhomogenization -- quantum algorithms for periodic two-phase heat conduction on a
structured Q4 mesh: the multiplexed state preparation of the isometry (arithmetic), the block encoding
of the preconditioned operator and its inverse (isometry, inversion), and the least-squares readout of
the effective conductivity (readout).

    arithmetic        CORDIC fixed-point arithmetic and multiplexed state preparation (Part 1, Part 2)
    fourier_symbol    closed-form symbol of K and functions of K by FFT
    isometry          Vhat(k), V, the fixed unitary T, and the gate-level circuit U_V
    cells             the three test cells (square, cross, laminate), matrix-free, material oracles
    polynomials       q_t (inverse), f_t (readout), and their QSVT phases
    inversion         the inverse applied matrix-free, the choice of F, resource counts
    circuits          Qiskit circuits for the inverse and the readout (square, state vector)
    emulation         the readout amplitude with the gate-level arithmetic U_V
    readout           the least-squares readout at three levels and its cost

    from quantumpoissonhomogenization import Cell, extreme_eigs, inverse_error, sigma_fe, ls_poly
    cell = Cell("square", N=256, rho=10.0)
    print(extreme_eigs(cell), inverse_error(cell, eps_p=1e-6), sigma_fe(cell), ls_poly(cell, 1e-2 / 3))
"""
from .fourier_symbol import symbol, symbol_function, apply_mode_function, reference
from .isometry import (IsometryCircuit, vhat_exact, V_assembled, T_matrix, exact_angles, canonical,
                       Q_exact, end_to_end_error)
from .cells import (Cell, GEOMETRIES, ALPHA, chi, check_oracle, k_function, square_chi, assemble,
                    e_hat, pcg, extreme_eigs, lambda_min_Kc, sigma_fe)
from .polynomials import q_poly, sup_norm, poly_plan, f_poly, even_phases
from .inversion import (inverse_plan, encoding_bits, inverse_error, uv_counts, inverse_resources,
                        stage_options)
from .readout import (ls_exact, ls_poly, qiskit_readout, arithmetic_readout, bits_for,
                      amplitude_estimation, ls_cost, baseline_cost)

__all__ = ["symbol", "symbol_function", "apply_mode_function", "reference", "IsometryCircuit",
           "vhat_exact", "V_assembled", "T_matrix", "exact_angles", "canonical", "Q_exact",
           "end_to_end_error", "Cell", "GEOMETRIES", "ALPHA", "chi", "check_oracle", "k_function",
           "square_chi", "assemble", "e_hat", "pcg", "extreme_eigs", "lambda_min_Kc", "sigma_fe",
           "q_poly", "sup_norm", "poly_plan", "f_poly", "even_phases", "inverse_plan",
           "encoding_bits", "inverse_error", "uv_counts", "inverse_resources", "stage_options",
           "ls_exact", "ls_poly", "qiskit_readout", "arithmetic_readout", "bits_for",
           "amplitude_estimation", "ls_cost", "baseline_cost"]
