"""makeFigures.py -- the figure of the multiplexed state preparation paper.

    python makeFigures.py           Figure 4 (figs/error.pdf), about 15 seconds
"""
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))     # tables/ and figs/ go to this folder
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from quantumpoissonhomogenization.arithmetic import poisson_isometry, exact_unit
from quantumpoissonhomogenization.reproduce import dispatch


def draw_error() -> None:
    """Figure of Sec. 4: the measured error of the prepared vector c~(k).

    (a) delta(k) = ||c~(k) - c(k)/||c(k)|| || against ||c(k)|| for every k != 0 at
        N = 64, F = 24, under the line 16 2^-F/||c||.
    (b) max over k of delta(k) 2^F / N against N, for F = 12, 16, ..., 32
        (the 42 cases), with the measured bound 6.
    Writes figs/error.pdf."""
    def errors(n, F):
        sv = poisson_isometry(n, F)
        amp, vreg, vex, treg, tex, ck = sv.simulate()
        assert ck["work_bits_left_set"] == 0 and ck["inputs_restored"] and ck["no_overflow"]
        return np.linalg.norm(amp - exact_unit(vex), axis=1), np.linalg.norm(vex, axis=1)

    # (a) one configuration, every k
    na, Fa = 6, 24
    ea, ca = errors(na, Fa)
    ua = 2.0 ** -Fa
    assert np.all(ea <= 6 * 2 ** na * ua)
    assert ca[np.argmax(ea)] < 0.1                      # the worst k is a short vector
    assert np.all(ea * ca <= 16 * ua)                   # delta(k) ||c(k)|| <= 16 u
    print(f"(a) N = {2**na}, F = {Fa}: max delta = {ea.max():.2e} at ||c|| = {ca[np.argmax(ea)]:.3f}; "
          f"max delta ||c|| 2^F = {(ea * ca).max() / ua:.2f}")

    # (b) the 42 cases
    Ns, Fs = range(2, 9), range(12, 33, 4)
    ratio = np.array([[errors(n, F)[0].max() * 2 ** F / 2 ** n for n in Ns] for F in Fs])
    assert ratio.max() <= 6
    print(f"(b) max delta 2^F/N over the 42 cases: {ratio.max():.2f} at N = {2 ** list(Ns)[ratio.max(0).argmax()]}")
    assert ratio[:, 2:].max() < 2                       # N >= 16
    print(f"(b) max delta 2^F/N for N >= 16: {ratio[:, 2:].max():.2f}")

    # figure
    plt.rcParams.update({"font.size": 8, "font.family": "serif", "mathtext.fontset": "cm", "axes.linewidth": 0.6, "xtick.major.width": 0.6,
                         "ytick.major.width": 0.6})
    fig, (axa, axb) = plt.subplots(2, 1, figsize=(3.3, 4.6))
    ink, grid = "#333333", "#dddddd"

    axa.loglog(ca, ea, ".", ms=2.5, color="#2b6cb0", alpha=0.5)
    c = np.geomspace(ca.min(), 1, 50)
    axa.loglog(c, 16 * ua / c, "-", lw=1.2, color=ink)
    axa.text(0.075, 16 * ua / 0.075 * 1.25, r"$16 \cdot 2^{-F}/\|c\|_2$", color=ink)
    axa.set_xlabel(r"$\|c(k)\|_2$")
    axa.set_ylabel(r"$\delta(k)$")
    axa.set_title(rf"(a) every $k \neq 0$, $N = {2**na}$, $F = {Fa}$", fontsize=8)

    colors = plt.cm.Blues(np.linspace(0.35, 0.95, len(Fs)))
    markers = "os^Dvp"
    xs = [2 ** n for n in Ns]
    for F, row, col, mk in zip(Fs, ratio, colors, markers):
        axb.semilogx(xs, row, "-", marker=mk, ms=4, lw=1.2, color=col, label=rf"$F = {F}$")
    axb.axhline(6, color=ink, lw=1.0, ls="--")
    axb.text(xs[-1], 6.15, "measured bound 6", ha="right", va="bottom", color=ink)
    axb.set_xscale("log", base=2)
    axb.set_xticks(xs, [str(x) for x in xs])
    axb.set_ylim(0, 7.2)
    axb.set_xlabel(r"$N$")
    axb.set_ylabel(r"$\max_k \delta(k)\, 2^F / N$")
    axb.set_title("(b) the 42 cases", fontsize=8)
    axb.legend(ncol=3, fontsize=7, frameon=False, loc="upper right", bbox_to_anchor=(1.0, 0.86),
               handlelength=1.6, columnspacing=0.8)

    for ax in (axa, axb):
        ax.grid(True, color=grid, lw=0.5)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.tight_layout()
    os.makedirs("figs", exist_ok=True)
    fig.savefig("figs/error.pdf")
    print("wrote figs/error.pdf")


FIGURES = {"error": (draw_error, "Figure 4: the measured error of the prepared vector (Sec. 4)")}


if __name__ == "__main__":
    dispatch(sys.argv[1:], {"figures": FIGURES})
