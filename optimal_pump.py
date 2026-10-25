"""Print the phase-matched pump frequency for the TWPA in transientNetlist.cir.

Usage:
    python optimal_pump.py --ip 3        # 3 uA peak in the line (6 uA source amplitude in the .cir)
"""
import argparse
import os

import numpy as np

# Line parameters, matching transientNetlist.cir
IC = 5e-6           # jjmod icrit
CJ = 240.522e-15    # jjmod cap
C_SHUNT = 26e-15    # C1, C2, ... to ground
CELLS_PER_RPM = 20  # 19 junction cells + 1 RPM cell per supercell
PHI0 = 2.067833848e-15
Z_REF = 50.0


def rpm_abcd(filename):
    """Frequencies and ABCD matrices of the RPM cell from its touchstone file (GHz S MA R 50)."""
    d = np.loadtxt(filename, comments=["!", "#"])
    f = d[:, 0] * 1e9
    S11, S21, S12, S22 = (d[:, k] * np.exp(1j * np.deg2rad(d[:, k + 1])) for k in (1, 3, 5, 7))
    den = 2 * S21
    abcd = np.empty((len(f), 2, 2), dtype=complex)
    abcd[:, 0, 0] = ((1 + S11) * (1 - S22) + S12 * S21) / den
    abcd[:, 0, 1] = Z_REF * ((1 + S11) * (1 + S22) - S12 * S21) / den
    abcd[:, 1, 0] = ((1 - S11) * (1 - S22) - S12 * S21) / (Z_REF * den)
    abcd[:, 1, 1] = ((1 - S11) * (1 + S22) + S12 * S21) / den
    return f, abcd


def dispersion(f, rpm):
    """Bloch phase per cell of the supercell (19 junction cells, then the RPM cell)."""
    w = 2 * np.pi * f
    L0 = PHI0 / (2 * np.pi * IC)
    Z = 1 / (1j * w * CJ + 1 / (1j * w * L0))  # series junction
    Y = 1j * w * C_SHUNT                        # shunt cap
    cell = np.empty((len(f), 2, 2), dtype=complex)
    cell[:, 0, 0] = 1 + Z * Y
    cell[:, 0, 1] = Z
    cell[:, 1, 0] = Y
    cell[:, 1, 1] = 1
    M = np.linalg.matrix_power(cell, CELLS_PER_RPM - 1) @ rpm
    cos_k = ((M[:, 0, 0] + M[:, 1, 1]) / 2).real
    return np.arccos(np.clip(cos_k, -1, 1)) / CELLS_PER_RPM, np.abs(cos_k) > 1


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ip", type=float, required=True, help="peak pump current in the line, uA")
    parser.add_argument("--fs", type=float, default=5.0, help="reference signal frequency, GHz")
    parser.add_argument("--s2p", default=os.path.join(here, "RPMCell_FabReady.s2p"))
    args = parser.parse_args()

    Ip = args.ip * 1e-6
    fs = args.fs * 1e9
    Xi = 1 / (2 * IC**2)  # L(I) ~ L0 (1 + I^2 / (2 Ic^2))

    f, rpm = rpm_abcd(args.s2p)
    beta, stopband = dispersion(f, rpm)

    # Search below the RPM stop band, and high enough that the idler is also below it
    f_gap = f[stopband & (f > fs)].min()
    fp = np.linspace((f_gap + fs) / 2, f_gap, 200001)[1:-1]
    beta_p = np.interp(fp, f, beta)

    # Linear mismatch (2 fp = fs + fi) minus pump self-phase modulation
    delta_beta = (2 * beta_p - np.interp(fs, f, beta) - np.interp(2 * fp - fs, f, beta)
                  - 0.25 * beta_p * Ip**2 * Xi)

    crossings = np.where(np.sign(delta_beta[:-1]) * np.sign(delta_beta[1:]) < 0)[0]
    if crossings.size == 0:
        raise SystemExit("No phase-matched pump found below the RPM stop band")
    i = crossings[-1]  # the crossing closest to the RPM resonance
    f_pump = fp[i] - delta_beta[i] * (fp[i + 1] - fp[i]) / (delta_beta[i + 1] - delta_beta[i])
    print(f"{f_pump / 1e9:.4f}e9")


if __name__ == "__main__":
    main()
