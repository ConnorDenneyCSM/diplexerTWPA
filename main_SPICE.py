"""Plot S21 and S11 from WRspice transient results for the TWPA.

Usage:
    python main_SPICE.py [datafile] [--netlist transientNetlist.cir]
"""
import argparse
import os
import re

import numpy as np
import matplotlib.pyplot as plt

Z_REF = 50.0
R_SENSE = 1e-3  # current-sense resistors (R11, R22) in the netlist


def read_rawfile(filename):
    """Parse a WRspice ASCII rawfile into its dimensions and a (vars, points) array."""
    with open(filename, "r") as fid:
        for line in fid:
            if line.startswith("No. Variables:"):
                n_vars = int(line[14:])
            elif line.startswith("Dimensions:"):
                dims = tuple(int(d) for d in line[11:].split(","))
            elif line.startswith("Values:"):
                break
        values = np.array(fid.read().split(), dtype=float)
    # Each point is: index, var0, var1, ... var(N-1)
    return dims, values.reshape(-1, n_vars + 1).T


def read_sweep(filename):
    """Signal frequencies from the sweep on the netlist's .tran line."""
    with open(filename, "r") as fid:
        m = re.search(r"^\.tran.*\bsweep\s+\S+\s+(\S+)\s+(\S+)\s+(\S+)", fid.read(),
                      re.IGNORECASE | re.MULTILINE)
    fstart, fstop, fstep = (float(v) for v in m.groups())
    return np.arange(fstart, fstop + fstep / 2, fstep)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("datafile", nargs="?",
                        default=os.path.join(here, "wrspiceResults", "RPMCell_FabReadyData.txt"))
    parser.add_argument("--netlist", default=os.path.join(here, "transientNetlist.cir"))
    args = parser.parse_args()

    fsignal = read_sweep(args.netlist)
    (n_runs, n_pts), data = read_rawfile(args.datafile)

    # Variables: time, v(N1), v(N2400), v(NP), v(NR1), v(NR2)
    time, Vin, Vout = (data[k].reshape(n_runs, n_pts) for k in (1, 2, 3))
    Iin = data[5].reshape(n_runs, n_pts) / R_SENSE

    # Single-sided, amplitude-scaled spectra
    Vin_fft, Vout_fft, Iin_fft = (2 * np.fft.rfft(x, axis=1) / n_pts for x in (Vin, Vout, Iin))
    f = np.fft.rfftfreq(n_pts, np.mean(np.diff(time[0])))

    def peak(spec, idx):
        """Largest-magnitude bin among idx-1, idx, idx+1."""
        window = spec[idx - 1: idx + 2]
        return window[np.argmax(np.abs(window))]

    S21 = np.zeros(n_runs, dtype=complex)
    S11 = np.zeros(n_runs, dtype=complex)
    for i in range(n_runs):
        idx = np.searchsorted(f, fsignal[i])
        Vin_s, Vout_s, Iin_s = (peak(x[i], idx) for x in (Vin_fft, Vout_fft, Iin_fft))
        # Incident wave from the Norton source is -I*Z/2
        S21[i] = Vout_s / (-Iin_s * Z_REF / 2)
        S11[i] = Vin_s / (-Iin_s * Z_REF / 2) - 1

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(fsignal / 1e9, 20 * np.log10(np.abs(S21)), lw=2, label="S21")
    ax.plot(fsignal / 1e9, 20 * np.log10(np.abs(S11)), lw=2, label="S11")
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("Magnitude (dB)")
    ax.set_title("Gain and Return Loss")
    ax.legend(loc="best")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(os.path.join(os.path.dirname(os.path.abspath(args.datafile)),
                             "SPICEgainProfile.png"), dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
