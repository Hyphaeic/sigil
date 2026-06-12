#!/usr/bin/env python3
"""Cross-validate sigil's channel processing against scikit-rf.

Loads a 4-port Touchstone file with scikit-rf (independent parser and
mixed-mode math), computes SDD21 for both common port conventions, and
compares against the channel_response.csv that `si-kernel simulate`
writes for the same file.

Usage:
  crosscheck_skrf.py <file.s4p>                      # report only
  crosscheck_skrf.py <file.s4p> <channel_response.csv>  # + numeric diff

Run inside a venv that has scikit-rf, e.g.:
  python3 -m venv /tmp/skrf-venv && /tmp/skrf-venv/bin/pip install scikit-rf
  /tmp/skrf-venv/bin/python crosscheck_skrf.py ...
"""

import sys

import numpy as np
import skrf as rf


def sdd21(s, inp, inn, outp, outn):
    """Mixed-mode SDD21 from single-ended S-params, 1-based port numbers."""
    i, j, k, l = inp - 1, inn - 1, outp - 1, outn - 1
    return 0.5 * (s[:, k, i] - s[:, k, j] - s[:, l, i] + s[:, l, j])


def db(x):
    return 20.0 * np.log10(np.maximum(np.abs(x), 1e-15))


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    path = sys.argv[1]

    n = rf.Network(path)
    f_ghz = n.f / 1e9
    print(f"file:        {path}")
    print(f"ports:       {n.nports}, points: {len(n.f)}, "
          f"range: {f_ghz[0]:.3f}-{f_ghz[-1]:.2f} GHz")

    if n.nports != 4:
        sys.exit("expected a 4-port file")

    # Try both common port conventions; the true through pair has the
    # higher LOW-frequency transmission. (Long channels are at the VNA
    # noise floor by mid-band, where the comparison is meaningless.)
    candidates = {
        "(1+,3-)->(2+,4-)": sdd21(n.s, 1, 3, 2, 4),
        "(1+,2-)->(3+,4-)": sdd21(n.s, 1, 2, 3, 4),
    }
    ref = int(np.argmin(np.abs(f_ghz - 2.0)))
    mid = len(f_ghz) // 2
    convention, sdd = max(
        candidates.items(), key=lambda kv: np.abs(kv[1][ref])
    )
    print(f"convention:  {convention} "
          f"(|SDD21| @ {f_ghz[ref]:.1f} GHz: {np.abs(sdd[ref]):.4f} vs "
          f"{min(np.abs(c[ref]) for c in candidates.values()):.4f})")

    for f_t in (4.0, 8.0, 16.0, 26.56):
        if f_t <= f_ghz[-1]:
            idx = int(np.argmin(np.abs(f_ghz - f_t)))
            print(f"IL @ {f_t:6.2f} GHz: {-db(sdd)[idx]:7.2f} dB  (skrf)")

    # Group delay mid-band from SDD21 phase.
    phase = np.unwrap(np.angle(sdd))
    gd = -np.gradient(phase, 2 * np.pi * n.f)
    print(f"group delay: {np.median(gd[mid - 50:mid + 50]) * 1e9:.3f} ns "
          f"(mid-band median)")

    # Optional numeric diff against sigil's channel_response.csv.
    if len(sys.argv) > 2:
        rows = np.genfromtxt(sys.argv[2], delimiter=",", skip_header=1)
        sig_f, sig_db = rows[:, 0], rows[:, 1]
        skrf_interp = np.interp(sig_f, f_ghz, db(sdd))
        delta = np.abs(sig_db - skrf_interp)
        print(f"sigil vs skrf SDD21: max |delta| = {delta.max():.4f} dB, "
              f"mean = {delta.mean():.4f} dB over {len(sig_f)} points")
        if delta.max() < 0.1:
            print("CROSS-CHECK PASS (agreement < 0.1 dB)")
        else:
            print("CROSS-CHECK FAIL (investigate parser/mixed-mode math)")


if __name__ == "__main__":
    main()
