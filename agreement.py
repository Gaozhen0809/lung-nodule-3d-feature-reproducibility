#!/usr/bin/env python3
"""ICC(2,1) and Bland-Altman agreement between software-derived and
independently computed features."""
import numpy as np


def icc21(x, y):
    d = np.column_stack([np.asarray(x, float), np.asarray(y, float)])
    d = d[np.isfinite(d).all(axis=1)]
    n, k = d.shape
    gm = d.mean()
    ms_r = k * ((d.mean(1) - gm) ** 2).sum() / (n - 1)
    ms_c = n * ((d.mean(0) - gm) ** 2).sum() / (k - 1)
    ss_t = ((d - gm) ** 2).sum()
    ss_e = ss_t - ms_r * (n - 1) - ms_c * (k - 1)
    ms_e = max(ss_e, 1e-12) / ((n - 1) * (k - 1))
    return (ms_r - ms_e) / (ms_r + (k - 1) * ms_e + k * (ms_c - ms_e) / n)


def bland_altman(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    diff = x - y
    md, sd = diff.mean(), diff.std(ddof=1)
    return md, md - 1.96 * sd, md + 1.96 * sd
