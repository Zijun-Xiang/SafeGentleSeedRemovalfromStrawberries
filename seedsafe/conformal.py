"""Split conformal prediction: turn calibration errors into bounds with a coverage guarantee.

Given n calibration scores s_1..s_n (errors of an estimate against ground truth) and a
miscoverage level alpha, the bound

    q = the ceil((n + 1) * (1 - alpha))-th smallest score

satisfies  P(new score <= q) >= 1 - alpha  for a new, exchangeable sample - whatever the
error distribution is. We use it for: seed position errors of the camera, and errors of
the indentation / force / squeeze estimates.
"""
import math

import numpy as np


def conformal_quantile(scores, alpha):
    s = np.sort(np.asarray(scores, float))
    n = len(s)
    if n == 0:
        return float("inf")
    k = math.ceil((n + 1) * (1 - alpha))
    if k > n:
        return float(s[-1] * 1.25 + 1e-12)       # not enough data for this level: inflate the max
    return float(s[k - 1])


def coverage(scores, q):
    return float(np.mean(np.asarray(scores) <= q))


def bounds_from_scores(scores: dict, alpha: float) -> dict:
    """scores: name -> array. Returns name -> conformal bound at level alpha."""
    return {k: conformal_quantile(v, alpha) for k, v in scores.items()}
