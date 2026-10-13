"""qstats.py — statistics shared by vegas.py and kwrev.py.

Every resampling routine is seeded, so reruns are deterministic.
"""
from __future__ import annotations

import itertools
import math

import numpy as np
from scipy.stats import kurtosis, norm, skew

EULER = 0.5772156649015329


# ---------------------------------------------------------------- resampling
def stationary_idx(n, mean_block, rng):
    """Politis-Romano stationary bootstrap: one resampled index vector of length n."""
    p = 1.0 / max(mean_block, 1.0)
    out = np.empty(n, np.int64)
    i = rng.integers(n)
    for k in range(n):
        if k and rng.random() < p:
            i = rng.integers(n)
        out[k] = i
        i = (i + 1) % n
    return out


def mbb_idx(n, block, rng):
    """moving-block bootstrap (circular), as in Kitron & Wengrowicz"""
    nb = int(math.ceil(n / block))
    starts = rng.integers(0, n, nb)
    return ((starts[:, None] + np.arange(block)[None, :]) % n).ravel()[:n]


def per_trade_mean_ci(day_sum, day_cnt, B=10_000, mean_block=5, seed=7, q=(5, 95)):
    """mean R per trade with a day-block (stationary) bootstrap: resample days, ratio of sums"""
    day_sum, day_cnt = np.asarray(day_sum, float), np.asarray(day_cnt, float)
    n = len(day_sum)
    if n == 0 or day_cnt.sum() == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = np.empty(B)
    for b in range(B):
        ix = stationary_idx(n, mean_block, rng)
        c = day_cnt[ix].sum()
        means[b] = day_sum[ix].sum() / c if c else 0.0
    return float(day_sum.sum() / day_cnt.sum()), float(np.percentile(means, q[0])), float(np.percentile(means, q[1]))


def boot_max_dd(day_sum, B=10_000, mean_block=5, seed=7, pct=95):
    """95th percentile of the maximum drawdown (in R) over resampled day paths of the same length"""
    day_sum = np.asarray(day_sum, float)
    n = len(day_sum)
    if n == 0:
        return np.nan
    rng = np.random.default_rng(seed)
    dds = np.empty(B)
    for b in range(B):
        eq = np.cumsum(day_sum[stationary_idx(n, mean_block, rng)])
        dds[b] = np.max(np.maximum.accumulate(np.r_[0.0, eq])[1:] - eq)
    return float(np.percentile(dds, pct))


def max_dd(x):
    eq = np.cumsum(np.asarray(x, float))
    if len(eq) == 0:
        return 0.0
    return float(np.max(np.maximum.accumulate(np.r_[0.0, eq])[1:] - eq))


# ---------------------------------------------------------------- overfitting
def deflated_sharpe(R, trial_srs, n_trials):
    """Bailey & López de Prado (2014): P(true SR > expected max SR of n_trials zero-skill trials)"""
    R = np.asarray(R, float)
    if len(R) < 30 or R.std(ddof=1) == 0 or len(trial_srs) < 2:
        return np.nan
    sr = R.mean() / R.std(ddof=1)
    N = max(int(n_trials), len(trial_srs), 2)
    v = float(np.var(trial_srs, ddof=1))
    sr0 = math.sqrt(max(v, 1e-12)) * ((1 - EULER) * norm.ppf(1 - 1 / N) + EULER * norm.ppf(1 - 1 / (N * math.e)))
    g3, g4 = skew(R), kurtosis(R, fisher=False)
    den = 1 - g3 * sr + (g4 - 1) / 4 * sr ** 2
    if den <= 0:
        return np.nan
    return float(norm.cdf((sr - sr0) * math.sqrt(len(R) - 1) / math.sqrt(den)))


def pbo_cscv(M, S=16):
    """Probability of backtest overfitting (Bailey, Borwein, López de Prado, Zhu).
    M: T x N matrix of per-period returns (one column per configuration). Splits T into S blocks,
    takes every combination of S/2 blocks as in-sample, picks the best in-sample Sharpe and records
    the logit of its out-of-sample relative rank. PBO = share of logits <= 0."""
    M = np.asarray(M, float)
    T, N = M.shape
    if N < 2 or T < S * 2:
        return np.nan
    edges = np.linspace(0, T, S + 1).astype(int)
    blocks = [M[edges[i]:edges[i + 1]] for i in range(S)]
    bsum = np.array([b.sum(0) for b in blocks])
    bsq = np.array([(b ** 2).sum(0) for b in blocks])
    bn = np.array([len(b) for b in blocks], float)
    lam = []
    for combo in itertools.combinations(range(S), S // 2):
        m = np.zeros(S, bool)
        m[list(combo)] = True

        def sharpe(mask):
            n = bn[mask].sum()
            mu = bsum[mask].sum(0) / n
            var = bsq[mask].sum(0) / n - mu ** 2
            return mu / np.sqrt(np.maximum(var, 1e-18))
        s_is, s_oos = sharpe(m), sharpe(~m)
        best = int(np.argmax(s_is))
        rank = (s_oos < s_oos[best]).sum() + 0.5 * ((s_oos == s_oos[best]).sum() - 1) + 1
        w = rank / (N + 1)
        lam.append(math.log(w / (1 - w)))
    return float(np.mean(np.asarray(lam) <= 0))


# ---------------------------------------------------------------- classification
def auc(p, y):
    """Mann-Whitney AUC with ties counted 1/2 (identical to sklearn.metrics.roc_auc_score)"""
    p, y = np.asarray(p, float), np.asarray(y).astype(bool)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return np.nan
    order = np.argsort(p, kind="mergesort")
    ps = p[order]
    ranks = np.empty(len(p))
    i = 0
    while i < len(ps):                      # average ranks of ties
        j = i
        while j + 1 < len(ps) and ps[j + 1] == ps[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1
        i = j + 1
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def auc_fast(p, y):
    from scipy.stats import rankdata
    p, y = np.asarray(p, float), np.asarray(y).astype(bool)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(p)
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def auc_mbb(p, y, block=384, B=300, seed=7):
    """one-sided moving-block-bootstrap p-value for AUC > 0.5 (paper's per-asset test)"""
    p, y = np.asarray(p, float), np.asarray(y).astype(bool)
    a = auc_fast(p, y)
    rng = np.random.default_rng(seed)
    bs = np.array([auc_fast(p[ix], y[ix]) for ix in (mbb_idx(len(p), block, rng) for _ in range(B))])
    pv = (np.sum(bs <= 0.5) + 1) / (B + 1)               # share of replicates at or below no-skill
    return a, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), float(pv)
