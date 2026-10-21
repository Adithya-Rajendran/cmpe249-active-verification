"""Binomial confidence bounds: Clopper-Pearson (exact) and Jeffreys, one- and two-sided.

Stdlib only. Both families are beta quantiles:

    Clopper-Pearson  lower = Beta(x, n-x+1).ppf(alpha)      (0 when x = 0)
                     upper = Beta(x+1, n-x).ppf(1-alpha)    (1 when x = n)
    Jeffreys         lower = Beta(x+1/2, n-x+1/2).ppf(alpha)    (0 when x = 0)
                     upper = Beta(x+1/2, n-x+1/2).ppf(1-alpha)  (1 when x = n)

where alpha = 1 - conf for a one-sided bound and (1 - conf) / 2 per side for a two-sided
interval (Brown, Cai & DasGupta 2001 for the Jeffreys convention at x = 0 and x = n).
The regularized incomplete beta function is a Lentz continued fraction; quantiles are found
by bisection. binom_cdf is an independent exact sum used to cross-check Clopper-Pearson.

Counts must be integers: these intervals are for deterministic policies (one decision per
episode). R's seed-averaged fractional counts are not binomial counts and are refused.
"""
from __future__ import annotations

import math
import numbers

_BISECT_ITERS = 200
_CF_MAX_ITERS = 10000
_CF_EPS = 3e-16
_TINY = 1e-300


def _check(x: object, n: object) -> tuple[int, int]:
    for name, v in (("x", x), ("n", n)):
        if isinstance(v, bool) or not isinstance(v, numbers.Integral):
            raise TypeError(f"{name} must be an int count, got {v!r}")
    x, n = int(x), int(n)  # type: ignore[arg-type]
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if not 0 <= x <= n:
        raise ValueError(f"need 0 <= x <= n, got x={x}, n={n}")
    return x, n


def _check_conf(conf: float) -> float:
    conf = float(conf)
    if not 0.0 < conf < 1.0:
        raise ValueError(f"conf must be in (0, 1), got {conf}")
    return conf


# ---------------------------------------------------------------- special functions

def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta function (modified Lentz)."""
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > _TINY else _TINY)
    h = d
    for m in range(1, _CF_MAX_ITERS + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > _TINY else _TINY)
        c = 1.0 + aa / c
        c = c if abs(c) > _TINY else _TINY
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > _TINY else _TINY)
        c = 1.0 + aa / c
        c = c if abs(c) > _TINY else _TINY
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < _CF_EPS:
            return h
    raise ArithmeticError(f"betainc continued fraction did not converge (a={a}, b={b}, x={x})")


def betainc(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta I_x(a, b) = Beta(a, b).cdf(x), for a, b > 0."""
    if a <= 0 or b <= 0:
        raise ValueError("a and b must be > 0")
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_front = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                 + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return math.exp(log_front) * _betacf(a, b, x) / a
    return 1.0 - math.exp(log_front) * _betacf(b, a, 1.0 - x) / b


def beta_ppf(q: float, a: float, b: float) -> float:
    """Quantile of Beta(a, b): the p with betainc(a, b, p) = q (bisection; monotone in p)."""
    if not 0.0 <= q <= 1.0:
        raise ValueError(f"q must be in [0, 1], got {q}")
    if q == 0.0:
        return 0.0
    if q == 1.0:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(_BISECT_ITERS):
        mid = 0.5 * (lo + hi)
        if betainc(a, b, mid) < q:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-15:
            break
    return 0.5 * (lo + hi)


def binom_cdf(x: int, n: int, p: float) -> float:
    """P(X <= x) for X ~ Binomial(n, p), by exact summation in log space."""
    x, n = _check(x, n)
    if p <= 0.0:
        return 1.0
    if p >= 1.0:
        return 1.0 if x >= n else 0.0
    lp, lq = math.log(p), math.log1p(-p)
    return min(1.0, sum(math.exp(math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)
                                 + i * lp + (n - i) * lq) for i in range(x + 1)))


# ---------------------------------------------------------------- Clopper-Pearson

def clopper_pearson_upper(x: int, n: int, conf: float = 0.95) -> float:
    """One-sided upper bound: P(rate <= bound) >= conf. 0/40 at 95% -> 0.0722."""
    x, n = _check(x, n)
    conf = _check_conf(conf)
    return 1.0 if x == n else beta_ppf(conf, x + 1, n - x)


def clopper_pearson_lower(x: int, n: int, conf: float = 0.95) -> float:
    """One-sided lower bound: P(rate >= bound) >= conf. 16/20 at 95% -> 0.5990."""
    x, n = _check(x, n)
    conf = _check_conf(conf)
    return 0.0 if x == 0 else beta_ppf(1.0 - conf, x, n - x + 1)


def clopper_pearson(x: int, n: int, conf: float = 0.95) -> tuple[float, float]:
    """Two-sided interval (alpha/2 in each tail)."""
    conf = _check_conf(conf)
    side = 1.0 - (1.0 - conf) / 2.0
    return clopper_pearson_lower(x, n, side), clopper_pearson_upper(x, n, side)


def clopper_pearson_upper_bisect(x: int, n: int, conf: float = 0.95) -> float:
    """Same bound as clopper_pearson_upper, from binom_cdf(x; n, p) = 1 - conf (cross-check)."""
    x, n = _check(x, n)
    conf = _check_conf(conf)
    if x == n:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(_BISECT_ITERS):
        mid = 0.5 * (lo + hi)
        if binom_cdf(x, n, mid) > 1.0 - conf:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-15:
            break
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------- Jeffreys

def jeffreys_upper(x: int, n: int, conf: float = 0.95) -> float:
    """One-sided Jeffreys upper bound (Beta(x+1/2, n-x+1/2) quantile; 1 when x = n)."""
    x, n = _check(x, n)
    conf = _check_conf(conf)
    return 1.0 if x == n else beta_ppf(conf, x + 0.5, n - x + 0.5)


def jeffreys_lower(x: int, n: int, conf: float = 0.95) -> float:
    """One-sided Jeffreys lower bound (0 when x = 0)."""
    x, n = _check(x, n)
    conf = _check_conf(conf)
    return 0.0 if x == 0 else beta_ppf(1.0 - conf, x + 0.5, n - x + 0.5)


def jeffreys(x: int, n: int, conf: float = 0.95) -> tuple[float, float]:
    """Two-sided Jeffreys interval (alpha/2 in each tail)."""
    conf = _check_conf(conf)
    side = 1.0 - (1.0 - conf) / 2.0
    return jeffreys_lower(x, n, side), jeffreys_upper(x, n, side)


def bounds_summary(x: int, n: int, conf: float = 0.95) -> dict[str, float | int]:
    """Point estimate plus every bound, for reports."""
    x, n = _check(x, n)
    return {"x": x, "n": n, "rate": x / n, "conf": conf,
            "cp_upper_1s": clopper_pearson_upper(x, n, conf), "cp_lower_1s": clopper_pearson_lower(x, n, conf),
            "jeffreys_upper_1s": jeffreys_upper(x, n, conf), "jeffreys_lower_1s": jeffreys_lower(x, n, conf),
            "cp_2s": list(clopper_pearson(x, n, conf)), "jeffreys_2s": list(jeffreys(x, n, conf))}
