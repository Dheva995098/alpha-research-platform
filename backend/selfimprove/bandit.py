"""Beta-Bernoulli Thompson sampling for arm selection (search-efficiency upgrade).

The autopilot must choose which (focus / dataset) to explore each batch. Greedy
pass-rate weighting lets a 1-of-1 lucky arm dominate a robust 80/100 arm and gives
no convergence guarantee. Thompson sampling is the hyperparameter-free, proven way
to maximize cumulative successes (passing alphas) under a fixed simulation budget:
keep a Beta(1+wins, 1+losses) posterior per arm, draw a sample from each, and pull
the arm with the highest draw — automatically balancing exploration/exploitation.

Stats come from the persistent attempt memory (real PASS/FAIL outcomes), so this
is grounded in observed BRAIN results, not heuristics.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple


def has_signal(arms: Sequence[str], stats: Dict[str, Tuple[int, int]]) -> bool:
    """True once any arm has at least one recorded trial (else: explore via fallback)."""
    return any(stats.get(arm, (0, 0))[1] > 0 for arm in arms)


def thompson_select(
    arms: Sequence[str],
    stats: Dict[str, Tuple[int, int]],
    rng,
    *,
    prior: Tuple[float, float] = (1.0, 1.0),
    priors: Optional[Dict[str, Tuple[float, float]]] = None,
) -> Optional[str]:
    """Pick an arm by sampling theta_i ~ Beta(1+wins, 1+losses) and taking the argmax.

    ``rng`` is a ``random.Random`` (uses ``betavariate``) so selection is reproducible
    under the autopilot's seed. ``priors`` allows a per-arm Beta prior (e.g. seed a
    high-value family slightly optimistic) while unseen arms use ``prior``.
    """
    best_arm: Optional[str] = None
    best_theta = -1.0
    for arm in arms:
        wins, trials = stats.get(arm, (0, 0))
        base = priors.get(arm, prior) if priors else prior
        a = base[0] + max(wins, 0)
        b = base[1] + max(trials - wins, 0)
        # Guard against degenerate parameters.
        a = a if a > 0 else 1.0
        b = b if b > 0 else 1.0
        theta = rng.betavariate(a, b)
        if theta > best_theta:
            best_theta = theta
            best_arm = arm
    return best_arm


def posterior_mean(arm: str, stats: Dict[str, Tuple[int, int]], prior: Tuple[float, float] = (1.0, 1.0)) -> float:
    """Exploit estimate wins/(wins+losses) with a Beta prior, for reporting/ranking."""
    wins, trials = stats.get(arm, (0, 0))
    a = prior[0] + max(wins, 0)
    b = prior[1] + max(trials - wins, 0)
    return round(a / (a + b), 4)
