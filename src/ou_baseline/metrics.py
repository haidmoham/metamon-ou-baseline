"""Dependency-free reporting and fail-closed action checks."""
import math
from statistics import median


def wilson(wins: int, total: int, z: float = 1.959963984540054):
    if not 0 <= wins <= total:
        raise ValueError("Require 0 <= wins <= total")
    if total == 0:
        return None
    p = wins / total
    d = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / d
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total**2)) / d
    return [max(0.0, centre - radius), min(1.0, centre + radius)]


def latency_summary(values):
    values = sorted(values)
    if not values:
        return {"count": 0, "median_ms": None, "p95_ms": None, "max_ms": None}
    return {"count": len(values), "median_ms": median(values),
            "p95_ms": values[math.ceil(0.95 * len(values)) - 1], "max_ms": values[-1]}


def require_legal(action: int, legal_actions):
    if action not in legal_actions:
        raise RuntimeError(f"Policy action {action} is not in legal mask {list(legal_actions)}; refusing fallback")


def summarize(battles, decision_ms):
    groups = {}
    for b in battles:
        group = groups.setdefault(b["opponent"], {"wins": 0, "losses": 0, "ties": 0, "incomplete": 0})
        group[{"win": "wins", "loss": "losses", "tie": "ties"}.get(b["outcome"], "incomplete")] += 1
    for group in groups.values():
        # Ties count as non-wins for a binary win-rate interval. Report them separately.
        n = group["wins"] + group["losses"] + group["ties"]
        group["completed"] = n
        group["win_rate"] = group["wins"] / n if n else None
        group["wilson_95_win_rate"] = wilson(group["wins"], n)
    return {"opponents": groups, "policy_decision_latency": latency_summary(decision_ms),
            "latency_scope": "single actor; tensor preparation + policy inference + device synchronization + action transfer; excludes simulator/network wait; includes first action"}
