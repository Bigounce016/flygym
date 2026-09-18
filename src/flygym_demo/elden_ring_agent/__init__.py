"""Decision-layer scaffolding for an offline fly game agent."""

from .policy import ActionCommand, FlyGameObservation, RuleBasedFlyPolicy


def run_live_elden_ring_demo(*args, **kwargs):
    """Lazily import and run the MuJoCo-backed demo."""
    from .live_demo import run_live_elden_ring_demo as run_demo

    return run_demo(*args, **kwargs)


__all__ = [
    "ActionCommand",
    "FlyGameObservation",
    "RuleBasedFlyPolicy",
    "run_live_elden_ring_demo",
]