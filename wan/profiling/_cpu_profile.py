"""Opt-in cProfile wrappers for specific hot regions.

Enable per-region via env: WAN_PROFILE_CPU_SCHEDULER=1 enables cProfile
around the scheduler_step span. Stats accumulate across all calls and
are dumped at flush() time to `<profile_dir>/cpu_<name>_rank<N>.prof`
(readable with `python -m pstats`).

Why not always-on: cProfile adds 30-50% overhead to the profiled region,
which distorts wall-clock measurements. Opt-in keeps baseline runs clean.
"""

import cProfile
import os
from contextlib import contextmanager


# name -> cProfile.Profile instance (one per named region, accumulates).
_PROFILES: dict[str, cProfile.Profile] = {}


def _enabled_for(name: str) -> bool:
    env_key = f"WAN_PROFILE_CPU_{name.upper()}"
    return os.environ.get(env_key, "0") == "1"


@contextmanager
def cpu_profile(name: str):
    """Accumulate cProfile stats for this named region.

    No-op unless WAN_PROFILE_CPU_<NAME>=1 is set.
    """
    if not _enabled_for(name):
        yield
        return
    prof = _PROFILES.get(name)
    if prof is None:
        prof = cProfile.Profile()
        _PROFILES[name] = prof
    prof.enable()
    try:
        yield
    finally:
        prof.disable()


def flush_all(output_dir: str, rank: int) -> None:
    """Dump accumulated profiles to disk. Called once from flush()."""
    for name, prof in _PROFILES.items():
        path = os.path.join(output_dir, f"cpu_{name}_rank{rank}.prof")
        try:
            prof.dump_stats(path)
        except Exception:
            pass
