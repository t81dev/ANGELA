"""Simplified math helpers for ToCA simulations."""
from __future__ import annotations

from typing import Iterable
import numpy as np

def phi_coherence(values: Iterable[float]) -> float:
    """Compute a simple coherence score in [0,1] for the given values."""
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return 0.0
    norm = np.linalg.norm(arr)
    return float(norm / arr.size) if arr.size else 0.0
