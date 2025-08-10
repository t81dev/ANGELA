"""Vector utility helpers for ANGELA modules."""
from __future__ import annotations

from typing import Iterable, List
import numpy as np

def normalize_vectors(vectors: Iterable[Iterable[float]]) -> List[List[float]]:
    """Normalize an iterable of vectors using L2 norm."""
    result: List[List[float]] = []
    for vec in vectors:
        arr = np.asarray(list(vec), dtype=float)
        norm = np.linalg.norm(arr)
        if norm == 0:
            result.append(arr.tolist())
        else:
            result.append((arr / norm).tolist())
    return result
