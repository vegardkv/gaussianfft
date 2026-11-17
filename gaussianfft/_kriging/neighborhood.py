from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Neighborhood:
    """Internal partition settings: node counts and physical padding per axis."""

    chunk_shape: tuple[int, ...]
    padding: tuple[float, ...]
