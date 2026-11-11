from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Neighborhood:
    """Internal partition settings: node counts and physical padding per axis."""

    chunk_shape: tuple[int, ...]
    padding: tuple[float, ...]

    @classmethod
    def from_parameters(
        cls, chunk_shape: tuple[int, ...] | None, padding_range: tuple[float, ...] | None,
    ) -> Neighborhood | None:
        """Validate and normalize public inputs; omitted settings mean no partitioning."""
        if chunk_shape is None and padding_range is None:
            return None
        if chunk_shape is None or padding_range is None:
            raise ValueError("chunk_shape and padding_range must be specified together")
        shape = tuple(chunk_shape)
        if not 1 <= len(shape) <= 3 or any(
            not isinstance(size, (int, np.integer))
            or isinstance(size, (bool, np.bool_)) or size <= 0 for size in shape
        ):
            raise ValueError("chunk_shape must contain 1 to 3 positive integers")
        padding = np.asarray(padding_range, dtype=float)
        if (
            padding.shape != (len(shape),)
            or not np.all(np.isfinite(padding)) or np.any(padding < 0)
        ):
            raise ValueError("padding_range must contain one non-negative finite distance per chunk dimension")
        return cls(shape, tuple(float(value) for value in padding))
