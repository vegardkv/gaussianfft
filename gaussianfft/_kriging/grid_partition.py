from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterator

import numpy as np
from scipy.spatial import cKDTree

from gaussianfft._kriging.grid_spec import GridSpec
from gaussianfft._kriging.observation_data import ObservationData
from gaussianfft._kriging.neighborhood import Neighborhood


@dataclass
class GridChunk:
    slices: tuple[slice, ...]
    grid: GridSpec
    observation_indices: np.ndarray


class GridPartition:
    """Partition grid nodes; select observations in padded, axis-aligned cell bounds.

    Cell bounds extend half a spacing beyond the outermost nodes of each chunk.
    Neighborhood bounds are inclusive. Without chunking, all observations are used,
    including observations outside the grid.
    """

    def __init__(
        self, neighborhood: Neighborhood | None, obs: ObservationData,
    ):
        self._neighborhood = neighborhood
        self._locations = obs.locations
        self._tree = None
        if neighborhood is not None:
            if not isinstance(neighborhood, Neighborhood):
                raise TypeError("neighborhood must be a Neighborhood or None")
            if len(neighborhood.chunk_shape) != obs.locations.shape[1]:
                raise ValueError("neighborhood must have one entry per observation dimension")
            self._tree = cKDTree(obs.locations)

    def partitions(self, grid: GridSpec) -> Iterator[GridChunk]:
        if self._locations.shape[1] != len(grid.shape):
            raise ValueError(f"obs_locations must have {len(grid.shape)} columns for this grid")
        chunk_shape = grid.shape if self._neighborhood is None else self._neighborhood.chunk_shape
        starts = (range(0, size, chunk) for size, chunk in zip(grid.shape, chunk_shape))
        for start in product(*starts):
            shape = tuple(min(chunk, size - offset) for offset, chunk, size in zip(
                start, chunk_shape, grid.shape,
            ))
            origin = grid.origin + np.asarray(start) * grid.spacing
            local_grid = GridSpec(shape, grid.spacing, origin)
            slices = tuple(slice(offset, offset + size) for offset, size in zip(start, shape))
            if self._tree is None:
                indices = np.arange(len(self._locations))
            else:
                assert self._neighborhood is not None
                lower = origin - np.asarray(grid.spacing) / 2 - self._neighborhood.padding
                upper = origin + (np.asarray(shape) - 0.5) * grid.spacing + self._neighborhood.padding
                center = (lower + upper) / 2
                radius = np.nextafter(np.max((upper - lower) / 2), np.inf)
                candidates = np.asarray(
                    self._tree.query_ball_point(center, radius, p=np.inf), dtype=int,
                )
                locations = self._locations[candidates]
                inside = np.all((locations >= lower) & (locations <= upper), axis=1)
                indices = np.sort(candidates[inside])
            yield GridChunk(slices, local_grid, indices)
