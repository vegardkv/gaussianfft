from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Literal
from collections.abc import Iterator

import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import minimize_scalar

from gaussianfft._kriging.covariance import Covariance
from gaussianfft._kriging.grid_spec import GridSpec
from gaussianfft._kriging.observation_data import ObservationData
from gaussianfft._kriging.neighborhood import Neighborhood


@dataclass
class GridChunk:
    slices: tuple[slice, ...]
    grid: GridSpec
    observation_indices: np.ndarray


class GridPartition:
    """
    Partition grid nodes; select observations in padded, axis-aligned cell bounds.
    """

    def __init__(
        self,
        neighborhood: Neighborhood | None | Literal['auto'],
        obs: ObservationData,
        variogram: Covariance | None,  # Only needed when neighborhood is 'auto'
        auto_corr_threshold: float = 0.01,  # Only used when neighborhood is 'auto'
    ):
        assert (neighborhood == 'auto' and variogram is not None) or neighborhood != 'auto'
        self._neighborhood = neighborhood
        self._variogram = variogram
        self._auto_corr_threshold = auto_corr_threshold
        self._locations = obs.locations
        self._tree = None
        if neighborhood is not None:
            self._tree = cKDTree(obs.locations)

    def partitions(self, grid: GridSpec) -> Iterator[GridChunk]:
        if self._locations.shape[1] != len(grid.shape):
            raise ValueError(f"obs_locations must have {len(grid.shape)} columns for this grid")

        # No neighborhood specified, use all observations
        if self._neighborhood is None:
            yield GridChunk(
                tuple(slice(0, size) for size in grid.shape),
                grid,
                np.arange(len(self._locations)),
            )
            return

        # Automatically determine the optimal chunking and padding
        elif self._neighborhood == 'auto':
            max_padding = tuple(np.array(grid.shape) * np.array(grid.spacing))
            padding = _find_padding(
                self._variogram,
                max_padding,
                corr_threshold=self._auto_corr_threshold,
            )
            chunk_shape = _find_optimal_chunk_shape(
                padding=padding,
                grid=grid,
                observation_tree=self._tree,
            )
            # TODO: If the "optimal" chunk shape is close to the full grid shape,
            # we might skip chunking altogether
            yield from _construct_chunks(
                chunk_shape,
                padding,
                grid,
                self._tree,
            )
            return

        # Use the specified neighborhood for chunking and padding
        else:
            yield from _construct_chunks(
                self._neighborhood.chunk_shape,
                self._neighborhood.padding,
                grid,
                self._tree,
            )


def _construct_chunks(
        chunk_shape: tuple[int, ...],
        padding: tuple[float, ...],
        grid: GridSpec,
        observation_tree: cKDTree,
) -> Iterator[GridChunk]:
    starts = (range(0, size, chunk) for size, chunk in zip(grid.shape, chunk_shape))
    for start in product(*starts):
        shape = tuple(min(chunk, size - offset) for offset, chunk, size in zip(
            start, chunk_shape, grid.shape,
        ))
        origin = grid.origin + np.asarray(start) * grid.spacing
        local_grid = GridSpec(shape, grid.spacing, origin)
        slices = tuple(slice(offset, offset + size) for offset, size in zip(start, shape))
        
        lower = origin - np.asarray(grid.spacing) / 2 - padding
        upper = origin + (np.asarray(shape) - 0.5) * grid.spacing + padding
        center = (lower + upper) / 2
        radius = np.nextafter(np.max((upper - lower) / 2), np.inf)
        candidates = np.asarray(
            observation_tree.query_ball_point(center, radius, p=np.inf), dtype=int,
        )
        locations = observation_tree.data[candidates]  # May need observations? Is .data correct?
        # Why do we need this additional filter?
        inside = np.all((locations >= lower) & (locations <= upper), axis=1)
        indices = np.sort(candidates[inside])
        yield GridChunk(slices, local_grid, indices)


def _find_optimal_chunk_shape(
        padding: tuple[float, ...],
        grid: GridSpec,
        observation_tree: cKDTree,
) -> tuple[int, ...]:
    """Choose among a small set of sizes without constructing candidate chunks."""
    shape = np.asarray(grid.shape)
    extent = shape * grid.spacing
    n_grid = float(np.prod(shape))
    padding_cells = np.maximum(np.asarray(padding) / grid.spacing, 1.0)
    ratios = padding_cells / padding_cells[0]

    # `c_overhead` represents the overhead related to constructing each chunk,
    # which in practice is dominated by the spatial query: query_ball_point.
    # The total overhead depends on the number of observations, but the cKDTree
    # ensures efficient spatial queries, so the overhead is sub-linear in nature,
    # most likely proportional to log(#observations) instead. For simplicity,
    # we use a fixed constant, determined by
    #   `examples/estimate_grid_partition_overhead.py`.
    C_OVERHEAD = 3.3e-05
    # The remaining constants represent the computational cost associated with
    # each chunk, and is estimated by
    #   `examples/estimate_kriging_timing.py`.
    C_COV_OBS = 0.0
    C_SOLVE = 0.0
    C_CROSS_COV = 2.273822e-08

    def _chunk_shape(cx: float) -> tuple[int, ...]:
        chunk_shape = (cx * ratios).round().astype(int)
        chunk_shape = np.clip(chunk_shape, 1, grid.shape)
        return tuple(int(size) for size in chunk_shape)

    def _estimate_time(chunk_shape: tuple[int, ...]) -> float:
        n_chunks = float(np.prod(np.ceil(shape / chunk_shape)))
        padded_extent = np.minimum(
            np.asarray(chunk_shape) * grid.spacing + 2 * np.asarray(padding), extent,
        )
        n_obs = observation_tree.n * float(np.prod(padded_extent / extent))
        return (
            n_chunks * (C_OVERHEAD + C_COV_OBS * n_obs**2 + C_SOLVE * n_obs**3)
            + C_CROSS_COV * n_obs * n_grid
        )

    max_cx = float(np.max(shape / ratios))
    candidates = [grid.shape] + [_chunk_shape(cx) for cx in np.geomspace(1.0, max_cx, 16)]
    return min(candidates, key=_estimate_time)


def _find_padding(
        variogram: Covariance,
        max_padding: tuple[float, ...],
        *,
        corr_threshold: float = 0.01,
) -> tuple[float, ...]:
    """ Estimates the axis-aligned padding distance based on a target correlation threshold. """
    # TODO: This does take into account non-stationary variograms
    # TODO: Padding matches variogram range, but should instead reflect the desired 
    #       accuracy-tradeoff when using common data neighborhoods
    ndim = len(max_padding)
    origin = np.zeros((1, ndim))

    def _correlation(distance: float, axis: int) -> float:
        point = np.zeros((1, ndim))
        point[0, axis] = distance
        return float(variogram.corr_matrix(point, origin)[0, 0])

    # Compute the axis-aligned padding in each dimension
    sol_x = minimize_scalar(
        lambda x: (_correlation(x, 0) - corr_threshold) ** 2,
        bounds=(0, max_padding[0]), method='bounded',
    )
    if ndim == 1:
        return (sol_x.x,)
    sol_y = minimize_scalar(
        lambda y: (_correlation(y, 1) - corr_threshold) ** 2,
        bounds=(0, max_padding[1]), method='bounded',
    )
    if ndim == 2:
        return (sol_x.x, sol_y.x)
    sol_z = minimize_scalar(
        lambda z: (_correlation(z, 2) - corr_threshold) ** 2,
        bounds=(0, max_padding[2]), method='bounded',
    )
    return (sol_x.x, sol_y.x, sol_z.x)
