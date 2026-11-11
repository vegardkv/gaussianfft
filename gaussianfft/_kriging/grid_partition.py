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
    """Partition grid nodes; select observations in padded, axis-aligned cell bounds.

    Cell bounds extend half a spacing beyond the outermost nodes of each chunk.
    Neighborhood bounds are inclusive. Without chunking, all observations are used,
    including observations outside the grid.
    """

    def __init__(
        self,
        neighborhood: Neighborhood | None | Literal['auto'],
        obs: ObservationData,
        variogram: Covariance | None,  # Only needed when neighborhood is 'auto'
    ):
        assert (neighborhood == 'auto' and variogram is not None) or neighborhood != 'auto'
        self._neighborhood = neighborhood
        self._variogram = variogram
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
        if self._neighborhood == 'auto':
            # Automatically determine chunk shape and padding
            max_padding = tuple(np.array(grid.shape) * np.array(grid.spacing))
            padding = _find_padding(
                self._variogram,
                len(grid.shape),
                max_padding=max_padding,
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
    ratios = np.array(padding) / padding[0]

    def _chunk_shape(cx: float) -> tuple[int, ...]:
        # cx corresponds to the first chunk dimension, the other dimensions are scaled proportionally
        chunk_shape = (cx * ratios).round().astype(int)
        chunk_shape = np.clip(chunk_shape, 1, grid.shape)
        return tuple(int(size) for size in chunk_shape)

    def _estimate_time(cx: float):
        chunk_shape = _chunk_shape(cx)
        t_current = 0
        for chunk in _construct_chunks(chunk_shape, padding, grid, observation_tree):
            t_current += _estimate_chunk_time(chunk)
        return t_current

    # Treat this as a floating-point optimization problem. The solution is highly
    # hardware dependent, so a perfectly accurate solution is not really necessary
    solution = minimize_scalar(
        _estimate_time, bounds=(1e-3, grid.shape[0]), method='bounded'
    )

    return _chunk_shape(solution.x)


def _find_padding(
        variogram: Covariance,
        ndim: int,
        max_padding: tuple[float, ...],
) -> tuple[float, ...]:
    """Approximate the axis-aligned 5% correlation distance, capped by physical grid extents."""
    # TODO: This does take into account anisotropic variograms
    # TODO: Padding matches variogram range, but should instead reflect the desired 
    #       accuracy-tradeoff when using common data neighborhoods
    origin = np.zeros((1, ndim))

    def _correlation(distance: float, axis: int) -> float:
        point = np.zeros((1, ndim))
        point[0, axis] = distance
        return float(variogram.corr_matrix(point, origin)[0, 0])

    # Compute the axis-aligned padding in each dimension
    sol_x = minimize_scalar(
        lambda x: (_correlation(x, 0) - 0.05) ** 2,
        bounds=(0, max_padding[0]), method='bounded',
    )
    if ndim == 1:
        return (sol_x.x,)
    sol_y = minimize_scalar(
        lambda y: (_correlation(y, 1) - 0.05) ** 2,
        bounds=(0, max_padding[1]), method='bounded',
    )
    if ndim == 2:
        return (sol_x.x, sol_y.x)
    sol_z = minimize_scalar(
        lambda z: (_correlation(z, 2) - 0.05) ** 2,
        bounds=(0, max_padding[2]), method='bounded',
    )
    return (sol_x.x, sol_y.x, sol_z.x)


def _estimate_chunk_time(chunk: GridChunk) -> float:
    # See examples/estimate_kriging_time.py for how these constants were estimated
    #   T_cov_obs   = C_cov_obs   * N_obs^2
    #   T_cross_cov = C_cross_cov * N_obs   * N_grid
    #   T_solve     = C_solve     * N_obs^3
    _C_COV_OBS = 0
    _C_SOLVE = 0
    _C_CROSS_COV = 2.273822e-08
    n_obs = len(chunk.observation_indices)
    n_grid = np.prod(chunk.grid.shape)
    t_cov_obs = _C_COV_OBS * n_obs**2
    t_cross_cov = _C_CROSS_COV * n_obs * n_grid
    t_solve = _C_SOLVE * n_obs**3
    return t_cov_obs + t_cross_cov + t_solve
