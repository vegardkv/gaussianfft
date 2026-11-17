from __future__ import annotations

import numpy as np

import gaussianfft
from gaussianfft._kriging.covariance import Covariance
from gaussianfft._kriging.grid_spec import GridSpec


# Memory budget for internal allocations during variogram computations.
# Note that this does not take into account the memory used by other parts
# of the program
_MEMORY_BUDGET = 5 * 1024**3


class Variogram(Covariance):
    """Stationary variogram, wrapping a ``gaussianfft.Variogram``."""

    def __init__(self, variogram: gaussianfft.Variogram):
        self._variogram = variogram

    @classmethod
    def from_parameters(
        cls, type, main_range: float, perp_range: float = -1.0, depth_range: float = -1.0, *,
        azimuth: float = 0.0, dip: float = 0.0, power: float = 1.5,
    ) -> Variogram:
        """Create from ``gaussianfft.variogram`` parameters."""
        return cls(gaussianfft.variogram(
            type, main_range=main_range, perp_range=perp_range, depth_range=depth_range,
            azimuth=azimuth, dip=dip, power=power,
        ))

    @property
    def variogram(self) -> gaussianfft.Variogram:
        """The wrapped ``gaussianfft.Variogram``."""
        return self._variogram

    def corr_matrix(self, points, locations) -> np.ndarray:
        points = np.asarray(points, dtype=float)
        locations = np.asarray(locations, dtype=float)
        bytes_per_location = len(points) * (points.shape[1] + 1) * points.itemsize
        if bytes_per_location * len(locations) <= _MEMORY_BUDGET:
            # Memory budget is not exceeded, calculate for all locations at once.
            return _non_chunked_corr_matrix(self._variogram, points, locations)

        # Memory budget is exceeded, calculate in chunks. Note that the size of
        # `corr` is extracted from the budget.
        corr = np.empty((len(points), len(locations)))
        chunk_size = max(1, (_MEMORY_BUDGET - corr.nbytes) // bytes_per_location)
        for start in range(0, len(locations), chunk_size):
            stop = start + chunk_size
            corr[:, start:stop] = _non_chunked_corr_matrix(
                self._variogram, points, locations[start:stop]
            )
        return corr

    def simulate(self, grid: GridSpec) -> np.ndarray:
        grid_args = tuple(value for axis in zip(grid.shape, grid.spacing) for value in axis)
        raw = gaussianfft.simulate(self._variogram, *grid_args)
        return np.asarray(raw).reshape(grid.shape, order='F')



def _non_chunked_corr_matrix(
    variogram: gaussianfft.Variogram,
    points: np.ndarray,
    locations: np.ndarray,
) -> np.ndarray:
    # The readable implementation:
    #   points = np.asarray(points, dtype=float)
    #   corr = np.empty((len(points), len(locations)))
    #   for j, location in enumerate(np.asarray(locations, dtype=float)):
    #       corr[:, j] = self._variogram.corr_array(points - location)
    #   return corr

    # The performant (but memory-intensive) implementation:
    points = np.asarray(points, dtype=float)
    locations = np.asarray(locations, dtype=float)
    offsets = points[:, None, :] - locations[None, :, :]
    return variogram.corr_array(
        offsets.reshape(-1, points.shape[1])
    ).reshape(len(points), len(locations))
