from __future__ import annotations

import numpy as np
try:
    import scipy.linalg
    import scipy.ndimage
except ImportError:
    raise ImportError("Scipy is not installed, and is required to use these utilities")

from gaussianfft._kriging.covariance import Covariance
from gaussianfft._kriging.grid_spec import GridSpec
from gaussianfft._kriging.observation_data import ObservationData


class FactorizationError(ValueError):
    def __init__(self):
        super().__init__(
            "Failed to factorize observation covariance matrix. "
            "The matrix is likely not positive definite. This can happen for "
            "nearly duplicate observation locations with too small observation "
            "uncertainties. Consider increasing obs_uncertainties or removing "
            "duplicate/near-duplicate observations."
        )


def _build_obs_cov_matrix(variogram: Covariance, obs_locations, obs_uncertainties):
    cov = variogram.corr_matrix(obs_locations, obs_locations)
    cov[np.diag_indices(len(obs_locations))] += obs_uncertainties ** 2 + 1e-10
    return cov


def _cholesky_factor(cov):
    """Factorize an observation covariance matrix, reporting non-positive definiteness."""
    try:
        return scipy.linalg.cho_factor(cov)
    except scipy.linalg.LinAlgError as exc:
        raise FactorizationError() from exc


class CholeskyCache:
    """Cache factors by ordered observation indices for one kriging operation."""

    def __init__(self, obs: ObservationData, variogram: Covariance):
        self._obs = obs
        self._variogram = variogram
        self._factors: dict[tuple[int, ...], tuple[np.ndarray, bool]] = {}

    def get_factor(self, obs_indices: np.ndarray) -> tuple[np.ndarray, bool]:
        key = tuple(obs_indices.tolist())
        factor = self._factors.get(key)
        if factor is None:
            factor = _cholesky_factor(_build_obs_cov_matrix(
                self._variogram, self._obs.locations[obs_indices], self._obs.uncertainties[obs_indices],
            ))
            self._factors[key] = factor
        return factor


def obs_cov_times_vector(variogram: Covariance, obs, vector):
    """Multiply by observation covariance without allocating the full matrix."""
    result = np.empty(len(obs.locations))
    for index in range(len(obs.locations)):
        correlations = variogram.corr_matrix(obs.locations, obs.locations[index:index + 1])
        result[index] = correlations[:, 0] @ vector
    result += (obs.uncertainties ** 2 + 1e-10) * vector
    return result


def build_cross_cov_matrix(variogram: Covariance, grid_coords, obs_locations):
    """Cross-covariance between supplied grid points and observations."""
    return variogram.corr_matrix(grid_coords, obs_locations)


def obs_interp_coords(obs_locations, grid: GridSpec):
    relative_locations = obs_locations - grid.origin
    grid_max = (np.asarray(grid.shape) - 1) * grid.spacing
    if np.any(relative_locations < 0) or np.any(relative_locations > grid_max):
        raise ValueError("obs_locations has values outside grid bounds for simulation")
    return (relative_locations / grid.spacing).T


def interpolate_at_obs(field, obs_coords):
    return scipy.ndimage.map_coordinates(field, obs_coords, order=1)


def simulate_uncond(variogram: Covariance, grid: GridSpec):
    return variogram.simulate(grid)
