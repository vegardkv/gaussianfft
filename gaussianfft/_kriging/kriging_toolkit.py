from __future__ import annotations
import abc

import numpy as np
try:
    import scipy.linalg
except ImportError:
    raise ImportError("Scipy is not installed, and is required to use these utilities")

from gaussianfft._kriging.covariance import Covariance
from gaussianfft._kriging.kriging_utils import (
    CholeskyCache,
    build_cross_cov_matrix,
    interpolate_at_obs,
    obs_cov_times_vector,
    obs_interp_coords,
)
from gaussianfft._kriging.grid_partition import GridPartition
from gaussianfft._kriging.grid_spec import GridSpec
from gaussianfft._kriging.observation_data import ObservationData
from gaussianfft._kriging.neighborhood import Neighborhood


# ---------------------------------------------------------------------------
# Kriging classes
# ---------------------------------------------------------------------------


class Kriging(abc.ABC):
    """Kriging interface accepting observations and a variogram per operation."""

    @abc.abstractmethod
    def predict(
        self, grid: GridSpec, obs: ObservationData, variogram: Covariance,
        calculate_stddev=False,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Return a grid-shaped prediction and optional standard deviation."""

    @abc.abstractmethod
    def simulate(
        self, grid: GridSpec, obs: ObservationData, variogram: Covariance,
        n_sim=1,
    ) -> list[np.ndarray]:
        """Generate conditional fields on a grid covering all observation locations."""


class SimpleKriging(Kriging):
    """Simple kriging with a known mean and optional common data neighborhoods.

    Padding selects observations in physical coordinates, not extra output nodes.
    Empty neighborhoods use the prior mean and unit standard deviation.
    Local factors are reused within an operation; no memory-budget batching is
    applied. Unconditional simulations always cover the entire grid.
    """

    def __init__(self, mean: float = 0.0, neighborhood: Neighborhood | None = None):
        self.mean = float(mean)
        self._neighborhood = neighborhood
        self._obs: ObservationData | None = None
        self._variogram: Covariance | None = None
        self._cholesky_cache: CholeskyCache | None = None
        self._partition: GridPartition | None = None

    def _initialize(self, obs: ObservationData, variogram: Covariance) -> None:
        partition = GridPartition(self._neighborhood, obs)
        self._obs = obs
        self._variogram = variogram
        self._cholesky_cache = CholeskyCache(obs, variogram)
        self._partition = partition

    def predict(
        self, grid: GridSpec, obs: ObservationData, variogram: Covariance,
        calculate_stddev=False,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        if obs.locations.shape[1] != len(grid.shape):
            raise ValueError(f"obs_locations must have {len(grid.shape)} columns for this grid")
        self._initialize(obs, variogram)
        residuals = obs.values - self.mean
        correction, stddev = self._krige(grid, residuals, calculate_stddev)
        correction += self.mean
        return correction, stddev

    def simulate(
        self, grid: GridSpec, obs: ObservationData, variogram: Covariance,
        n_sim=1,
    ) -> list[np.ndarray]:
        if obs.locations.shape[1] != len(grid.shape):
            raise ValueError(f"obs_locations must have {len(grid.shape)} columns for this grid")
        self._initialize(obs, variogram)
        obs_coords = obs_interp_coords(obs.locations, grid)
        results = []

        for _ in range(n_sim):
            uncond = self._variogram.simulate(grid)
            uncond_at_obs = interpolate_at_obs(uncond, obs_coords)
            obs_residuals = (obs.values - self.mean) - uncond_at_obs

            correction, _ = self._krige(grid, obs_residuals)
            results.append(self.mean + uncond + correction)
        return results

    def _krige(self, grid: GridSpec, residuals: np.ndarray, calculate_stddev=False):
        """Apply simple kriging to mean-adjusted observation or simulation residuals."""
        obs = self._obs
        if (
            obs is None or self._variogram is None
            or self._partition is None or self._cholesky_cache is None
        ):
            raise RuntimeError("SimpleKriging has not been initialized")
        correction = np.zeros(grid.shape)
        stddev = np.ones(grid.shape) if calculate_stddev else None
        for part in self._partition.partitions(grid):
            indices = part.observation_indices
            if len(indices) == 0:
                continue
            factor = self._cholesky_cache.get_factor(indices)
            weights = scipy.linalg.cho_solve(factor, residuals[indices])
            cross_cov = build_cross_cov_matrix(
                self._variogram, part.grid.coordinates(), obs.locations[indices],
            )
            correction[part.slices] = (cross_cov @ weights).reshape(part.grid.shape)
            if stddev is not None:
                alpha = scipy.linalg.cho_solve(factor, cross_cov.T)
                variance = 1.0 - np.sum(cross_cov * alpha.T, axis=1)
                stddev[part.slices] = np.sqrt(np.maximum(variance, 0.0)).reshape(part.grid.shape)
        return correction, stddev


class OrdinaryKriging(Kriging):
    """Local residual kriging around one globally estimated constant mean.

    GLS is used for at most max_obs_gls observations, diagonal-covariance WLS
    above that threshold. Zero forces WLS. Empty neighborhoods use the global
    mean. Standard deviations include global mean uncertainty and its covariance
    with local residual errors under the full covariance model.
    """

    def __init__(self, neighborhood: Neighborhood | None = None, *, max_obs_gls: int = 2000):
        if (
            not isinstance(max_obs_gls, (int, np.integer))
            or isinstance(max_obs_gls, (bool, np.bool_)) or max_obs_gls < 0
        ):
            raise ValueError("max_obs_gls must be a non-negative integer")
        self.max_obs_gls = max_obs_gls
        self._neighborhood = neighborhood
        self._obs: ObservationData | None = None
        self._variogram: Covariance | None = None
        self._cholesky_cache: CholeskyCache | None = None
        self._partition: GridPartition | None = None
        self._mean_weights: np.ndarray | None = None
        self._covariance_mean: np.ndarray | None = None
        self._mean_variance = 0.0
        self.estimated_mean: float | None = None

    def _initialize(
        self, obs: ObservationData, variogram: Covariance, calculate_stddev=False,
    ) -> None:
        partition = GridPartition(self._neighborhood, obs)
        self._obs = obs
        self._variogram = variogram
        self._cholesky_cache = CholeskyCache(obs, variogram)
        self._partition = partition
        self._mean_weights = None
        self._covariance_mean = None
        self._mean_variance = 0.0
        self.estimated_mean = None
        n_obs = len(obs.locations)
        if n_obs <= self.max_obs_gls:
            factor = self._cholesky_cache.get_factor(np.arange(n_obs))
            weights = scipy.linalg.cho_solve(factor, np.ones(n_obs))
        else:
            with np.errstate(over='ignore'):
                diagonal = 1.0 + obs.uncertainties ** 2 + 1e-10
            if not np.all(np.isfinite(diagonal)):
                raise ValueError("Observation covariance diagonal must be finite")
            weights = 1.0 / diagonal
        normalization = float(weights.sum())
        self._mean_weights = weights / normalization
        self.estimated_mean = float(self._mean_weights @ obs.values)
        if calculate_stddev:
            self._covariance_mean = (
                np.full(n_obs, 1.0 / normalization) if n_obs <= self.max_obs_gls
                else obs_cov_times_vector(variogram, obs, self._mean_weights)
            )
            self._mean_variance = float(self._mean_weights @ self._covariance_mean)

    def predict(
        self, grid: GridSpec, obs: ObservationData, variogram: Covariance,
        calculate_stddev=False,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        if obs.locations.shape[1] != len(grid.shape):
            raise ValueError(f"obs_locations must have {len(grid.shape)} columns for this grid")
        self._initialize(obs, variogram, calculate_stddev)
        return self._krige(grid, obs.values, calculate_stddev)

    def simulate(
        self, grid: GridSpec, obs: ObservationData, variogram: Covariance,
        n_sim=1,
    ) -> list[np.ndarray]:
        if obs.locations.shape[1] != len(grid.shape):
            raise ValueError(f"obs_locations must have {len(grid.shape)} columns for this grid")
        self._initialize(obs, variogram)
        obs_coords = obs_interp_coords(obs.locations, grid)
        results = []

        for _ in range(n_sim):
            uncond = self._variogram.simulate(grid)
            uncond_at_obs = interpolate_at_obs(uncond, obs_coords)
            obs_residuals = obs.values - uncond_at_obs

            correction, _ = self._krige(grid, obs_residuals)
            results.append(uncond + correction)
        return results

    def _krige(self, grid: GridSpec, residuals: np.ndarray, calculate_stddev=False):
        """Apply ordinary kriging to observation values or simulation residuals."""
        obs = self._obs
        if (
            obs is None or self._variogram is None or self._partition is None
            or self._mean_weights is None or self._cholesky_cache is None
        ):
            raise RuntimeError("OrdinaryKriging has not been initialized")
        mean = float(self._mean_weights @ residuals)
        correction = np.full(grid.shape, mean)
        stddev = np.empty(grid.shape) if calculate_stddev else None
        for part in self._partition.partitions(grid):
            indices = part.observation_indices
            coords = part.grid.coordinates()
            cross_cov = build_cross_cov_matrix(
                self._variogram, coords, obs.locations[indices],
            )
            factor = None
            if len(indices):
                factor = self._cholesky_cache.get_factor(indices)
                weights = scipy.linalg.cho_solve(factor, residuals[indices] - mean)
                correction[part.slices] += (cross_cov @ weights).reshape(part.grid.shape)
            if stddev is not None:
                variance = self._prediction_variance(coords, indices, factor, cross_cov)
                stddev[part.slices] = np.sqrt(np.maximum(variance, 0.0)).reshape(part.grid.shape)
        return correction, stddev

    def _prediction_variance(
        self, coords: np.ndarray, indices: np.ndarray,
        factor: tuple[np.ndarray, bool] | None, cross_cov: np.ndarray,
    ) -> np.ndarray:
        if self._obs is None or self._variogram is None or self._mean_weights is None or self._covariance_mean is None:
            raise RuntimeError("OrdinaryKriging uncertainty has not been initialized")
        beta = (
            scipy.linalg.cho_solve(factor, cross_cov.T)
            if factor is not None else np.empty((0, len(coords)))
        )
        variance = 1.0 - np.sum(cross_cov * beta.T, axis=1)
        global_cross_cov = cross_cov if len(indices) == len(self._obs.locations) else build_cross_cov_matrix(
            self._variogram, coords, self._obs.locations,
        )
        q = 1.0 - beta.sum(axis=0)
        # For m = a^T z, include its covariance with the local residual error.
        covariance_error_mean = global_cross_cov @ self._mean_weights - self._covariance_mean[indices] @ beta
        return variance + q ** 2 * self._mean_variance - 2 * q * covariance_error_mean
