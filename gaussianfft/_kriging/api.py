from __future__ import annotations

from typing import Literal

from gaussianfft._kriging.grid_spec import GridSpec
from gaussianfft._kriging.kriging_toolkit import OrdinaryKriging, SimpleKriging
from gaussianfft._kriging.observation_data import ObservationData
from gaussianfft._kriging.neighborhood import Neighborhood


KrigingType = Literal["simple", "ordinary"]


def _build_kriging(
    method: KrigingType, mean: float, chunk_shape: tuple[int, ...] | None,
    padding_range: tuple[float, ...] | None, max_obs_gls: int,
):
    neighborhood = Neighborhood.from_parameters(chunk_shape, padding_range)
    if method == 'simple':
        return SimpleKriging(mean=mean, neighborhood=neighborhood)
    if method == 'ordinary':
        return OrdinaryKriging(neighborhood=neighborhood, max_obs_gls=max_obs_gls)
    raise ValueError(
        f"Unknown kriging method: {method!r}. Choose from 'simple' or 'ordinary'"
    )


def predict_impl(
        variogram, grid_shape: tuple[int, ...], grid_spacing: tuple[float, ...],
        obs_locations, obs_values, obs_uncertainties, *,
        method: KrigingType = 'simple', mean: float = 0.0, calculate_stddev=False,
        chunk_shape: tuple[int, ...] | None = None,
        padding_range: tuple[float, ...] | None = None, max_obs_gls: int = 2000,
):
    """Predict on a grid defined by shape and spacing tuples in x, y, z order."""
    grid = GridSpec(grid_shape, grid_spacing)
    obs = ObservationData(obs_locations, obs_values, obs_uncertainties)
    estimator = _build_kriging(method, mean, chunk_shape, padding_range, max_obs_gls)
    return estimator.predict(grid, obs, variogram, calculate_stddev=calculate_stddev)


def simulate_impl(
        variogram, grid_shape: tuple[int, ...], grid_spacing: tuple[float, ...],
        obs_locations, obs_values, obs_uncertainties, *,
        method: KrigingType = 'simple', mean: float = 0.0, n_sim=1,
        chunk_shape: tuple[int, ...] | None = None,
        padding_range: tuple[float, ...] | None = None, max_obs_gls: int = 2000,
):
    """Simulate on a grid defined by shape and spacing tuples in x, y, z order."""
    grid = GridSpec(grid_shape, grid_spacing)
    obs = ObservationData(obs_locations, obs_values, obs_uncertainties)
    estimator = _build_kriging(method, mean, chunk_shape, padding_range, max_obs_gls)
    return estimator.simulate(grid, obs, variogram, n_sim=n_sim)
