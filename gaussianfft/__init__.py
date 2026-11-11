from ._version import __version__
from enum import Enum
from importlib.util import find_spec
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gaussianfft._kriging import api

if find_spec("numpy") is None:
    raise ImportError("gaussianfft requires NumPy to be installed")

import gaussianfft._platform  # noqa: F401  (sets up DLL paths before loading C extension)

import _gaussianfft

from _gaussianfft import *


class VariogramType(Enum):
    GAUSSIAN = 'gaussian'
    EXPONENTIAL = 'exponential'
    GENERAL_EXPONENTIAL = 'general_exponential'
    SPHERICAL = 'spherical'
    MATERN_32 = 'matern32'
    MATERN_52 = 'matern52'
    MATERN_72 = 'matern72'
    CONSTANT = 'constant'


def variogram(type, *args, **kwargs):
    if isinstance(type, Enum):
        type = type.value
    return _gaussianfft.variogram(type, *args, **kwargs)


__all__ = [
    'variogram', 'simulate', 'seed', 'advanced', 'simulation_size',
    'quote', 'Variogram', 'VariogramType', 'util', 'SizeTVector', 'DoubleVector',
    'conditional_simulate', 'predict',
    '__version__',
]


def conditional_simulate(
        variogram_,
        grid_shape: tuple[int, ...],
        grid_spacing: tuple[float, ...],
        obs_locations,
        obs_values,
        obs_uncertainties,
        *,
        simple_kriging_mean=0.0,
        n=1,
        seed=None,
        method: 'api.KrigingType' = 'simple',
        cdn_chunk_shape: 'tuple[int, ...] | None' = None,
        cdn_padding_range: 'tuple[float, ...] | None' = None,
        ord_kriging_max_obs_gls: int = 2000,
):
    """Generate conditional fields using shape and spacing tuples in x, y, z order."""
    from gaussianfft._kriging import api

    if seed is not None:
        _gaussianfft.seed(seed)
    return api.simulate(
        variogram_, grid_shape, grid_spacing, obs_locations, obs_values, obs_uncertainties,
        mean=simple_kriging_mean, n_sim=n, method=method,
        chunk_shape=cdn_chunk_shape, padding_range=cdn_padding_range, max_obs_gls=ord_kriging_max_obs_gls,
    )


def predict(
        variogram_,
        grid_shape: tuple[int, ...],
        grid_spacing: tuple[float, ...],
        obs_locations,
        obs_values,
        obs_uncertainties,
        *,
        simple_kriging_mean=0.0,
        method: 'api.KrigingType' = 'simple',
        calculate_stddev=False,
        cdn_chunk_shape: 'tuple[int, ...] | None' = None,
        cdn_padding_range: 'tuple[float, ...] | None' = None,
        ord_kriging_max_obs_gls: int = 2000,
):
    """Predict a field using shape and spacing tuples in x, y, z order."""
    from gaussianfft._kriging import api

    return api.predict(
        variogram_, grid_shape, grid_spacing, obs_locations, obs_values, obs_uncertainties,
        mean=simple_kriging_mean, method=method, calculate_stddev=calculate_stddev,
        chunk_shape=cdn_chunk_shape, padding_range=cdn_padding_range, max_obs_gls=ord_kriging_max_obs_gls,
    )
