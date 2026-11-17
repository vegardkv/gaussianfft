import platform

import numpy as np
import pytest

if platform.python_implementation() == 'PyPy':
    pytest.skip('Kriging requires SciPy, which is unavailable on PyPy', allow_module_level=True)

from gaussianfft._kriging.grid_partition import GridPartition
from gaussianfft._kriging.grid_spec import GridSpec
from gaussianfft._kriging.neighborhood import Neighborhood
from gaussianfft._kriging.observation_data import ObservationData
from gaussianfft._kriging.variogram import Variogram


@pytest.fixture
def observations():
    locations = np.array([[0, 0], [4, 4], [5, 5], [9, 9], [-10, 0]])
    return ObservationData(locations, np.zeros(5), np.zeros(5))


def test_no_neighborhood_yields_full_grid(observations):
    grid = GridSpec((10, 10), (1.0, 1.0))
    chunks = list(GridPartition(None, observations, None).partitions(grid))

    assert len(chunks) == 1
    assert chunks[0].grid.shape == grid.shape
    assert chunks[0].slices == (slice(0, 10), slice(0, 10))
    np.testing.assert_array_equal(chunks[0].observation_indices, np.arange(5))


def test_zero_padding_uses_only_observations_in_each_chunk(observations):
    grid = GridSpec((10, 10), (1.0, 1.0))
    neighborhood = Neighborhood((5, 5), (0.0, 0.0))
    chunks = list(GridPartition(neighborhood, observations, None).partitions(grid))

    assert len(chunks) == 4
    assert all(chunk.grid.shape == (5, 5) for chunk in chunks)
    for chunk, expected in zip(chunks, ([0, 1], [], [], [2, 3])):
        np.testing.assert_array_equal(chunk.observation_indices, expected)


def test_large_padding_uses_all_observations_in_each_chunk(observations):
    grid = GridSpec((10, 10), (1.0, 1.0))
    neighborhood = Neighborhood((5, 5), (100.0, 100.0))
    chunks = list(GridPartition(neighborhood, observations, None).partitions(grid))

    assert len(chunks) == 4
    for chunk in chunks:
        np.testing.assert_array_equal(chunk.observation_indices, np.arange(5))


def test_geomspaced_observations_give_uneven_observation_indices():
    locations = np.geomspace(1.0, 100.0, 9)[:, None]
    obs = ObservationData(locations, np.zeros(9), np.zeros(9))
    grid = GridSpec((101,), (1.0,))
    neighborhood = Neighborhood((25,), (0.0,))
    chunks = list(GridPartition(neighborhood, obs, None).partitions(grid))

    assert len(chunks) == 5
    for chunk, expected in zip(chunks, ([0, 1, 2, 3, 4, 5], [6], [7], [], [8])):
        np.testing.assert_array_equal(chunk.observation_indices, expected)


def test_auto_yields_intermediate_sized_chunks():
    grid = GridSpec((100, 100), (10.0, 10.0))
    locations = np.random.default_rng(42).uniform(0.0, 990.0, size=(200, 2))
    obs = ObservationData(locations, np.zeros(200), np.zeros(200))
    variogram = Variogram.from_parameters('matern52', main_range=200.0, perp_range=100.0)
    chunks = list(GridPartition('auto', obs, variogram).partitions(grid))

    assert 1 < len(chunks) < 100
    assert all(np.prod(chunk.grid.shape) > 1 for chunk in chunks)
    assert sum(np.prod(chunk.grid.shape) for chunk in chunks) == np.prod(grid.shape)
