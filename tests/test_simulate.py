import gaussianfft as grf
import numpy as np
import pytest

def test_simulation_size_1d():
    v = grf.variogram('spherical', 100.0)
    a = grf.simulation_size(v, 100, 1.0)
    assert len(a) == 1
    assert a[0] == 201

def test_simulation_size_2d():
    v = grf.variogram('spherical', 2113.0, 997.0)
    a = grf.simulation_size(v, 132, 10.0, 657, 12.0)
    assert len(a) == 2
    assert a[0] == 406
    assert a[1] == 741

def test_simulation_size_3d_equal():
    v = grf.variogram('spherical', 1000.0, 1000.0, 1000.0)
    a = grf.simulation_size(v, 100, 10.0, 100, 10.0, 100, 10.0)
    assert len(a) == 3
    assert a[0] == 201
    assert a[1] == 201
    assert a[2] == 201

def test_simulation_size_3d_uneven():
    v = grf.variogram('spherical', 1000.0, 250.0, 125.0)
    a = grf.simulation_size(v, 507, 50.0, 305, 17.0, 103, 4.0)
    assert len(a) == 3
    assert a[0] == 528
    assert a[1] == 320
    assert a[2] == 135


@pytest.mark.parametrize(
    ("variogram_args", "grid_args", "expected"),
    [
        ((10.0, 2.0, 1.0, 45.0, 0.0), (100, 1.0), [111]),
        ((10.0, 2.0, 1.0, 45.0, 0.0), (100, 1.0, 100, 1.0), [111, 111]),
        ((10.0, 2.0, 1.0, 45.0, 0.0), (2, 1.0, 2, 1.0), [20, 20]),
        ((100.0, 20.0, 10.0, 0.0, 45.0), (100, 1.0, 100, 1.0), [201, 121]),
        ((100.0, 20.0, 10.0, 0.0, 45.0), (100, 1.0, 100, 1.0, 100, 1.0),
         [201, 121, 172]),
        ((2.0, 10.0, 1.0, 0.0, 0.0), (100, 1.0, 100, 1.0), [111, 111]),
    ],
)
def test_simulation_size_uses_axis_aligned_extents(variogram_args, grid_args, expected):
    v = grf.variogram('spherical', *variogram_args)
    assert list(grf.simulation_size(v, *grid_args)) == expected


def test_explicit_padding_matches_automatic_padding():
    v = grf.variogram('spherical', 10.0, 2.0, 1.0, 45.0)
    grid_args = (20, 1.0, 20, 1.0)
    size = grf.simulation_size(v, *grid_args)
    grf.seed(123)
    automatic = grf.simulate(v, *grid_args)
    grf.seed(123)
    explicit = grf.advanced.simulate(
        v, *grid_args, padx=size[0] - grid_args[0], pady=size[1] - grid_args[2],
    )
    np.testing.assert_array_equal(automatic, explicit)
