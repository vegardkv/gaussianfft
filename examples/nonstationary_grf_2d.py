"""Simulate and plot a 2D Gaussian field with a spatially varying azimuth."""

import matplotlib.pyplot as plt

import gaussianfft as grf
from gaussianfft._kriging import GridSpec

grf.seed(42)
grid = GridSpec((100, 100), (10.0, 10.0))
variogram = grf.nonstationary_variogram(
    'matern52', main_range=120.0, perp_range=20.0,
    azimuth_callable=lambda xy: 0.06 * xy[:, 1],  # Degrees, increasing with y.
)
field = variogram.simulate(grid)

plt.imshow(field.T, origin='lower', cmap='RdBu_r')
plt.title('2D non-stationary Gaussian random field')
plt.colorbar()
plt.show()
