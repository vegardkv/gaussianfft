"""Compare global and automatic-neighborhood kriging.

Run: python examples/compare_auto_neighborhood.py
Edit n_obs below to change the number of observations.
Reports wall-clock time for each prediction call.
"""

from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np

import gaussianfft as grf


def main():
    nx, ny = 100, 100
    dx, dy = 10.0, 10.0
    n_obs = 200
    seed = 42
    variogram = grf.variogram('matern52', main_range=200.0, perp_range=100.0)

    grf.seed(seed)
    field = np.asarray(grf.simulate(variogram, nx, dx, ny, dy)).reshape((nx, ny), order='F')
    rng = np.random.default_rng(seed)
    ix, iy = np.unravel_index(rng.choice(nx * ny, n_obs, replace=False), (nx, ny))
    obs_locations = np.column_stack((ix * dx, iy * dy))
    obs_values = field[ix, iy]
    obs_uncertainties = np.zeros(n_obs)

    start = perf_counter()
    global_prediction, _ = grf.predict(
        variogram, (nx, ny), (dx, dy), obs_locations, obs_values, obs_uncertainties,
    )
    global_elapsed = perf_counter() - start
    start = perf_counter()
    auto_prediction, _ = grf.predict(
        variogram, (nx, ny), (dx, dy), obs_locations, obs_values, obs_uncertainties,
        use_neighborhood=True,
    )
    auto_elapsed = perf_counter() - start
    print(f'Kriging (all observations): {global_elapsed:.6f} s')
    print(f'Kriging (auto): {auto_elapsed:.6f} s')
    difference = auto_prediction - global_prediction
    print(f'Max absolute difference: {np.max(np.abs(difference)):.6g}')
    print(f'RMS difference: {np.sqrt(np.mean(difference**2)):.6g}')

    fig, axes = plt.subplots(2, 2, figsize=(11, 9), layout='constrained')
    extent = (-dx / 2, (nx - 0.5) * dx, -dy / 2, (ny - 0.5) * dy)
    vmin = min(field.min(), global_prediction.min(), auto_prediction.min())
    vmax = max(field.max(), global_prediction.max(), auto_prediction.max())
    for ax, surface, title in zip(
        axes.flat,
        (field, global_prediction, auto_prediction),
        ('Random field and observations', 'Kriging: all observations', 'Kriging: auto'),
    ):
        image = ax.imshow(
            surface.T, origin='lower', extent=extent, cmap='viridis', vmin=vmin, vmax=vmax,
        )
        ax.scatter(obs_locations[:, 0], obs_locations[:, 1], s=5, c='black')
        ax.set_title(title)
        fig.colorbar(image, ax=ax)

    limit = max(float(np.max(np.abs(difference))), np.finfo(float).eps)
    image = axes[1, 1].imshow(
        difference.T, origin='lower', extent=extent, cmap='RdBu_r', vmin=-limit, vmax=limit,
    )
    axes[1, 1].set_title('Difference: auto - all observations')
    fig.colorbar(image, ax=axes[1, 1])
    for ax in axes.flat:
        ax.set_xlabel('x')
        ax.set_ylabel('y')
    plt.show()


if __name__ == '__main__':
    main()
