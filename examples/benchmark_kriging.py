"""Plot end-to-end kriging speedup against the correlation-work estimate M*M + N*M."""

from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np

import gaussianfft
from gaussianfft._kriging import predict

MODES = {'vectorized': True, 'scalar': False}
GRID_TOTALS = [
    100 ** 2,
    250 ** 2,
    500 ** 2,
    1000 ** 2,
]
OBS_COUNTS = (10, 25, 50, 100)
REPEATS = 3
SEED = 42
OBS_UNCERTAINTY = 0.1
VARIOGRAM_TYPE = 'matern52'
VARIOGRAM_PARAMETERS = {
    'main_range': 300.0,
    'perp_range': 100.0,
    'depth_range': 50.0,
    'azimuth': 30.0,
}
GRID_EXTENT = 900  # = 3x main_range of variogram
DIMENSION_COLORS = {2: 'tab:blue', 3: 'tab:orange'}
OUTPUT_PATH = Path('benchmark_results/kriging_speedup.png')
SHOW_PLOT = True


def benchmark_case(ndims, size, n_obs):
    assert size > 1
    spacing = GRID_EXTENT / (size - 1)
    rng = np.random.default_rng(SEED)
    obs_locations = rng.uniform(0.0, GRID_EXTENT, size=(n_obs, ndims))
    obs_values = rng.standard_normal(n_obs)
    obs_uncertainties = np.full(n_obs, OBS_UNCERTAINTY)
    variogram = gaussianfft.variogram(VARIOGRAM_TYPE, **VARIOGRAM_PARAMETERS)
    grid_args = (size, spacing) * ndims
    timings = {}
    reference = None
    for mode, vectorized in MODES.items():
        elapsed = []
        for _ in range(REPEATS):
            start = perf_counter()
            prediction, _ = predict(
                variogram, *grid_args, obs_locations, obs_values, obs_uncertainties,
                vectorized=vectorized,
            )
            elapsed.append(perf_counter() - start)
            if reference is None:
                reference = prediction
            else:
                np.testing.assert_allclose(prediction, reference, rtol=1e-12, atol=1e-12)
        timings[mode] = float(np.median(elapsed))
        grid_label = 'x'.join([str(size)] * ndims)
        print(f'{mode}: {grid_label}, {n_obs} observations: {timings[mode]:.3f} s', flush=True)
    return timings


def main():
    figure, axis = plt.subplots(figsize=(8, 5), layout='constrained')
    grid_sizes = {
        2: np.sqrt(GRID_TOTALS).astype(int),
        3: (np.array(GRID_TOTALS) ** (1/3)).round().astype(int),
    }
    for ndims, sizes in grid_sizes.items():
        benchmark_case(ndims, sizes[0], OBS_COUNTS[0])
        problem_sizes = []
        speedups = []
        for size in sizes:
            for n_obs in OBS_COUNTS:
                timings = benchmark_case(ndims, size, n_obs)
                n_grid = size ** ndims
                problem_size = n_obs * n_obs + n_grid * n_obs
                speedup = timings['scalar'] / timings['vectorized']
                problem_sizes.append(problem_size)
                speedups.append(speedup)
                print(f'Problem size: {problem_size:,}; speedup: {speedup:.2f}x', flush=True)
        axis.scatter(problem_sizes, speedups, label=f'{ndims}D',
                     color=DIMENSION_COLORS[ndims], alpha=0.8)

    axis.set_xscale('log')
    axis.set_xlabel(r'Problem size: $M^2 + NM$ ($M$: observations, $N$: grid cells)')
    axis.set_ylabel('Speedup (scalar time / vectorized time)')
    axis.set_title(f'Simple kriging speedup (median of {REPEATS} run(s))')
    axis.axhline(1.0, color='gray', linestyle='--', linewidth=1)
    axis.grid(True, which='both', alpha=0.25)
    axis.legend(title='Dimensionality')
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_PATH, dpi=150)
    print(f'Saved {OUTPUT_PATH}', flush=True)
    if SHOW_PLOT:
        plt.show()
    else:
        plt.close(figure)


if __name__ == '__main__':
    main()
