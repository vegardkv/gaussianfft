"""Report vectorized kriging timings by grid size, dimension, and observation density."""

from time import perf_counter

import numpy as np

import gaussianfft

GRID_TOTALS = (10_000, 100_000)
DIMENSIONS = (2, 3)
OBS_FRACTIONS = (0.001, 0.01)
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


def benchmark_case(total: int, ndims: int, obs_fraction: float):
    size = round(np.power(total, 1 / ndims))
    assert size > 1
    spacing = GRID_EXTENT / (size - 1)
    n_obs = round(total * obs_fraction)
    assert n_obs > 0
    rng = np.random.default_rng(SEED)
    obs_locations = rng.uniform(0.0, GRID_EXTENT, size=(n_obs, ndims))
    obs_values = rng.standard_normal(n_obs)
    obs_uncertainties = np.full(n_obs, OBS_UNCERTAINTY)
    variogram = gaussianfft.variogram(VARIOGRAM_TYPE, **VARIOGRAM_PARAMETERS)
    grid_args = (size, spacing) * ndims

    elapsed = []
    for _ in range(REPEATS):
        start = perf_counter()
        result = gaussianfft.predict(
            variogram, *grid_args, obs_locations, obs_values, obs_uncertainties,
            vectorized=True,
        )
        elapsed.append(perf_counter() - start)
        del result
    return float(np.median(elapsed)), max(elapsed)


def main():
    # Prepare table print-out
    columns = [
        (ndims, fraction)
        for ndims in DIMENSIONS
        for fraction in OBS_FRACTIONS
    ]
    headers = ['Grid total size'] + [
        f'{ndims}D, observations = {fraction:.1%}' for ndims, fraction in columns
    ]
    widths = [max(len(headers[0]), *(len(f'{total:,}') for total in GRID_TOTALS))]
    widths += [len(header) for header in headers[1:]]
    border = '+' + '+'.join('-' * (width + 2) for width in widths) + '+'

    def print_row(values):
        print('| ' + ' | '.join(
            value.rjust(width) for value, width in zip(values, widths)
        ) + ' |', flush=True)

    # Run benchmarks and print continuously
    print(f'Vectorized simple kriging: median of {REPEATS} prediction timings, in seconds.')
    print(border)
    print_row(headers)
    print(border)
    longest = 0.0
    for total in GRID_TOTALS:
        row = [f'{total:,}']
        for ndims, fraction in columns:
            median, maximum = benchmark_case(total, ndims, fraction)
            row.append(f'{median:.3f} s')
            longest = max(longest, maximum)
        print_row(row)
    print(border)
    print('Longest execution time:')
    print(f'{longest:.3f} s')


if __name__ == '__main__':
    main()
