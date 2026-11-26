"""Compare auto chunking with fixed shapes on the 3D Huge prediction benchmark.

Run from the repository root: python -m examples.benchmark_auto_chunk_shapes
Use --repeats 1 for a single timing per shape.
"""

import argparse
from pathlib import Path
from random import Random
from statistics import median
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

import gaussianfft as grf
from gaussianfft._kriging import api
from gaussianfft._kriging.grid_partition import _find_optimal_chunk_shape, _find_padding
from gaussianfft._kriging.grid_spec import GridSpec


GRID_SHAPE = (292, 205, 58)
SPACING = (10.0, 10.0, 10.0)
N_OBS = 3600
SEED = 42
CHUNK_SHAPES = (
    (8, 8, 8), (16, 16, 16), (16, 16, 8),
    # Vary footprint at fixed depth around the observed auto choice (36, 26, 1).
    (24, 18, 1), (36, 26, 1), (48, 34, 1), (72, 52, 1),
    # Isolate depth while keeping the observed auto footprint.
    (36, 26, 2), (36, 26, 4), (36, 26, 8),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repeats', type=int, default=3, help='timings per shape (default: 3)')
    parser.add_argument('--output', type=Path, default=Path('auto_chunk_shapes.png'))
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('--repeats must be positive')

    obs_rng = np.random.default_rng(SEED)
    extent = (np.array(GRID_SHAPE) - 1) * SPACING
    locations = obs_rng.uniform(0.0, extent, size=(N_OBS, len(GRID_SHAPE)))
    values = obs_rng.standard_normal(N_OBS)
    uncertainties = np.full(N_OBS, 0.1)
    variogram = grf.variogram('exponential', main_range=50.0)
    grid = GridSpec(GRID_SHAPE, SPACING)
    padding = _find_padding(api._as_covariance(variogram), tuple(np.multiply(grid.spacing, grid.shape)))
    selected = _find_optimal_chunk_shape(padding, grid, cKDTree(locations))
    print(f'3D Huge: auto selects {selected}; shared padding: {np.round(padding, 2)}')

    # Include the selected shape as a fixed run to separate choice from selection overhead.
    shapes = list(dict.fromkeys((*CHUNK_SHAPES, selected)))
    modes = [('auto', 'auto')] + [(str(shape), shape) for shape in shapes]
    samples = {label: [] for label, _ in modes}
    rng = Random(SEED)
    for repeat in range(args.repeats):
        rng.shuffle(modes)
        for label, shape in modes:
            start = perf_counter()
            result = api.predict(
                variogram, GRID_SHAPE, SPACING, locations, values, uncertainties,
                chunk_shape=shape, padding_range=None if shape == 'auto' else padding,
                calculate_stddev=False,
            )
            elapsed = perf_counter() - start
            del result
            samples[label].append(elapsed)
            print(f'Run {repeat + 1}: {label}: {elapsed:.3f} s', flush=True)

    timings = {label: median(times) for label, times in samples.items()}
    best = min(map(str, shapes), key=timings.__getitem__)
    ratio = timings['auto'] / timings[best]
    summary = f'Fastest tested fixed shape: {best}; auto / fastest fixed: {ratio:.2f}x'
    print(summary)

    labels = ['auto', *map(str, shapes)]
    fig, ax = plt.subplots(figsize=(9, 5), layout='constrained')
    bars = ax.barh([f'auto {selected}', *labels[1:]], [timings[label] for label in labels],
                   color=['tab:orange' if label == 'auto' else 'tab:blue' for label in labels])
    ax.bar_label(bars, fmt='%.2f', padding=3)
    ax.margins(x=0.1)
    ax.invert_yaxis()
    ax.set_xlabel(f'Median prediction wall time (s), {args.repeats} run(s) per mode')
    ax.set_ylabel('Chunk shape (x, y, z)')
    ax.set_title(f'3D Huge: automatic vs. fixed chunk shapes\n{summary}')
    fig.savefig(args.output, dpi=150)
    plt.close(fig)
    print(f'Plot saved to {args.output}')


if __name__ == '__main__':
    main()
