"""Estimate c_overhead as median seconds per chunk using GridPartition.partitions.

Run: python examples/estimate_grid_partition_overhead.py
Uses fixed, seeded inputs. Excludes setup/tree construction, auto size selection,
and kriging calculations. Results depend on hardware and neighborhood settings.
"""

from time import perf_counter

import numpy as np

from gaussianfft._kriging.grid_partition import GridPartition
from gaussianfft._kriging.grid_spec import GridSpec
from gaussianfft._kriging.neighborhood import Neighborhood
from gaussianfft._kriging.observation_data import ObservationData


def main():
    grid = GridSpec((100, 100), (10.0, 10.0))
    neighborhood = Neighborhood(chunk_shape=(1, 1), padding=(200.0, 100.0))
    n_obs = 200
    repeats = 5
    rng = np.random.default_rng(42)
    indices = np.column_stack(np.unravel_index(
        rng.choice(np.prod(grid.shape), n_obs, replace=False), grid.shape,
    ))
    obs = ObservationData(indices * grid.spacing, np.zeros(n_obs), np.zeros(n_obs))
    partition = GridPartition(neighborhood, obs, variogram=None)

    # Consume the lazy iterator so warm-up and timings include actual construction.
    sum(1 for _ in partition.partitions(grid))
    print(f'Grid: {grid.shape}, spacing: {grid.spacing}, observations: {n_obs}')
    print(f'Chunk shape: {neighborhood.chunk_shape}, padding: {neighborhood.padding}')
    seconds_per_chunk = []
    for repeat in range(repeats):
        start = perf_counter()
        n_chunks = sum(1 for _ in partition.partitions(grid))
        elapsed = perf_counter() - start
        seconds_per_chunk.append(elapsed / n_chunks)
        print(
            f'Run {repeat + 1}: {elapsed:.6f} s / {n_chunks} chunks '
            f'= {elapsed / n_chunks:.6e} s/chunk'
        )

    print(f'\nc_overhead = {np.median(seconds_per_chunk):.6e}  # seconds per chunk (median)')


if __name__ == '__main__':
    main()
