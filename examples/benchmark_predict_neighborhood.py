"""Compare global and automatic-neighborhood predict

Run: python examples/benchmark_predict_neighborhood.py

Deviations compare the first ON/OFF run over the full grid: mean absolute
error (MAE), root mean square error (RMSE), and maximum absolute error.
Relative RMSE is normalized by the RMS of the OFF field (not pointwise).
These statistics are printed and included in a table below the charts.
Timing and peak RSS exclude result transfer and deviation calculations.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path
import statistics
import sys
from time import perf_counter, process_time
from typing import TYPE_CHECKING, NamedTuple, Optional

if TYPE_CHECKING:
    import numpy as np


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    shape: tuple[int, ...]
    spacing: tuple[float, ...]
    n_obs: int
    model: str
    variogram_options: dict[str, float]
    calculate_stddev: bool = False


# Status (might need verification):
# - Some cases show significant deviation in RMSE:
#   - 2D rotated: The current _find_padding approach assumes axis-aligned grids.
#     See work on padding-fix, which might be relevant here
#   - 2D with stddev: Padding is based on a correlation threshold, but the Gaussian
#     variogram tapers very slowly, so the threshold is perhaps too high
# - Performance:
#   - 2D with stddev is 5x slower with neighborhood enabled. This is due to the
#     objective function not accounting for the additional Cholesky solves needed
#     Note that wall time is not that affected, probably due to some BLAS multi-
#     threading.


CASES = (
    #             Name              Shape           Spacing           N_obs  Model          Variogram options
    BenchmarkCase(      '2D small', ( 32,  24,   ), (10.0, 10.0,     ),  30, 'exponential', {'main_range': 80.0}),
    BenchmarkCase('2D short range', (120, 100,   ), (10.0, 10.0,     ), 500,   'spherical', {'main_range': 80.0}),
    BenchmarkCase( '2D long range', (120, 100,   ), (10.0, 10.0,     ), 500,   'spherical', {'main_range': 800.0}),
    BenchmarkCase(    '2D rotated', (160,  96,   ), (10.0, 15.0,     ), 600,    'matern52', {'main_range': 240.0, 'perp_range': 60.0, 'azimuth': 35.0}),
    BenchmarkCase('2D with stddev', (100,  80,   ), (10.0, 10.0,     ), 250,    'gaussian', {'main_range': 140.0}, calculate_stddev=True),
    BenchmarkCase(     '3D sparse', ( 40,  36, 24), (10.0, 10.0,  5.0),  40, 'exponential', {'main_range': 100.0}),
    BenchmarkCase('3D short range', ( 48,  40, 32), (10.0, 10.0, 10.0), 450,    'matern32', {'main_range': 45.0}),
    BenchmarkCase( '3D long range', ( 48,  40, 32), (10.0, 10.0, 10.0), 450,    'matern32', {'main_range': 350.0}),
    BenchmarkCase('3D anisotropic', ( 64,  48, 20), (10.0, 10.0,  5.0), 350,    'matern72', {'main_range': 135.0, 'perp_range': 70.0, 'depth_range': 25.0, 'azimuth': 25.0}),
    BenchmarkCase('3D with stddev', ( 32,  24, 20), (10.0, 10.0,  5.0), 180,   'spherical', {'main_range': 75.0}, calculate_stddev=True),
    BenchmarkCase(      '3D large', (112,  96, 30), (10.0, 10.0, 10.0), 450,    'matern32', {'main_range': 45.0}),
    BenchmarkCase( '3D large 2000', (112,  96, 30), (10.0, 10.0, 10.0),2000,    'matern32', {'main_range': 45.0}),
)

SEED = 42
PARITY_MARGIN = 0.05
METRICS = ('CPU time (s)', 'Wall time (s)', 'Peak RSS (MiB)')


class Measurement(NamedTuple):
    cpu_seconds: float
    wall_seconds: float
    peak_rss_mib: float


DeviationRow = tuple[str, str, str, str, str]


class BenchmarkResult(NamedTuple):
    case: BenchmarkCase
    off: Measurement
    on: Measurement
    deviations: list[DeviationRow]


def measure_predict(
    case: BenchmarkCase, use_neighborhood: bool,
) -> tuple[Measurement, tuple[np.ndarray, Optional[np.ndarray]]]:
    import numpy as np

    import gaussianfft as grf

    rng = np.random.default_rng(SEED)
    extent = (np.array(case.shape) - 1) * case.spacing
    locations = rng.uniform(0.0, extent, size=(case.n_obs, len(case.shape)))
    values = rng.standard_normal(case.n_obs)
    uncertainties = np.full(case.n_obs, 0.1)
    variogram = grf.variogram(case.model, **case.variogram_options)

    wall_start = perf_counter()
    cpu_start = process_time()
    result = grf.predict(
        variogram, case.shape, case.spacing, locations, values, uncertainties,
        use_neighborhood=use_neighborhood, calculate_stddev=case.calculate_stddev,
    )
    cpu_seconds = process_time() - cpu_start
    wall_seconds = perf_counter() - wall_start

    if sys.platform == 'win32':
        import psutil

        peak_bytes = psutil.Process().memory_info().peak_wset
    else:
        import resource

        peak_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # macOS reports bytes; Linux reports KiB.
        if sys.platform != 'darwin':
            peak_bytes *= 1024
    return Measurement(cpu_seconds, wall_seconds, peak_bytes / 1024**2), result


def report_deviations(
    off: tuple[np.ndarray, Optional[np.ndarray]],
    on: tuple[np.ndarray, Optional[np.ndarray]],
) -> list[DeviationRow]:
    import numpy as np

    rows = []
    for name, reference, actual in zip(('Prediction', 'Stddev'), off, on):
        if reference is None and actual is None:
            continue
        if reference is None or actual is None or reference.shape != actual.shape:
            raise ValueError(f'{name}: ON and OFF must have matching output shapes')
        if not (np.isfinite(reference).all() and np.isfinite(actual).all()):
            raise ValueError(f'{name}: cannot compare non-finite output values')
        difference = actual - reference
        mae = np.mean(np.abs(difference))
        rmse = np.sqrt(np.mean(difference**2))
        max_error = np.max(np.abs(difference))
        reference_rms = np.sqrt(np.mean(reference**2))
        relative = f'{100 * rmse / reference_rms:.6g}%' if reference_rms > 0 else 'n/a (OFF RMS is zero)'
        row = (name, f'{mae:.6g}', f'{rmse:.6g}', f'{max_error:.6g}', relative)
        rows.append(row)
        print('  {} deviation (ON - OFF): MAE={}, RMSE={}, '
              'max abs={}, relative RMSE={}'.format(*row))
    return rows


def benchmark(case: BenchmarkCase, repeats: int) -> BenchmarkResult:
    samples: dict[bool, list[Measurement]] = {False: [], True: []}
    predictions = {}
    for repeat in range(repeats):
        order = (False, True) if repeat % 2 == 0 else (True, False)
        for use_neighborhood in order:
            with ProcessPoolExecutor(max_workers=1, mp_context=get_context('spawn')) as executor:
                measurement, prediction = executor.submit(measure_predict, case, use_neighborhood).result()
            samples[use_neighborhood].append(measurement)
            if repeat == 0:
                predictions[use_neighborhood] = prediction
    off, on = (
        Measurement(*(statistics.median(values) for values in zip(*samples[mode])))
        for mode in (False, True)
    )
    deviations = report_deviations(predictions[False], predictions[True])
    return BenchmarkResult(case, off, on, deviations)


def summarize(results: list[BenchmarkResult], output: Path) -> None:
    import matplotlib.pyplot as plt

    print('\nMedians (off -> on, with on/off ratio):')
    print(f'{"Case":<20}' + ''.join(f'{metric:>27}' for metric in METRICS))
    for case, off, on, _ in results:
        cells = [f'{a:.4g} -> {b:.4g} ({b / a:.2f}x)' for a, b in zip(off, on)]
        print(f'{case.name:<20}' + ''.join(f'{cell:>27}' for cell in cells))

    deviation_rows = [
        (result.case.name, *row) for result in results for row in result.deviations
    ]
    chart_height = max(5.5, 3.2 + 0.65 * len(results))
    table_height = 0.8 + 0.28 * (len(deviation_rows) + 1)
    fig = plt.figure(figsize=(13, chart_height + table_height), layout='constrained')
    grid = fig.add_gridspec(2, 1, height_ratios=[chart_height, table_height])
    axes = grid[0].subgridspec(2, len(METRICS)).subplots(sharey=True)
    below, above = 1 - PARITY_MARGIN, 1 + PARITY_MARGIN
    positions = range(len(results))
    for index, metric in enumerate(METRICS):
        ax_values, ax_ratios = axes[:, index]
        off_values = [result.off[index] for result in results]
        on_values = [result.on[index] for result in results]
        ax_values.barh([y - 0.2 for y in positions], off_values, height=0.36,
                       color='tab:gray', label='Off (global)')
        ax_values.barh([y + 0.2 for y in positions], on_values, height=0.36,
                       color='tab:blue', label='On (automatic)')
        ax_values.set_title(metric)
        ax_values.set_xlabel('Median')
        ax_values.grid(axis='x', alpha=0.25)
        ax_values.set_axisbelow(True)

        ratios = [on / off for off, on in zip(off_values, on_values)]
        lower = sum(ratio < below for ratio in ratios)
        higher = sum(ratio > above for ratio in ratios)
        near = len(ratios) - lower - higher
        print(f'{metric}: {lower} lower, {near} within 5%, {higher} higher')
        ax_ratios.axvspan(below, above, color='0.9')
        ax_ratios.axvline(1, color='0.5', linewidth=1)
        colors = ['tab:blue' if r < below else 'tab:orange' if r > above else '0.4' for r in ratios]
        ax_ratios.scatter(ratios, positions, color=colors, zorder=3)
        ax_ratios.set_xscale('log')
        ax_ratios.set_title(f'{lower} lower / {near} within 5% / {higher} higher')
        ax_ratios.set_xlabel('On / off (lower is better)')
    axes[0, 0].legend(loc='lower right')
    axes[0, 0].set_yticks(positions, [result.case.name for result in results])
    axes[0, 0].invert_yaxis()
    ax_table = fig.add_subplot(grid[1])
    ax_table.set_axis_off()
    ax_table.set_title('First-run output deviations (ON - OFF); relative RMSE = RMSE / RMS(OFF)')
    table = ax_table.table(
        cellText=deviation_rows,
        colLabels=['Case', 'Output', 'MAE', 'RMSE', 'Max abs', 'Relative RMSE'],
        colWidths=[0.19, 0.13, 0.13, 0.13, 0.13, 0.29],
        cellLoc='center', bbox=[0, 0, 1, 0.95],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    fig.suptitle('Automatic neighborhoods vs. global predict\n'
                 'Top: measured medians  |  Middle: ratios (shaded: within 5%)  |  Bottom: deviations')
    fig.savefig(output, dpi=150)
    print(f'Plot saved to {output}')
    plt.show(block=True)
    plt.close(fig)


def main() -> None:
    cases_by_name = {case.name: case for case in CASES}
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--case', action='append', choices=cases_by_name, help='run only this case (repeatable)')
    parser.add_argument('--repeats', type=int, default=3, help='runs per case and mode (default: 3)')
    parser.add_argument('--output', type=Path, default=Path('predict_neighborhood_benchmark.png'),
                        help='destination for the plot (default: predict_neighborhood_benchmark.png)')
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('--repeats must be positive')

    cases = [cases_by_name[name] for name in args.case] if args.case else CASES
    print(f'{args.repeats} fresh-process run(s) per mode.')
    print('Timing covers predict only. Peak RSS includes worker imports, inputs, and outputs.')
    print('Deviations use the first run per mode; relative RMSE = RMSE / RMS(OFF).')
    results = []
    for case in cases:
        print(f'Running {case.name}: {case.shape}, {case.n_obs} observations, '
              f'{case.model} {case.variogram_options}', flush=True)
        results.append(benchmark(case, args.repeats))
    summarize(results, args.output)


if __name__ == '__main__':
    main()
