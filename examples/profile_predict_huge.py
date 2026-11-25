"""
Run one 3D Huge prediction with neighborhoods ON in the foreground. This script
is simpler to profile than the benchmark script.
"""

from examples.benchmark_predict_neighborhood import CASES, measure_predict


def main() -> None:
    case = next(case for case in CASES if case.name == '3D Huge')
    print(f'{case.name}: {case.shape}, {case.n_obs} observations, neighborhood ON', flush=True)
    measurement, _ = measure_predict(case, use_neighborhood=True, keep_prediction=False)
    print(f'Predict: {measurement.wall_seconds:.2f} s wall, '
          f'{measurement.peak_rss_mib:.0f} MiB peak RSS')


if __name__ == '__main__':
    main()
