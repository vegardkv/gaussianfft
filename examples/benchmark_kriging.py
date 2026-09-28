from time import perf_counter

import numpy as np

import gaussianfft


def main():
    nx = ny = 1000
    dx = dy = 1.0
    rng = np.random.default_rng(42)
    obs_locations = rng.uniform(0.0, 999.0, size=(100, 2))
    obs_values = rng.standard_normal(100)
    obs_uncertainties = np.full(100, 0.1)
    variogram = gaussianfft.variogram(
        'matern52', main_range=300.0, perp_range=100.0, azimuth=30.0
    )

    predictions = []
    timings = []
    for vectorized in (False, True):
        start = perf_counter()
        prediction, _ = gaussianfft.predict(
            variogram, nx, dx, ny, dy,
            obs_locations, obs_values, obs_uncertainties,
            vectorized=vectorized,
        )
        elapsed = perf_counter() - start
        predictions.append(prediction)
        timings.append(elapsed)
        label = 'Vectorized' if vectorized else 'Scalar'
        print(f'{label} kriging (1000x1000, 100 observations): {elapsed:.3f} s', flush=True)

    np.testing.assert_allclose(predictions[0], predictions[1], rtol=1e-12, atol=1e-12)
    print(f'Speedup: {timings[0] / timings[1]:.2f}x (predictions match)')


if __name__ == '__main__':
    main()
