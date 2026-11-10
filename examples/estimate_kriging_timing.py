# The purpose of this script is to fit a function that estimates the time required
# to perform simple kriging, and report peak process memory for each run.
# Simple kriging run time is approximated by three components:
# - Calculating the covariance matrix between observations
# - Calculating the cross-covariance between observations and grid nodes
# - Solving the kriging system (Cholesky factorization)
# A polynomial function can then represent the run time:
#   T_total = T_cov_obs + T_cross_cov + T_solve
# where
#   T_cov_obs   = C_cov_obs   * N_obs^2
#   T_cross_cov = C_cross_cov * N_obs   * N_grid
#   T_solve     = C_solve     * N_obs^3

import sys
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from time import perf_counter

import numpy as np
import psutil

import gaussianfft


def measure_prediction(
    grid_shape: tuple[int, ...], grid_spacing: tuple[float, ...],
    obs_locations: np.ndarray, obs_values: np.ndarray, obs_uncertainties: np.ndarray,
) -> tuple[float, float]:
    """Return peak process RSS in MiB and prediction time in seconds."""
    variogram = gaussianfft.variogram(
        "matern52",
        main_range=300.0,
        perp_range=100.0,
        depth_range=50.0,
    )
    # Warm up lazy imports in each fresh worker without allocating a full-size case.
    gaussianfft.predict(
        variogram, (2,) * len(grid_shape), grid_spacing,
        obs_locations[:1], obs_values[:1], obs_uncertainties[:1],
    )
    start = perf_counter()
    result = gaussianfft.predict(
        variogram, grid_shape, grid_spacing, obs_locations, obs_values, obs_uncertainties,
    )
    elapsed = perf_counter() - start
    del result

    # OS high-water marks capture temporary allocations without polling.
    # This includes the worker's imports, inputs, and small warm-up.
    if sys.platform == "win32":
        peak_bytes = psutil.Process().memory_info().peak_wset
    else:
        import resource

        peak_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform != "darwin":
            peak_bytes *= 1024
    return peak_bytes / 1024**2, elapsed


def main():
    # observation_counts = (2000,)
    observation_counts = (50, 100, 200, 500, 1000, 2000)
    grid_shapes = (
        (20, 20, 10),
        (40, 40, 20),
        (80, 80, 40),
        # (160, 160, 80),
        # (320, 320, 160),
    )
    repeats = 3
    extent = 900.0
    rng = np.random.default_rng(42)

    # ----------------------------------------------------------------------------
    # Timing
    # ----------------------------------------------------------------------------

    # (N_obs, Nx, Ny, Nz, peak process RSS in MiB, run_time in seconds)
    timings: list[tuple[int, int, int, int, float, float]] = []

    for n_obs in observation_counts:
        obs_locations = rng.uniform(0.0, extent, size=(n_obs, 3))
        obs_values = rng.standard_normal(n_obs)
        obs_uncertainties = np.full(n_obs, 0.1)

        for nx, ny, nz in grid_shapes:
            grid_shape = (nx, ny, nz)
            grid_spacing = tuple(extent / (size - 1) for size in grid_shape)

            for _ in range(repeats):
                # A fresh worker keeps previous runs out of the memory high-water mark.
                with ProcessPoolExecutor(max_workers=1, mp_context=get_context("spawn")) as executor:
                    peak_memory, elapsed = executor.submit(
                        measure_prediction, grid_shape, grid_spacing,
                        obs_locations, obs_values, obs_uncertainties,
                    ).result()
                timings.append((n_obs, nx, ny, nz, peak_memory, elapsed))
                print(
                    f"{n_obs=}, {nx=}, {ny=}, {nz=}: {elapsed:.6f} s, "
                    f"peak RSS: {peak_memory:.1f} MiB",
                    flush=True,
                )

    grid_totals = [nx * ny * nz for nx, ny, nz in grid_shapes]
    median_times = np.empty((len(observation_counts), len(grid_shapes)))
    median_memory = np.empty_like(median_times)
    for obs_index, n_obs in enumerate(observation_counts):
        for grid_index, grid_shape in enumerate(grid_shapes):
            median_memory[obs_index, grid_index], median_times[obs_index, grid_index] = np.median([
                (peak_memory, elapsed) for count, nx, ny, nz, peak_memory, elapsed in timings
                if count == n_obs and (nx, ny, nz) == grid_shape
            ], axis=0)

    # ----------------------------------------------------------------------------
    # Estimation
    # ----------------------------------------------------------------------------

    obs_counts = np.repeat(np.asarray(observation_counts, dtype=float), len(grid_shapes))
    node_counts = np.tile(np.asarray(grid_totals, dtype=float), len(observation_counts))
    measured_times = median_times.ravel()
    design_matrix = np.column_stack((
        obs_counts**2, obs_counts**3, obs_counts * node_counts,
    ))
    column_scales = np.linalg.norm(design_matrix, axis=0)
    scaled_coefficients, _, rank, _ = np.linalg.lstsq(
        design_matrix / column_scales, measured_times, rcond=None,
    )
    if rank < 3:
        raise ValueError("Vary observation counts and grid sizes to identify all three coefficients.")

    c_cov_obs, c_solve, c_cross_cov = scaled_coefficients / column_scales
    estimated_times = design_matrix @ (scaled_coefficients / column_scales)
    fitted_times = estimated_times.reshape(median_times.shape)
    residuals = measured_times - estimated_times
    rmse = np.sqrt(np.mean(residuals**2))
    total_variation = np.sum((measured_times - measured_times.mean())**2)
    r_squared = 1 - np.sum(residuals**2) / total_variation if total_variation > 0 else np.nan

    print("\nLeast-squares fit to median runtimes (seconds):")
    print(f"C_cov_obs = {c_cov_obs:.6e}")
    print(f"C_solve = {c_solve:.6e}")
    print(f"C_cross_cov = {c_cross_cov:.6e}")
    print("T = C_cov_obs * N_obs^2 + C_solve * N_obs^3 + C_cross_cov * N_obs * N_grid")
    print(f"RMSE: {rmse:.6f} s; R^2: {r_squared:.4f}")

    largest_obs = max(observation_counts)
    largest_grid = max(grid_totals)
    print(f"\nFitted contributions at N_obs = {largest_obs}, N_grid = {largest_grid}:")
    print(
        f"C_cov_obs * N_obs^2 = C_cov_obs * {largest_obs**2} "
        f"= {c_cov_obs * largest_obs**2:.6f} s"
    )
    print(
        f"C_solve * N_obs^3 = C_solve * {largest_obs**3} "
        f"= {c_solve * largest_obs**3:.6f} s"
    )
    print(
        f"C_cross_cov * N_obs * N_grid = C_cross_cov * {largest_obs * largest_grid} "
        f"= {c_cross_cov * largest_obs * largest_grid:.6f} s"
    )

    # ----------------------------------------------------------------------------
    # Visualization
    # ----------------------------------------------------------------------------

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), layout="constrained")
    axes = axes.ravel()
    for grid_index, total in enumerate(grid_totals):
        line, = axes[0].plot(
            observation_counts, median_times[:, grid_index], marker="o", label=f"{total:,}",
        )
        axes[0].plot(
            observation_counts, fitted_times[:, grid_index], linestyle="--", color=line.get_color(),
        )
    axes[0].set_title("Scaling with observation count (dashed: fit)")
    axes[0].set_xlabel("Number of observations")
    axes[0].set_ylabel("Median runtime (seconds)")
    axes[0].legend(title="Grid nodes")

    for obs_index, n_obs in enumerate(observation_counts):
        line, = axes[1].plot(
            grid_totals, median_times[obs_index], marker="o", label=f"{n_obs:,}",
        )
        axes[1].plot(
            grid_totals, fitted_times[obs_index], linestyle="--", color=line.get_color(),
        )
    axes[1].set_title("Scaling with grid size (dashed: fit)")
    axes[1].set_xlabel("Number of grid nodes (nx * ny * nz)")
    axes[1].set_ylabel("Median runtime (seconds)")
    axes[1].legend(title="Observations")

    axes[2].scatter(measured_times, estimated_times)
    limits = (min(0.0, estimated_times.min()), max(measured_times.max(), estimated_times.max()))
    axes[2].plot(limits, limits, color="black", linestyle="--")
    axes[2].set_xlim(limits)
    axes[2].set_ylim(limits)
    axes[2].set_aspect("equal", adjustable="box")
    axes[2].set_title(f"Measured vs. predicted (R^2 = {r_squared:.4f})")
    axes[2].set_xlabel("Measured runtime (seconds)")
    axes[2].set_ylabel("Predicted runtime (seconds)")

    axes[3].scatter(estimated_times, residuals)
    axes[3].axhline(0, color="black", linestyle="--")
    axes[3].set_title(f"Residuals (RMSE = {rmse:.4f} s)")
    axes[3].set_xlabel("Predicted runtime (seconds)")
    axes[3].set_ylabel("Measured - predicted (seconds)")

    for axis in axes[:2]:
        axis.set_xlim(left=0)
        axis.set_ylim(bottom=0)
    for axis in axes:
        axis.grid(alpha=0.3)

    plt.show(block=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    for grid_index, total in enumerate(grid_totals):
        axes[0].plot(
            observation_counts, median_memory[:, grid_index], marker="o", label=f"{total:,}",
        )
    axes[0].set_title("Memory scaling with observation count")
    axes[0].set_xlabel("Number of observations")
    axes[0].legend(title="Grid nodes")

    for obs_index, n_obs in enumerate(observation_counts):
        axes[1].plot(
            grid_totals, median_memory[obs_index], marker="o", label=f"{n_obs:,}",
        )
    axes[1].set_title("Memory scaling with grid size")
    axes[1].set_xlabel("Number of grid nodes (nx * ny * nz)")
    axes[1].legend(title="Observations")

    for axis in axes:
        axis.set_ylabel("Median peak RSS (MiB)")
        axis.set_xlim(left=0)
        axis.set_ylim(bottom=0)
        axis.grid(alpha=0.3)

    plt.show(block=True)


if __name__ == "__main__":
    main()
