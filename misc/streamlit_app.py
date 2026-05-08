"""
Streamlit app: Pure-Python circulant embedding FFT vs C++ gaussianfft benchmark.

Launch:
    streamlit run misc/streamlit_app.py
"""

from __future__ import annotations

import time
import traceback
from typing import Callable

import numpy as np
import psutil
import streamlit as st
from scipy.fft import irfftn, next_fast_len, rfftn

# ---------------------------------------------------------------------------
# Phase 1 – Pure-Python simulation engine
# ---------------------------------------------------------------------------

# Ratio constants from variogramtypes.hpp
_MIN_RANGE_TO_GRID_RATIO: dict[str, float] = {
    "constant": 1.0,
    "exponential": 2.33,
    "gaussian": 6.67,
    "spherical": 1.92,
    "matern32": 4.0,
    "matern52": 4.76,
    "matern72": 5.26,
    # general_exponential: computed dynamically from power
}


def _ratio_gen_exp(power: float) -> float:
    """Replicates GetMinimumRangeToGridRatio for GenExpVario."""
    return 4.0 + (power - 1.5) * 5.34


def _corr_fn(vario_type: str, power: float = 1.5) -> Callable[[np.ndarray], np.ndarray]:
    """Return a correlation function mapping normalised distance -> [0,1]."""
    if vario_type == "constant":
        return lambda d: np.ones_like(d)
    elif vario_type == "exponential":
        return lambda d: np.exp(-3.0 * d)
    elif vario_type == "gaussian":
        return lambda d: np.exp(-3.0 * d * d)
    elif vario_type == "general_exponential":
        return lambda d: np.exp(-3.0 * np.power(d, power))
    elif vario_type == "spherical":
        def _sph(d: np.ndarray) -> np.ndarray:
            out = np.where(d < 1.0, 1.0 - d * (1.5 - 0.5 * d * d), 0.0)
            return out
        return _sph
    elif vario_type == "matern32":
        def _m32(d: np.ndarray) -> np.ndarray:
            sd = 4.744 * d
            return np.exp(-sd) * (1.0 + sd)
        return _m32
    elif vario_type == "matern52":
        def _m52(d: np.ndarray) -> np.ndarray:
            sd = 5.918 * d
            return np.exp(-sd) * (1.0 + sd + sd * sd / 3.0)
        return _m52
    elif vario_type == "matern72":
        def _m72(d: np.ndarray) -> np.ndarray:
            sd = 6.877 * d
            return np.exp(-sd) * (1.0 + sd + (2.0 / 5.0) * sd**2 + sd**3 / 15.0)
        return _m72
    else:
        raise ValueError(f"Unknown variogram type: {vario_type!r}")


def _padded_size(n: int, cell_size: float, vario_range: float, ratio: float) -> int:
    """Compute FFT-padded size for one axis."""
    min_padding = int(np.ceil(ratio * vario_range / cell_size))
    return next_fast_len(n + min_padding)


def simulate_python(
    nx: int, dx: float,
    ny: int, dy: float,
    nz: int, dz: float,
    vario_type: str,
    vario_range: float,
    power: float = 1.5,
    seed: int | None = None,
) -> np.ndarray:
    """
    3D circulant embedding Gaussian simulation (isotropic).

    Returns a float64 array of shape (nx, ny, nz) with mean≈0, std≈1.
    """
    ratio = (
        _ratio_gen_exp(power)
        if vario_type == "general_exponential"
        else _MIN_RANGE_TO_GRID_RATIO[vario_type]
    )
    corr = _corr_fn(vario_type, power)

    px = _padded_size(nx, dx, vario_range, ratio)
    py = _padded_size(ny, dy, vario_range, ratio)
    pz = _padded_size(nz, dz, vario_range, ratio)

    # Build wrapped covariance grid – isotopic: single range for all axes
    ix = np.minimum(np.arange(px), px - np.arange(px)).astype(float) * dx / vario_range
    jy = np.minimum(np.arange(py), py - np.arange(py)).astype(float) * dy / vario_range
    kz = np.minimum(np.arange(pz), pz - np.arange(pz)).astype(float) * dz / vario_range

    # d[i,j,k] = sqrt(ix[i]^2 + jy[j]^2 + kz[k]^2)  — broadcast via outer sums
    d2 = (
        ix[:, None, None] ** 2
        + jy[None, :, None] ** 2
        + kz[None, None, :] ** 2
    )
    cov = corr(np.sqrt(d2))

    # Spectral factorisation
    cov_fft = rfftn(cov, workers=-1)
    # Clamp small negatives that arise from numerical rounding
    spectral_factor = np.sqrt(np.maximum(cov_fft.real, 0.0)).astype(np.float64)

    rng = np.random.default_rng(seed)
    # Complex white noise in frequency domain – scale by total grid size for parseval
    total = px * py * pz
    noise_real = rng.standard_normal((px, py, pz // 2 + 1))
    noise_imag = rng.standard_normal((px, py, pz // 2 + 1))
    white = (noise_real + 1j * noise_imag) / np.sqrt(2.0 * total)

    field_fft = spectral_factor * white
    field = irfftn(field_fft, s=(px, py, pz), workers=-1)

    return field[:nx, :ny, :nz]


# ---------------------------------------------------------------------------
# Benchmarking helpers
# ---------------------------------------------------------------------------

def _measure(fn: Callable, repeats: int) -> tuple[np.ndarray, float, float]:
    """Run fn() `repeats` times; return (last_result, avg_wall_s, peak_rss_MB)."""
    proc = psutil.Process()
    rss_before = proc.memory_info().rss
    t0 = time.perf_counter()
    result = None
    for _ in range(repeats):
        result = fn()
    elapsed = time.perf_counter() - t0
    rss_after = proc.memory_info().rss
    avg_time = elapsed / repeats
    peak_mb = max(rss_after - rss_before, 0) / (1024 ** 2)
    return result, avg_time, peak_mb


def _estimated_memory_bytes(n3: int) -> int:
    """Rough estimate: ~4 float64 arrays of size n^3."""
    return n3 * 8 * 4


def _padded_memory_bytes(
    nx: int, dx: float,
    ny: int, dy: float,
    nz: int, dz: float,
    vario_type: str, vario_range: float, power: float = 1.5,
) -> int:
    """
    Memory estimate that accounts for the padded FFT grid used by simulate_python.

    Arrays allocated inside simulate_python (dominant terms):
      float64 (px*py*pz): d2, cov, irfftn scratch       → 3 × p3 × 8 bytes
      complex128 (px*py*(pz//2+1)): cov_fft, white,
        field_fft, spectral_factor (as float64 ≈ half)  → ~5 × p3r × 16 bytes
    """
    ratio = (
        _ratio_gen_exp(power)
        if vario_type == "general_exponential"
        else _MIN_RANGE_TO_GRID_RATIO.get(vario_type, 5.0)
    )
    px = _padded_size(nx, dx, vario_range, ratio)
    py = _padded_size(ny, dy, vario_range, ratio)
    pz = _padded_size(nz, dz, vario_range, ratio)
    p3 = px * py * pz
    p3r = px * py * (pz // 2 + 1)
    return int(3 * p3 * 8 + 5 * p3r * 16)


# ---------------------------------------------------------------------------
# Slice visualisation
# ---------------------------------------------------------------------------

def _plot_slices(arr: np.ndarray, title: str):
    """Plot 3 orthogonal centre slices using matplotlib, return figure."""
    import matplotlib.pyplot as plt

    nx, ny, nz = arr.shape
    cx, cy, cz = nx // 2, ny // 2, nz // 2
    vmin, vmax = arr.min(), arr.max()
    cmap = "RdBu_r"

    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    fig.suptitle(title, fontsize=10)

    im0 = axes[0].imshow(arr[:, :, cz].T, origin="lower", aspect="auto",
                         cmap=cmap, vmin=vmin, vmax=vmax)
    axes[0].set_title(f"XY (z={cz})")
    axes[0].set_xlabel("x"); axes[0].set_ylabel("y")

    axes[1].imshow(arr[cx, :, :].T, origin="lower", aspect="auto",
                   cmap=cmap, vmin=vmin, vmax=vmax)
    axes[1].set_title(f"YZ (x={cx})")
    axes[1].set_xlabel("y"); axes[1].set_ylabel("z")

    axes[2].imshow(arr[:, cy, :].T, origin="lower", aspect="auto",
                   cmap=cmap, vmin=vmin, vmax=vmax)
    axes[2].set_title(f"XZ (y={cy})")
    axes[2].set_xlabel("x"); axes[2].set_ylabel("z")

    plt.colorbar(im0, ax=axes, shrink=0.8)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Phase 2 – Streamlit UI
# ---------------------------------------------------------------------------

st.set_page_config(page_title="GaussianFFT Benchmark", layout="wide")
st.title("GaussianFFT: Python vs C++ Benchmark")

# Try importing the C++ library once and cache the result
@st.cache_resource
def _load_gaussianfft():
    try:
        import gaussianfft
        return gaussianfft, None
    except Exception as exc:
        return None, str(exc)


gaussianfft_mod, gaussianfft_err = _load_gaussianfft()

# Sidebar – shared inputs
with st.sidebar:
    st.header("Settings")

    mode = st.radio("Mode", ["Single Comparison", "Scaling Analysis"], index=0)

    st.subheader("Grid")
    nx = st.slider("nx", 8, 256, 32, step=8)
    ny = st.slider("ny", 8, 256, 32, step=8)
    nz = st.slider("nz", 8, 256, 32, step=8)
    dx = st.number_input("dx", value=1.0, min_value=0.01)
    dy = st.number_input("dy", value=1.0, min_value=0.01)
    dz = st.number_input("dz", value=1.0, min_value=0.01)

    st.subheader("Variogram")
    vario_options = [
        "gaussian", "exponential", "general_exponential",
        "spherical", "matern32", "matern52", "matern72", "constant",
    ]
    vario_type = st.selectbox("Type", vario_options, index=0)
    vario_range = st.number_input("Range", value=10.0, min_value=0.1)
    power = 1.5
    if vario_type == "general_exponential":
        power = st.slider("Power", 1.0, 2.0, 1.5, step=0.05)

    st.subheader("Run settings")
    repeats = st.number_input("Repeats", value=1, min_value=1, max_value=20)
    use_seed = st.checkbox("Fix seed", value=True)
    seed_val = st.number_input("Seed", value=42, min_value=0) if use_seed else None

# ---------------------------------------------------------------------------
# Single Comparison mode
# ---------------------------------------------------------------------------

if mode == "Single Comparison":
    if st.button("Run Comparison", type="primary"):
        n3 = nx * ny * nz
        est_bytes = _estimated_memory_bytes(n3)
        if est_bytes > 4 * 1024 ** 3:
            st.warning(
                f"Estimated memory usage ~{est_bytes / 1024**3:.1f} GB may exceed 4 GB. "
                "Consider reducing grid size."
            )

        left, right = st.columns(2)

        # --- Python ---
        with left:
            st.subheader("Pure-Python (scipy FFT)")
            with st.spinner("Running Python simulation…"):
                try:
                    py_result, py_time, py_mem = _measure(
                        lambda: simulate_python(
                            nx, dx, ny, dy, nz, dz,
                            vario_type, vario_range, power, seed_val,
                        ),
                        int(repeats),
                    )
                    st.metric("Avg time", f"{py_time:.3f} s")
                    st.metric("RSS delta", f"{py_mem:.1f} MB")
                    st.metric("Mean", f"{py_result.mean():.4f}")
                    st.metric("Std", f"{py_result.std():.4f}")
                    st.pyplot(_plot_slices(py_result, "Python"))
                except Exception:
                    st.error(f"Python simulation failed:\n{traceback.format_exc()}")

        # --- C++ ---
        with right:
            st.subheader("C++ gaussianfft")
            if gaussianfft_err:
                st.error(f"gaussianfft not available: {gaussianfft_err}")
            else:
                with st.spinner("Running C++ simulation…"):
                    try:
                        gff = gaussianfft_mod
                        if use_seed and seed_val is not None:
                            gff.seed(int(seed_val))
                        v = gff.variogram(vario_type, vario_range, vario_range, vario_range,
                                          power=power)

                        cpp_result, cpp_time, cpp_mem = _measure(
                            lambda: np.array(
                                gff.simulate(v, nx, dx, ny, dy, nz, dz)
                            ).reshape(nx, ny, nz),
                            int(repeats),
                        )
                        st.metric("Avg time", f"{cpp_time:.3f} s")
                        st.metric("RSS delta", f"{cpp_mem:.1f} MB")
                        st.metric("Mean", f"{cpp_result.mean():.4f}")
                        st.metric("Std", f"{cpp_result.std():.4f}")
                        st.pyplot(_plot_slices(cpp_result, "C++"))
                    except Exception:
                        st.error(f"C++ simulation failed:\n{traceback.format_exc()}")

# ---------------------------------------------------------------------------
# Phase 3 – Scaling Analysis mode
# ---------------------------------------------------------------------------

else:
    st.subheader("Scaling Analysis Settings")
    col1, col2 = st.columns(2)
    with col1:
        n0 = st.number_input("Base grid size n₀", value=16, min_value=4, max_value=64)
        base_range = st.number_input("Base range", value=5.0, min_value=0.1)
    with col2:
        doublings = st.number_input("Number of doublings", value=3, min_value=1, max_value=50)
        scale_repeats = st.number_input("Repeats per size", value=1, min_value=1, max_value=5)

    # Preview max dimension
    _cbrt2 = 2 ** (1 / 3)
    _n_max = int(round(int(n0) * (_cbrt2 ** doublings)))
    _r_max = base_range * (_cbrt2 ** doublings)
    _cells_max = _n_max ** 3
    _est_mem = _padded_memory_bytes(
        _n_max, dx, _n_max, dy, _n_max, dz, vario_type, _r_max, power
    )
    _est_mem_str = (
        f"{_est_mem / 1024**3:.2f} GB"
        if _est_mem >= 1024**3
        else f"{_est_mem / 1024**2:.0f} MB"
    )
    st.info(
        f"Max dimension: **n = {_n_max}** (range = {_r_max:.2f}), "
        f"n³ ≈ {_cells_max:,} cells "
        f"(each step multiplies total cells by ×2) — "
        f"estimated memory at max: **{_est_mem_str}**"
    )

    if st.button("Run Scaling Analysis", type="primary"):
        import plotly.graph_objects as go

        cbrt2 = 2 ** (1 / 3)
        sizes = [int(round(int(n0) * (cbrt2 ** i))) for i in range(doublings + 1)]
        ranges = [base_range * (cbrt2 ** i) for i in range(doublings + 1)]

        py_times, cpp_times = [], []
        py_mems, cpp_mems = [], []
        cell_counts = []

        progress = st.progress(0)
        status = st.empty()
        steps = (doublings + 1) * (2 if gaussianfft_mod else 1)
        step = 0

        for idx, (n, r) in enumerate(zip(sizes, ranges)):
            n3 = n ** 3
            est_bytes = _padded_memory_bytes(
                n, dx, n, dy, n, dz, vario_type, r, power
            )
            if est_bytes > 4 * 1024 ** 3:
                st.warning(
                    f"n={n}: estimated padded-FFT memory ~{est_bytes/1024**3:.1f} GB "
                    "exceeds 4 GB — skipping."
                )
                continue

            cell_counts.append(n3)

            # Python
            status.text(f"Python n={n}…")
            try:
                _, pt, pm = _measure(
                    lambda n=n, r=r: simulate_python(n, dx, n, dy, n, dz,
                                                     vario_type, r, power, seed_val),
                    int(scale_repeats),
                )
                py_times.append(pt)
                py_mems.append(pm)
            except Exception as exc:
                py_times.append(float("nan"))
                py_mems.append(float("nan"))
                st.error(
                    f"Python n={n} failed: {type(exc).__name__}: {exc}\n\n"
                    f"```\n{traceback.format_exc()}\n```"
                )
            step += 1
            progress.progress(step / steps)

            # C++
            if gaussianfft_mod:
                status.text(f"C++ n={n}…")
                try:
                    gff = gaussianfft_mod
                    if use_seed and seed_val is not None:
                        gff.seed(int(seed_val))
                    v = gff.variogram(vario_type, r, r, r, power=power)
                    _, ct, cm = _measure(
                        lambda v=v, n=n: np.array(
                            gff.simulate(v, n, dx, n, dy, n, dz)
                        ).reshape(n, n, n),
                        int(scale_repeats),
                    )
                    cpp_times.append(ct)
                    cpp_mems.append(cm)
                except Exception as exc:
                    cpp_times.append(float("nan"))
                    cpp_mems.append(float("nan"))
                    st.error(
                        f"C++ n={n} failed: {type(exc).__name__}: {exc}\n\n"
                        f"```\n{traceback.format_exc()}\n```"
                    )
                step += 1
                progress.progress(step / steps)
            else:
                cpp_times.append(float("nan"))
                cpp_mems.append(float("nan"))

        status.empty()
        progress.empty()

        if len(cell_counts) == 0:
            st.error("No data points collected (all runs skipped or failed).")
        else:
            xs = [str(c) for c in cell_counts]

            # Time chart
            fig_time = go.Figure()
            fig_time.add_trace(go.Scatter(
                x=cell_counts, y=py_times, mode="lines+markers",
                name="Python", line=dict(color="royalblue"),
            ))
            if gaussianfft_mod:
                fig_time.add_trace(go.Scatter(
                    x=cell_counts, y=cpp_times, mode="lines+markers",
                    name="C++", line=dict(color="firebrick"),
                ))
            fig_time.update_layout(
                title="Wall-clock time vs grid cells (log-log)",
                xaxis=dict(title="Total cells (n³)", type="log"),
                yaxis=dict(title="Avg time (s)", type="log"),
            )
            st.plotly_chart(fig_time, use_container_width=True)

            # Memory chart
            fig_mem = go.Figure()
            fig_mem.add_trace(go.Scatter(
                x=cell_counts, y=py_mems, mode="lines+markers",
                name="Python", line=dict(color="royalblue"),
            ))
            if gaussianfft_mod:
                fig_mem.add_trace(go.Scatter(
                    x=cell_counts, y=cpp_mems, mode="lines+markers",
                    name="C++", line=dict(color="firebrick"),
                ))
            fig_mem.update_layout(
                title="RSS memory delta vs grid cells (log-log)",
                xaxis=dict(title="Total cells (n³)", type="log"),
                yaxis=dict(title="RSS delta (MB)", type="log"),
            )
            st.plotly_chart(fig_mem, use_container_width=True)

            # Summary table
            import pandas as pd
            rows = []
            for i, n in enumerate(sizes[:len(cell_counts)]):
                row = {"n": n, "n³": cell_counts[i]}
                if i < len(py_times):
                    row["Python time (s)"] = f"{py_times[i]:.4f}" if not np.isnan(py_times[i]) else "error"
                    row["Python mem (MB)"] = f"{py_mems[i]:.1f}" if not np.isnan(py_mems[i]) else "error"
                if gaussianfft_mod and i < len(cpp_times):
                    row["C++ time (s)"] = f"{cpp_times[i]:.4f}" if not np.isnan(cpp_times[i]) else "error"
                    row["C++ mem (MB)"] = f"{cpp_mems[i]:.1f}" if not np.isnan(cpp_mems[i]) else "error"
                rows.append(row)
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
