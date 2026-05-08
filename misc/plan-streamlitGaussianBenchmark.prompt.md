# Plan: Streamlit App for Python vs C++ Gaussian Simulation

Create `misc/streamlit_app.py` — a pure-Python circulant embedding FFT simulation matching the `gaussianfft.simulate` interface, benchmarked side-by-side with the C++ library. Two modes: single comparison and scaling analysis.

**Decisions made:** Isotropic only, psutil for memory, 3D only, file at `misc/streamlit_app.py`.

---

## Phase 1: Pure-Python Simulation Engine

1. **Implement variogram correlation functions** — all 8 types as standalone functions taking normalized distance → correlation [0,1]:
   - `constant`: 1.0, `exponential`: $e^{-3d}$, `gaussian`: $e^{-3d^2}$, `general_exponential`: $e^{-3d^p}$, `spherical`: $1-1.5t+0.5t^3$, `matern32/52/72` with their exact constants (4.744, 5.918, 6.877)

2. **Implement padding calculation** — replicate C++ logic using per-variogram `min_range_to_grid_ratio` constants + `scipy.fft.next_fast_len()` for FFT-friendly sizes

3. **Implement 3D circulant embedding simulation**:
   - Build wrapped 3D covariance grid on padded domain
   - `rfftn(cov)` → element-wise sqrt (clamp negatives) → generate white noise → `rfftn(noise)` → multiply → `irfftn` → extract `[0:nx, 0:ny, 0:nz]`
   - Use `scipy.fft.rfftn` / `irfftn` with `workers=-1` for fairer multi-threaded comparison vs MKL-backed C++

## Phase 2: Streamlit UI — Single Comparison Mode

4. **Sidebar inputs**: `nx, ny, nz` (sliders 8–256), `dx, dy, dz` (floats), variogram type dropdown, range, power (conditional), repeats, seed

5. **"Run Comparison" button**: runs both implementations, measures `time.perf_counter` wall-clock and `psutil.Process().memory_info().rss` delta

6. **Output display**: side-by-side columns — 3 center slices each (XY, YZ, XZ via matplotlib), total time, peak memory (MB)

## Phase 3: Scaling Mode

7. **Mode toggle**: radio button — "Single Comparison" vs "Scaling Analysis"

8. **Scaling inputs**: base grid size `n0`, cell size `d`, variogram type + base range, number of doublings (1–8), repeats per size

9. **Scaling execution**: for each doubling, `n = n0 × 2^i`, `range = base_range × 2^i`, run both, record avg time + memory, with progress bar

10. **Scaling plots**: two log-log charts (matplotlib/plotly), X = total cells ($n^3$), Y = time / memory, two series per chart (C++ vs Python)

---

## Relevant Files
- `misc/streamlit_app.py` — **create** (all code here)
- `src/gaussfft.cpp` — reference for simulation algorithm, padding
- `src/nrlib/variogram/variogramtypes.hpp` — exact correlation formulas & ratio constants
- `gaussianfft/__init__.py` — import pattern for C++ library

## Verification
1. **Visual**: both implementations produce qualitatively similar center slices (not bitwise identical — different RNG engines)
2. **Statistical**: mean ≈ 0, std ≈ 1 for both
3. **Correlation**: verify Python variogram functions match `gaussianfft.Variogram.corr()` at sample distances
4. **Scaling**: both show similar exponents in log-log (time ~ $O(N \log N)$, memory ~ $O(N)$)
5. **Launch**: `streamlit run misc/streamlit_app.py`

## Further Considerations
1. **scipy.fft vs numpy.fft**: `scipy.fft` supports `workers=-1` for multi-threaded FFT, making a fairer comparison against MKL-backed C++. Recommend using scipy.fft.
2. **Large grids**: scaling mode can exceed RAM. Add `st.warning` when estimated memory ($n^3 \times 8 \times 4$ bytes for temporaries) exceeds ~4 GB.
3. **RNG**: C++ uses DSFMT, Python uses `numpy.random.Generator`. Results differ but are statistically equivalent — expected for feasibility assessment.
