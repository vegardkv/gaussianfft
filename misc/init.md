the main function of this repo is "gaussianfft.simulate", exposed via pybind11 in #file:gaussfftinterface.cpp.

i want to assess the feasibility of creating a pure python implementation of the same interface (with numpy/scipy as needed).

to do this, i want you to create a streamlit app that also implements this python implementation

The streamlit app should allow me to define the following input:
- 3D grid dimensions (lower dimensions not needed)
- variogram settings (single range, type)
- number of repeats (for consistent timing)

the output should be (for both gaussianfft.simulate and the python equivalent):
- a small preview window of the three center slices of the cube (XY-, YZ, and ZX-planes)
- total execution time
- total memory usage

use third party packages if required and if it significantly improves readability.

i also want a "scaling" mode that allows me to give similar inputs, but doubles the grid size and variogram range X times (user input), and finally plots the memory usage and execution time as a function of problem size (loglog scale, number of grid cells on X-axis, time/memory on Y-axis).

see README-md for additional info if required.