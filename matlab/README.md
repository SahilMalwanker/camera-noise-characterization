# MATLAB acquisition scripts

The hardware sessions were run in MATLAB with a Baumer TXG08 GigE camera. These scripts are
the cleaned-up, parameterised versions of that lab code: each one is a function, releases the
camera on exit (`onCleanup`), and the sweeps write CSV files with exactly the columns the Python
package reads. MATLAB acquires, `camera-noise-characterization` analyses.

| Script | Lab task | What it does | Python counterpart |
| --- | --- | --- | --- |
| [`cnc_acquire_snapshot.m`](cnc_acquire_snapshot.m) | 1.1 / 1.2 | live preview, one frame, histogram, clipping report | `camera_noise_characterization.io` |
| [`cnc_noise_pair.m`](cnc_noise_pair.m) | 1.4 a | two frames, `std(A - B)`, per-frame noise `std(A - B)/sqrt(2)` | `noise.pair_stats` |
| [`cnc_exposure_sweep.m`](cnc_exposure_sweep.m) | 1.4 b | noise and mean over an exposure range, CSV export | `protocols.exposure_sweep` |
| [`cnc_gain_sweep.m`](cnc_gain_sweep.m) | 1.4 c | automatic re-exposure to 128 DN for every gain, CSV export | `protocols.gain_sweep` |
| [`cnc_denoise_filters.m`](cnc_denoise_filters.m) | 1.4 d | binomial 3/5/7 and median 3x3 on a flat-field pair vs theory | `filters.denoise_benchmark` |
| [`cnc_edge_filters.m`](cnc_edge_filters.m) | 1.5 | Sobel, Prewitt, Laplacian, LoG and Canny side by side | `camera_noise_characterization.edges` |

## From MATLAB to a fit in two commands

```matlab
T = cnc_gain_sweep(1:10, 'CsvFile', "gain_sweep.csv");
```

```bash
camera-noise-characterization fit gain_sweep.csv
```

## What changed compared with the original lab scripts

- Scripts became functions with validated arguments instead of edit-and-rerun constants.
- The gain sweep finds the exposure automatically instead of by trial and error.
- Exposure sweeps report per-frame noise and export CSV, so no value is typed by hand.
- The camera is always released, even if a script stops with an error.
- The edge comparison runs on any image, defaulting to a real frame from the lab camera.

## Requirements

MATLAB R2019b or newer, Image Acquisition Toolbox with the *GigE Vision Hardware* support
package, Image Processing Toolbox. `ExposureTimeAbs` and `GainFactor` are the GenICam feature
names of the Baumer TXG series; other cameras may call them `ExposureTime` and `Gain`.
