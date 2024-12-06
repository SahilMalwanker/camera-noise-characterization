# Data

| File | Content | Provenance |
| --- | --- | --- |
| `lab_gain_sweep.csv` | gain 1-10, exposure, `std(A - B)` at a constant mean of about 129 DN | Baumer TXG08, 8-bit, white target |
| `lab_exposure_sweep.csv` | `std(A - B)` and relative noise at six signal levels, gain 1 | same camera and target |
| `coins_txg08.png` | 1032 x 776 frame of coins under front light | same camera, used for the edge gallery |

## How the numbers were recovered

The lab values were partly typed into the MATLAB workspace by hand. The `source` column marks
each value:

- **logged** - read directly from the workspace (gains 1-4 noise, gains 1-5 exposure, the exposure-sweep noise).
- **digitised** - read back from the plotted lab figures with a pixel-accurate digitiser. The
  digitiser reproduced every logged noise value to within 0.01 DN, so the remaining noise values
  carry an uncertainty of roughly +-0.03 DN and the exposure times of gains 6-10 about +-10 us.
- For the exposure sweep the mean was not logged directly; it follows from the logged noise and
  the logged relative noise as `mean = std(A - B) / relative_noise`.

## Why some rows are excluded

`include = 0` rows are kept for transparency but are not fitted:

- **Gain 9 and 10**: exposure x gain falls 7 % and 43 % below the constant value of the other
  settings, so these frames held less signal and lower shot noise.
- **Exposure sweep, last two rows**: within 4 % of full scale, where clipping starts to squeeze the
  variance (the very last one even implies a mean above 255 DN after rounding).
