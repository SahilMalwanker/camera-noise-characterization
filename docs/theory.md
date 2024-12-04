# Theory notes

The equations implemented in `camera-noise-characterization`, with the assumptions behind them.

## 1. Thin-lens field calculations - `camera_noise_characterization.optics`

For a lens of focal length $f$ focused on an object at distance $a$, the lateral magnification is

$$ m = \frac{f}{a - f}. $$

Everything else follows from it:

| Quantity | Formula | Lab set-up (12 mm, f/1.4, 342 mm) |
| --- | --- | --- |
| field of view | $\mathrm{FOV} = d_\text{sensor} / m$ | 131.97 x 99.23 mm |
| field of view, pinhole shortcut | $d_\text{sensor}\, a / f$ | 136.77 x 102.84 mm |
| object-plane pixel | $p / m$ | 0.128 mm |
| angle of view | $2 \arctan\!\big(d_\text{sensor} / 2f\big)$ | 22.62 deg x 17.10 deg |

The pinhole shortcut ignores that the image plane sits at $a f/(a-f)$ rather than at $f$; at
342 mm it overstates the field by 3.6 %.

**Depth of field.** With f-number $N$ and acceptable circle of confusion $u$ (default: one pixel),

$$ a_{n,f} = \frac{a}{1 \pm N u \,(a - f) / f^2}, \qquad
   H = \frac{f^2}{N u} + f . $$

The far limit runs to infinity once $a \ge H$ (the hyperfocal distance, 22.1 m in the lab).
For the lab: $a_n = 336.97$ mm, $a_f = 347.18$ mm, 10.2 mm in total.

**Diffraction.** The Airy disk has diameter $2.44\,\lambda N$. Once it covers two pixels,
stopping down no longer sharpens anything; for 4.65 um pixels and $\lambda = 550$ nm that
happens beyond f/6.9. Depth of field grows roughly linearly with $N$, the exposure time needed
grows with $N^2$ - the trade-off shown in `docs/figures/optics_tradeoffs.png`.

## 2. Sensor model - `camera_noise_characterization.sensor`

The digital twin follows the linear camera model of the EMVA 1288 standard:

$$ \mu_e = \eta\, \mu_p\,(1 + \text{PRNU}) + \mu_\text{dark}, \qquad
   y = \operatorname{round}\!\Big( g K \big(\mathcal{P}(\mu_e) + \text{DSNU} + \mathcal{N}(0, \sigma_r^2)\big) + y_\text{black} \Big) $$

with quantum efficiency $\eta$, Poisson photo-electrons $\mathcal{P}$, analogue gain $g$,
system gain $K$ in DN per electron, read noise $\sigma_r$ in electrons, full-well clipping and
an ideal ADC. PRNU and DSNU are frozen per sensor (fixed pattern); everything else is redrawn
per frame.

## 3. Photon transfer - `camera_noise_characterization.noise`

**Why two frames.** Subtracting two frames taken with identical settings cancels the scene,
shading, PRNU and DSNU. What remains is temporal noise, and since the difference contains two
independent draws,

$$ \sigma_\text{frame} = \frac{\operatorname{std}(A - B)}{\sqrt{2}} . $$

**The photon-transfer line.** Photo-electrons are Poisson distributed, so their variance equals
their mean. Propagated through the gain $K$:

$$ \sigma_y^2 = K\,\mu_y + K^2 \sigma_r^2 + \tfrac{1}{12} . $$

A straight-line fit of variance against mean gives the conversion gain $K$ as the slope and the
noise floor as the intercept - without a photometer and without trusting a datasheet. Points
close to full scale are excluded because clipping removes the upper tail of the distribution.

**Gain sweep.** If the exposure is re-adjusted so the mean stays at $\mu$ for every analogue gain
$g$, the per-frame variance becomes

$$ \sigma^2(g) = K_1\, g\, \mu + (K_1 \sigma_r)^2 g^2 + \tfrac{1}{12} . $$

Shot noise grows like $\sqrt{g}$, read noise added before the amplifier like $g$. The model is
linear in its two unknowns, so ordinary least squares separates them and returns standard
errors. On the lab data the two terms cross at $g \approx 6.8$.

**SNR.** $\mathrm{SNR} = \mu / \sigma$ rises with slope 1 (log-log) while read noise dominates
and with slope 1/2 once shot noise dominates - the kink sits where $K \mu = K^2 \sigma_r^2$,
about 20 DN for the TXG08. Gain cannot move a frame along this curve; only more photons can.

## 4. Denoising - `camera_noise_characterization.filters`

A linear filter with weights $w$ applied to white noise scales its standard deviation by

$$ G = \sqrt{\textstyle\sum w^2}. $$

For separable binomial kernels this equals the sum of squares of the 1-D row:
$6/16 = 0.375$ (3x3), $70/256 = 0.273$ (5x5), $924/4096 = 0.226$ (7x7).
The median of $n$ Gaussian samples has a standard deviation of about $\sqrt{\pi / 2n}$
(0.41 for the 3x3 window, computed exactly by Monte Carlo). On 8-bit data the median measures
0.44 instead: its output stays on the integer grid, so quantisation noise survives. Against
impulse noise the ranking flips - the median ignores outliers, a linear filter smears them.

## 5. Edges - `camera_noise_characterization.edges`

- **Sobel / Prewitt**: first derivatives with 1-2-1 or 1-1-1 smoothing across the edge; edges
  are pixels whose gradient magnitude exceeds twice its rms (MATLAB's default rule).
- **Laplacian**: second derivative, zero at the edge, very noise sensitive. The 3x3 kernel family
  with diagonal weight $\alpha$ matches MATLAB's `fspecial('laplacian', alpha)`.
- **LoG**: Gaussian smoothing first, then zero crossings with a slope threshold
  (Marr-Hildreth).
- **Canny**: Gaussian smoothing, Sobel gradients, non-maximum suppression along the gradient
  direction (4 sectors) and hysteresis thresholding via connected components. Thresholds default
  to the 70 % non-edge rule; `non_edge_fraction` raises it for textured scenes.

The benchmark renders an anti-aliased Siemens star: every pixel that straddles a spoke boundary
is a ground-truth edge, the unresolvable centre is masked, and precision/recall are counted with
a 1.5-pixel tolerance. The result: the smoothing scale, not the operator, decides robustness -
Canny with $\sigma = 2.5$ keeps F1 = 0.95 at a contrast-to-noise ratio of 2.
