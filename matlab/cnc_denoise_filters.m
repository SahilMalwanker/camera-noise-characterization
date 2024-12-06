function T = cnc_denoise_filters(a, b)
%CNC_DENOISE_FILTERS Temporal noise of a flat-field pair before and after filtering.
%   T = CNC_DENOISE_FILTERS(A, B) filters both frames with 3x3, 5x5 and 7x7 binomial
%   kernels and a 3x3 median, re-measures std(A - B) / sqrt(2) and compares the
%   reduction with the white-noise prediction sqrt(sum(w.^2)) of each linear kernel.
%
%   Example (A, B: two frames of a white target taken with identical settings)
%       T = cnc_denoise_filters(imread('flat_a.tif'), imread('flat_b.tif'))
a = double(a);
b = double(b);
filterName = ["unfiltered"; "binomial 3x3"; "binomial 5x5"; "binomial 7x7"; "median 3x3"];
noise_dn = zeros(5, 1);
white_noise_theory = [1; NaN(4, 1)];

noise_dn(1) = std(a(:) - b(:)) / sqrt(2);
sizes = [3 5 7];
for k = 1:numel(sizes)
    w = binomialKernel(sizes(k));
    fa = imfilter(a, w, 'symmetric');
    fb = imfilter(b, w, 'symmetric');
    noise_dn(k + 1) = std(fa(:) - fb(:)) / sqrt(2);
    white_noise_theory(k + 1) = sqrt(sum(w(:) .^ 2));
end
ma = medfilt2(a, [3 3], 'symmetric');
mb = medfilt2(b, [3 3], 'symmetric');
noise_dn(5) = std(ma(:) - mb(:)) / sqrt(2);
white_noise_theory(5) = 0.41;  % median of 9 Gaussian samples (Monte Carlo)

measured_gain = noise_dn / noise_dn(1);
T = table(filterName, noise_dn, measured_gain, white_noise_theory);
disp(T);
end

function w = binomialKernel(n)
row = arrayfun(@(k) nchoosek(n - 1, k), 0:n - 1);
row = row / sum(row);
w = row.' * row;
end
