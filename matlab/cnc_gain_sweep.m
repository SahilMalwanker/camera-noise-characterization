function T = cnc_gain_sweep(gains, options)
%CNC_GAIN_SWEEP Temporal noise at a constant mean signal for several analogue gains.
%   T = CNC_GAIN_SWEEP(gains) re-adjusts the exposure time for every gain until the
%   mean signal is back at TargetMean, then measures a frame pair. With the signal
%   held constant, any growth of the noise is caused by the amplifier itself.
%
%       T = cnc_gain_sweep(1:10, 'CsvFile', "gain_sweep.csv");
%       % shell:  camera-noise-characterization fit gain_sweep.csv
%
%   Name-value options: TargetMean (128 DN), Tolerance (2 DN), StartExposure (8000 us),
%   Device (1), CsvFile.
arguments
    gains (1,:) double {mustBePositive} = 1:10
    options.TargetMean (1,1) double = 128
    options.Tolerance (1,1) double = 2
    options.StartExposure (1,1) double {mustBePositive} = 8000
    options.Device (1,1) double = 1
    options.CsvFile (1,1) string = ""
end

vid = videoinput('gige', options.Device, 'Mono8');
cleanup = onCleanup(@() delete(vid));
src = getselectedsource(vid);

n = numel(gains);
[exposure_us, mean_dn, sigma_diff_dn] = deal(zeros(n, 1));
t = options.StartExposure;
for k = 1:n
    src.GainFactor = gains(k);
    t = autoExpose(vid, src, t, options.TargetMean, options.Tolerance);
    s = cnc_noise_pair(vid);
    exposure_us(k) = t;
    mean_dn(k) = s.mean;
    sigma_diff_dn(k) = s.sigmaDiff;
    fprintf('gain %4.1f   t = %6d us   mean = %6.2f DN   std(A-B) = %.3f DN\n', ...
        gains(k), t, s.mean, s.sigmaDiff);
end
gain = gains(:);
T = table(exposure_us, gain, mean_dn, sigma_diff_dn);

figure('Name', 'Gain sweep');
plot(T.gain, T.sigma_diff_dn, 'o-');
grid on;
xlabel('analogue gain');
ylabel('std(A - B) [DN]');
title(sprintf('noise at a constant %.0f DN', options.TargetMean));

if strlength(options.CsvFile) > 0
    writetable(T, options.CsvFile);
end
end

function t = autoExpose(vid, src, t, target, tolerance)
% Proportional steps converge quickly because the sensor response is linear;
% a clipped frame halves the exposure instead.
m = NaN;
for iteration = 1:15
    src.ExposureTimeAbs = round(t);
    pause(0.2);
    m = mean(double(getsnapshot(vid)), 'all');
    if abs(m - target) <= tolerance
        t = round(t);
        return
    end
    if m >= 254
        t = t / 2;
    else
        t = t * target / max(m, 1);
    end
    t = min(max(t, 4), 60e6);
end
t = round(t);
warning('cnc:autoExpose', 'Target %.0f DN not reached, last mean %.1f DN.', target, m);
end
