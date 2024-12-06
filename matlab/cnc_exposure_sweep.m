function T = cnc_exposure_sweep(tStartUs, tStopUs, nSteps, options)
%CNC_EXPOSURE_SWEEP Mean signal and temporal noise for evenly spaced exposure times.
%   T = CNC_EXPOSURE_SWEEP(tStartUs, tStopUs, nSteps) keeps the gain fixed, steps the
%   exposure time from tStartUs to tStopUs and measures a frame pair at every step.
%   Point the camera at an evenly lit white target and disable auto exposure/gain.
%
%   The returned table uses the column names of the Python package, so the
%   photon-transfer fit is one command away:
%       T = cnc_exposure_sweep(1920, 14080, 20, 'CsvFile', "exposure_sweep.csv");
%       % shell:  camera-noise-characterization fit exposure_sweep.csv
%
%   Name-value options: Gain (default 1), Device (default 1), CsvFile.
arguments
    tStartUs (1,1) double {mustBePositive}
    tStopUs (1,1) double {mustBePositive}
    nSteps (1,1) double {mustBeInteger, mustBePositive} = 20
    options.Gain (1,1) double {mustBePositive} = 1
    options.Device (1,1) double = 1
    options.CsvFile (1,1) string = ""
end

vid = videoinput('gige', options.Device, 'Mono8');
cleanup = onCleanup(@() delete(vid));
src = getselectedsource(vid);
src.GainFactor = options.Gain;

exposure_us = round(linspace(tStartUs, tStopUs, nSteps)).';
[mean_dn, sigma_diff_dn] = deal(zeros(nSteps, 1));
for k = 1:nSteps
    src.ExposureTimeAbs = exposure_us(k);
    pause(0.2);  % let the new exposure time reach the sensor
    s = cnc_noise_pair(vid);
    mean_dn(k) = s.mean;
    sigma_diff_dn(k) = s.sigmaDiff;
    fprintf('%2d/%d  t = %6d us   mean = %6.2f DN   std(A-B) = %.3f DN\n', ...
        k, nSteps, exposure_us(k), s.mean, s.sigmaDiff);
end
gain = repmat(options.Gain, nSteps, 1);
T = table(exposure_us, gain, mean_dn, sigma_diff_dn);

figure('Name', 'Exposure sweep');
subplot(1, 2, 1);
plot(T.mean_dn, T.sigma_diff_dn .^ 2 / 2, 'o-');
grid on;
xlabel('mean signal [DN]');
ylabel('temporal variance [DN^2]');
title('photon transfer: slope = conversion gain');
subplot(1, 2, 2);
plot(T.exposure_us, T.sigma_diff_dn ./ T.mean_dn, 'o-');
grid on;
xlabel('exposure time [\mus]');
ylabel('std(A - B) / mean');
title('relative noise');

if strlength(options.CsvFile) > 0
    writetable(T, options.CsvFile);
end
end
