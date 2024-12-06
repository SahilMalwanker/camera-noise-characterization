function img = cnc_acquire_snapshot(exposureUs, gain, options)
%CNC_ACQUIRE_SNAPSHOT Grab one Mono8 frame from a GigE Vision camera and show its histogram.
%   img = CNC_ACQUIRE_SNAPSHOT(exposureUs, gain) opens a live preview so focus and
%   aperture can be adjusted, waits for a key press, captures one frame and plots it
%   next to its histogram - the quickest check for under- or over-exposure.
%
%   Name-value options
%       Device   - GigE device index (default 1)
%       Preview  - show the live preview first (default true)
%       SaveAs   - file name to store the frame, e.g. "exp8000.tif"
%
%   Example
%       img = cnc_acquire_snapshot(8000, 1, 'SaveAs', "exp8000.tif");
%
%   Requires the Image Acquisition Toolbox with the GigE Vision support package.
%   ExposureTimeAbs / GainFactor are the feature names of the Baumer TXG series.
arguments
    exposureUs (1,1) double {mustBePositive} = 8000
    gain (1,1) double {mustBePositive} = 1
    options.Device (1,1) double = 1
    options.Preview (1,1) logical = true
    options.SaveAs (1,1) string = ""
end

vid = videoinput('gige', options.Device, 'Mono8');
cleanup = onCleanup(@() delete(vid));
src = getselectedsource(vid);
src.ExposureTimeAbs = exposureUs;
src.GainFactor = gain;

if options.Preview
    preview(vid);
    disp('Adjust focus and aperture, then press any key to capture.');
    pause;
    closepreview(vid);
end

img = getsnapshot(vid);

figure('Name', sprintf('Exposure %g us, gain %g', exposureUs, gain));
subplot(1, 2, 1);
imshow(img);
title(sprintf('mean %.1f DN, %.1f %% clipped', mean(img, 'all'), 100 * mean(img(:) == 255)));
subplot(1, 2, 2);
imhist(img);
title('histogram');

if strlength(options.SaveAs) > 0
    imwrite(img, options.SaveAs);
end
end
