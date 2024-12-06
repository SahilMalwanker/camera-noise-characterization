function stats = cnc_noise_pair(vid, options)
%CNC_NOISE_PAIR Mean signal and temporal noise from two consecutive frames.
%   stats = CNC_NOISE_PAIR(vid) grabs two frames with the current camera settings.
%   Subtracting them cancels the scene, shading and fixed-pattern noise, so only
%   temporal noise is left. Because the difference holds two independent noise
%   samples, the noise of a single frame is std(A - B) / sqrt(2).
%
%   Fields of stats
%       mean          - mean signal of both frames [DN]
%       sigmaDiff     - std(A - B) [DN]
%       sigma         - temporal noise of one frame [DN]
%       relativeNoise - sigmaDiff / mean
%
%   Example
%       vid = videoinput('gige', 1, 'Mono8');
%       src = getselectedsource(vid);
%       src.ExposureTimeAbs = 8192;  src.GainFactor = 1;
%       s = cnc_noise_pair(vid, 'Show', true)
arguments
    vid
    options.Show (1,1) logical = false
end

a = double(getsnapshot(vid));
b = double(getsnapshot(vid));
d = a - b;

stats.mean = (mean(a, 'all') + mean(b, 'all')) / 2;
stats.sigmaDiff = std(d(:));
stats.sigma = stats.sigmaDiff / sqrt(2);
stats.relativeNoise = stats.sigmaDiff / stats.mean;

if options.Show
    figure('Name', 'Frame difference');
    subplot(1, 2, 1);
    imagesc(d);
    axis image off;
    colorbar;
    title('A - B (signed, pseudo-colour)');
    subplot(1, 2, 2);
    histogram(d(:), 'BinMethod', 'integers');
    xlabel('A - B [DN]');
    title(sprintf('std(A - B) = %.3f DN', stats.sigmaDiff));
end
end
