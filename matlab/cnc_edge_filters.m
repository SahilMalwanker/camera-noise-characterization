function cnc_edge_filters(imagePath)
%CNC_EDGE_FILTERS Compare first- and second-order edge operators on one image.
%   CNC_EDGE_FILTERS(imagePath) shows Sobel and Prewitt derivatives in x and y, the
%   3x3 Laplacian, Laplacian-of-Gaussian zero crossings and the Canny detector.
%   Without an argument the real camera frame in ../data is used.
arguments
    imagePath (1,1) string = fullfile(fileparts(mfilename('fullpath')), '..', 'data', 'coins_txg08.png')
end

I = imread(imagePath);
if size(I, 3) == 3
    I = rgb2gray(I);
end
I = im2double(I);

sobel = -fspecial('sobel');      % rows: d/dy, positive where brightness grows downwards
prewitt = -fspecial('prewitt');
panels = {
    'original',               I
    'Sobel x',                imfilter(I, sobel.', 'replicate')
    'Sobel y',                imfilter(I, sobel, 'replicate')
    'Laplacian (alpha 0.2)',  imfilter(I, fspecial('laplacian', 0.2), 'replicate')
    'Prewitt x',              imfilter(I, prewitt.', 'replicate')
    'Prewitt y',              imfilter(I, prewitt, 'replicate')
    'LoG zero crossings',     edge(I, 'log')
    'Canny',                  edge(I, 'canny')
    };

figure('Name', 'Edge operators');
tiledlayout(2, 4, 'TileSpacing', 'compact', 'Padding', 'compact');
for k = 1:size(panels, 1)
    nexttile;
    imshow(panels{k, 2}, []);
    title(panels{k, 1});
end
end
