clear;close all;
%% settings
folder = 'Set5';
scale = 4;

%% generate data
filepaths = [];
filepaths = [filepaths; dir(fullfile(folder, '*.jpg'))];
filepaths = [filepaths; dir(fullfile(folder, '*.bmp'))];
filepaths = [filepaths; dir(fullfile(folder, '*.png'))];

for i = 1 : length(filepaths)        
    image = imread(fullfile(folder,filepaths(i).name));
    image = modcrop(image, scale);
    im_gt = double(image);
    [H,W,C] = size(im_gt);
    t = find('.'==filepaths(i).name);
    imname = filepaths(i).name(1:t-1);
    filename = sprintf('Set5-output-4/%s.mat',imname);
    if C==3
        im_gt_ycbcr = rgb2ycbcr(im_gt / 255.0);
        im_gt_y = im_gt_ycbcr(:,:,1) * 255.0;
        im_l_y = imresize(im_gt_y,1/scale,'bicubic');
        im_b_y = imresize(im_l_y,scale,'bicubic');
        
        im_gt_cbcr = im_gt_ycbcr(:,:,2:3);
        im_b_cbcr = imresize(imresize(im_gt_cbcr,1/scale,'bicubic'),scale,'bicubic');
        
    save(filename,'C' ,'im_gt_y','im_b_y','im_l_y', 'im_b_cbcr');
    else 
        im_gt_y = im_gt;
        im_l_y = imresize(im_gt_y/255.0,1/scale,'bicubic')*255.0;
        im_b_y = imresize(im_l_y/255.0,scale,'bicubic')*255.0;
    save(filename,'C', 'im_gt_y','im_b_y','im_l_y');
    end
    
end
