clear;
close all;
%% settings

folder ='./DIV2K_val_HR';
savepath = 'val_DIV2K_96_8.h5';
size_label =96;
scale = 4;
size_input = size_label/scale;

stride=96;
%% initialization

data = zeros(size_input, size_input, 1, 1);
label = zeros(size_label, size_label, 1, 1);
count = 0;

%% generate data
filepaths = [];
filepaths = [filepaths; dir(fullfile(folder, '*.jpg'))];
filepaths = [filepaths; dir(fullfile(folder, '*.bmp'))];
filepaths = [filepaths; dir(fullfile(folder, '*.png'))];

for i = 1 : length(filepaths)
        image = imread(fullfile(folder,filepaths(i).name));
        disp(filepaths(i).name);
        if size(image,3)==3
            image = rgb2ycbcr(image);
            image = im2double(image(:, :, 1));
        else
            image = im2double(image);  
        end
        im_label = modcrop(image, scale);
        [hei,wid] = size(im_label);

            for x = 1 : stride : hei-size_label+1
                for y = 1 :stride : wid-size_label+1
                    sub_label = im_label(x : x+size_label-1, y : y+size_label-1);
                     
                    sub_input = imresize(sub_label,1/scale,'bicubic');

                    count=count+1;
                    data(:, :, 1, count) = sub_input;
                    label(:, :, 1, count) = sub_label;
                end
            end
        
end

order = randperm(count);
data= data(:, :, 1, order);
label= label(:, :, 1, order); 

%% writing to HDF5
chunksz = 128;
created_flag = false;
totalct = 0;

for batchno = 1:floor(count/chunksz)
    last_read=(batchno-1)*chunksz;
    batchdata = data(:,:,1,last_read+1:last_read+chunksz);
    batchlabs = label(:,:,1,last_read+1:last_read+chunksz);

    startloc = struct('dat',[1,1,1,totalct+1], 'lab', [1,1,1,totalct+1]);
    curr_dat_sz = store2hdf5(savepath, batchdata, batchlabs, ~created_flag, startloc, chunksz);
    created_flag = true;
    totalct = curr_dat_sz(end);
end
h5disp(savepath);


