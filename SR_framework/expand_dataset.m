clc
close all
clear all
rotate = [];        %设置旋转角度
down_scale = [];    %设置图像缩放系数，这个值需要小于1
is_flipud = false;  %设置是否上下翻转
is_fliplr = false;  %设置是否水平翻转

folder_list = dir('./data/train/');
for i = 3 : length(folder_list)
    dataset = ['./data/train/', folder_list(i).name, '/'];
    for pos = 1 : 4
        switch pos
            case 1
                disp("Now rotating...")
            case 2
                disp("Now doing down-sample...")
            case 3
                disp("Now doing flip horizontally...")
            case 4
                disp("Now doing vertical flip...")
        end
        img_path = dir(fullfile(dataset,'*.*'));
        for num = 3 : length(img_path)
            [add, imname, type] = fileparts(img_path(num).name);
            im = imread([dataset imname type]);
            switch pos
                case 1
                    for r = 1 : length(rotate)
                        rot = rotate(r) / 90;
                        im_new = rot90(im, rot);
                        imname_n = [imname, '_rot', num2str(rot)];
                        imwrite(im_new, [dataset, imname_n, type]);
                    end
                case 2
                    for d = 1 : length(down_scale)
                        im_new = imresize(im, down_scale(d), 'bicubic');
                        imname_n = [imname, '_down', num2str(down_scale(d))];
                        imwrite(im_new, [dataset, imname_n, type]);
                    end
                case 3
                    if is_flipud
                        im_new = flipud(im);
                        imname_n = [imname, '_pud'];
                        imwrite(im_new, [dataset, imname_n, type]);
                    end
                case 4
                    if is_fliplr
                        im_new = fliplr(im);
                        imname_n = [imname, '_plr'];
                        imwrite(im_new, [dataset, imname_n, type]);
                    end
            end
        end
    end
end