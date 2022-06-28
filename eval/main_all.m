clear 
close all

gtDir = '../dataset/SCARED2019_small/disp_left';   
% salDir = '../predict/scared2019_small/STTR'; 
salDir = '../predict/scared2019_small/LEAStereo'; 
% salDir = '../predict/scared2019_small/HybridStereo';

seqPath = [salDir '/'];  % sequence Path
seqFiles = dir(seqPath);
seqNUM = length(seqFiles)-4; % remove unwanted names

%% Parameter setting
num_samples = seqNUM;
epe = zeros(1,num_samples);
rmse= zeros(1,num_samples);
a1  = zeros(1,num_samples);
a2  = zeros(1,num_samples);
a3  = zeros(1,num_samples);
err_2 = zeros(1,num_samples);
err_3 = zeros(1,num_samples);
err_5 = zeros(1,num_samples);
density = zeros(1,num_samples);

for i = 1:num_samples
    %% read lidar points
    
    name = seqFiles(i+2).name;
    stereo_disp_name_gt  = [gtDir '/' name];
    stereo_disp_name_est = [salDir '/' name];
    
    stereo_disp_gt = double(read(Tiff(stereo_disp_name_gt,'r')));
    stereo_disp_est= double(read(Tiff(stereo_disp_name_est,'r')));
   
    [h,w] = size(stereo_disp_gt);
    mask1 = stereo_disp_est>0 & stereo_disp_gt>0;
    mask2 = stereo_disp_gt>0;
    density(i) = sum(mask1(:))/sum(mask2(:));
       
    err_3(i) = disp_error(stereo_disp_gt,stereo_disp_est,[3 0.05]);  
    err_2(i) = disp_error(stereo_disp_gt,stereo_disp_est,[2 0.05]);  
    err_5(i) = disp_error(stereo_disp_gt,stereo_disp_est,[5 0.05]);  
    diff = stereo_disp_gt(mask2) - stereo_disp_est(mask2);
    epe(i)   = mae(abs(diff));
    
    diff_sq  = diff.^ 2;
    rmse(i)  = sqrt(mean(diff_sq));
    
    name_stereo = name;
    fprintf('%s results: EPE %.4f, rmse %.4f, bad_2 %.4f, bad_3 %.4f, bad_5 %.4f \n',...
            name_stereo, epe(i),  rmse(i), err_2(i), err_3(i), err_5(i))
end

epe_mean  = mean(epe);
rmse_mean = mean(rmse);
err2_mean = mean(err_2); err3_mean = mean(err_3); err5_mean = mean(err_5);

fprintf('mean results: \n EPE %.4f, rmse %.4f, bad_2 %.4f, bad_3 %.4f, bad_5 %.4f',...
                     epe_mean, rmse_mean, err2_mean*100,  err3_mean*100,  err5_mean*100)

function img_log = logarithm(img,c)
img_log = c*log(double(img)+1);
img_log = im2uint8(mat2gray(img_log));
end
