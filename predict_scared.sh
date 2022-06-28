###########generate 1080*1024 on scared2019_small --maxdisp=192 ######################
# CUDA_VISIBLE_DEVICES=0 python predict.py \
#                 --scared2019_small=1   --dataset scared2019_small --maxdisp=192 \
#                 --crop_height=1032  --crop_width=1296  \
#                 --data_path='./dataset/SCARED2019_small/' \
#                 --test_list='./dataloaders/lists/scared2019_test_small_sttr.list' \
#                 --save_path='./predict/scared2019_small/LEAStereo/' \
#                 --resume './pretrained_ckpt/leastereo_sf/checkpoint/best.pth' 

# CUDA_VISIBLE_DEVICES=0 python predict.py \
#                 --cfg='configs/transtereo_test.yaml' \
#                 --scared2019_small=1  --dataset scared2019 --maxdisp=192 \
#                 --crop_height=1176  --crop_width=1176  \
#                 --data_path='./dataset/SCARED2019_small/' \
#                 --test_list='./dataloaders/lists/scared2019_test_small_sttr.list' \
#                 --save_path='./predict/scared2019_small/HybridStereo/' \
#                 --resume './run/hybrid_best_0.83.pth' 

###########generate 1280*1024 on Scared --maxdisp=192 ######################
# CUDA_VISIBLE_DEVICES=0 python predict.py \
#                 --scared2019=1  --dataset scared2019   --maxdisp=192 \
#                 --crop_height=1032  --crop_width=1296  \
#                 --data_path='./dataset/SCARED2019/' \
#                 --test_list='./dataloaders/lists/scared2019_test.list' \
#                 --save_path='./predict/scared2019/LEAStereo/' \
#                 --resume './pretrained_ckpt/leastereo_sf/checkpoint/best.pth' 

# CUDA_VISIBLE_DEVICES=0 python predict.py \
#                 --cfg='configs/transtereo_test.yaml' \
#                 --scared2019=1  --dataset scared2019 --maxdisp=192 \
#                 --crop_height=1344  --crop_width=1344  \
#                 --data_path='./dataset/SCARED2019/' \
#                 --test_list='./dataloaders/lists/scared2019_test.list' \
#                 --save_path='./predict/scared2019/HybridStereo/' \
#                 --resume './run/hybrid_best_0.83.pth' 


###########generate 1280*1024 on Scared test keyframe --maxdisp=263 ######################
# CUDA_VISIBLE_DEVICES=0 python predict.py \
#                 --cfg='configs/transtereo_test.yaml' \
#                 --scared2019_small=1  --dataset scared2019 --maxdisp=263 \
#                 --crop_height=1344  --crop_width=1344  \
#                 --data_path='./dataset/SCARED2019_small/' \
#                 --test_list='./dataloaders/lists/scared2019_test_small.list' \
#                 --save_path='./predict/scared2019_test_small/HybridStereo_192/' \
#                  --resume './run/hybrid_best_0.83.pth' 

# CUDA_VISIBLE_DEVICES=0 python predict.py \
#                 --scared2019_small=1   --dataset scared2019 --maxdisp=263 \
#                 --crop_height=1032  --crop_width=1296  \
#                 --data_path='./dataset/SCARED2019_small/' \
#                 --test_list='./dataloaders/lists/scared2019_test_small.list' \
#                 --save_path='./predict/scared2019_test_small/LEAStereo/' \
#                 --resume './pretrained_ckpt/leastereo_sf/checkpoint/best.pth' 

###########generate 1280*1024 on Scared test keyframe --maxdisp=263 ######################
# CUDA_VISIBLE_DEVICES=0 python predict.py \
#                 --cfg='configs/transtereo_test.yaml' \
#                 --scared2019_small=1  --dataset scared2019 --maxdisp=263 \
#                 --crop_height=1344  --crop_width=1344  \
#                 --data_path='./dataset/SCARED2019_small/' \
#                 --test_list='./dataloaders/lists/scared2019_test19_all.list' \
#                 --save_path='./predict/scared2019_test19_all/HybridStereo/' \
#                  --resume './run/hybrid_best_0.83.pth' 

CUDA_VISIBLE_DEVICES=0 python predict.py \
                --scared2019_small=1   --dataset scared2019 --maxdisp=263 \
                --crop_height=1032  --crop_width=1296  \
                --data_path='./dataset/SCARED2019_small/' \
                --test_list='./dataloaders/lists/scared2019_test19_all.list' \
                --save_path='./predict/scared2019_test19_all/LEAStereo/' \
                --resume './pretrained_ckpt/leastereo_sf/checkpoint/best.pth' 