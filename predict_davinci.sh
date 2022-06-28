CUDA_VISIBLE_DEVICES=0 python predict.py \
                --cfg='configs/transtereo_test.yaml' \
                --davinci=1  --dataset davinci  --maxdisp=192 \
                --crop_height=336  --crop_width=504  \
                --data_path='./dataset/daVinci/' \
                --test_list='./dataloaders/lists/davinci_test.list' \
                --save_path='./predict/davinci/' \
                --resume='./run/hybrid_best_0.83.pth' 2>&1 |tee predict/davinci/eval_psnr_best.txt

# CUDA_VISIBLE_DEVICES=1 python predict.py \
#                 --davinci=1   --dataset davinci  --maxdisp=192 \
#                 --crop_height=192  --crop_width=384  \
#                 --data_path='./dataset/daVinci/' \
#                 --test_list='./dataloaders/lists/davinci_test.list' \
#                 --save_path='./predict/davinci/LEAStereo/' \
#                 --resume 'pretrained_ckpt/leastereo_sf/checkpoint/best.pth'  2>&1 |tee predict/davinci/eval_psnr_leastereo.txt
