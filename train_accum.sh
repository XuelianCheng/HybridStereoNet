CUDA_VISIBLE_DEVICES=0  python3 train_accum.py --accum_iter=3 \
                --batch_size=8 \
                --testBatchSize=8 \
                --crop_height=336  \
                --crop_width=336  \
                --maxdisp=192 \
                --threads=8 \
                --lr=1e-3 \
                --cfg='configs/transtereo_s.yaml' \
                --save_path='./run/exp1/' \
                --load_mn='./pretrained_ckpt/leastereo_sf/checkpoint/best.pth' 2>&1 |tee run/exp1/train.txt
