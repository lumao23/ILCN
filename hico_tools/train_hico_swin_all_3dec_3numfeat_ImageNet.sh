python -m torch.distributed.launch \
    --nproc_per_node=2 \
    --use_env main.py \
    --batch_size 4 \
    --backbone swin_tiny \
    --pretrained ./params/swin_tiny_patch4_window7_224.pth \
    --output_dir hico_logs/nopre_allflow \
    --epochs 100 \
    --lr_drop 60 \
    --num_feature_levels 3 \
    --num_queries 128 \
    --use_nms \
    --dec_layers 3