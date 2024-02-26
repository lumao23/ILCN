python -m torch.distributed.launch \
    --nproc_per_node=2 \
    --use_env main.py \
    --batch_size 4 \
    --backbone swin_tiny \
    --pretrained ./params/swin_tiny_iterative_box_refinement_COCO.pth \
    --output_dir hico_logs/all_flow \
    --epochs 100 \
    --lr_drop 60 \
    --num_feature_levels 4 \
    --num_queries 128 \
    --use_nms