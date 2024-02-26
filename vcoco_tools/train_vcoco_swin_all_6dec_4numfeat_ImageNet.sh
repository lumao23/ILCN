#!/bin/bash
python -m torch.distributed.launch \
    --nproc_per_node=2\
    --use_env main.py \
    --batch_size 4 \
    --backbone swin_tiny \
    --pretrained params/swin_tiny_patch4_window7_224.pth \
    --output_dir logs/swin_all_no_pretrain \
    --epochs 100 \
    --lr_drop 60 \
    --num_feature_levels 3 \
    --num_queries 128 \
    --dataset_file vcoco \
    --hoi_path data/v-coco \
    --num_obj_classes 81 \
    --num_verb_classes 29 \
    --use_nms \
    --no_obj
    