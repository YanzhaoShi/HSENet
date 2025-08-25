#!/bin/bash
# run "accelerate config" first!

accelerate launch LaMed/src/train/train_CLIP_stage1.py \
    --language_model_name_or_path /path/to/bert-base-uncased \
    --version v0 \
    --local_loss False \
    --gather_loss True \
    --bf16 True \
    --output_dir /path/to/save_dir \
    --num_train_epochs 50 \
    --per_device_train_batch_size 3 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "steps" \
    --eval_accumulation_steps 1 \
    --eval_steps 0.04 \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate 1e-4 \
    --weight_decay 0.1 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 0.001 \
    --gradient_checkpointing False \
    --dataloader_pin_memory True\
    --dataloader_num_workers 24 \
    --report_to tensorboard