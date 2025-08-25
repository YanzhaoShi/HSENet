#!/bin/bash
# run "accelerate config" first!

accelerate launch LaMed/src/train/train_VLM.py \
    --cap_data_path "/path/to/vqa_data.json" \
    --use_training_data "openvqa"\
    --ablation_type "dualvits_spatialpacker" \
    --vision_tower "vit_stage2_dual_encoders" \
    --mm_projector_type 'VisualPacker_3d_phi_v3' \
    --proj_layer_type "mlp" \
    --model_max_length 330 \
    --output_dir "/path/to/save_dir" \
    --num_train_epochs 4 \
    --learning_rate 5e-5 \
    --per_device_train_batch_size 5 \
    --per_device_eval_batch_size 10 \
    --version v0 \
    --tune_mm_mlp_adapter True \
    --bf16 True \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "steps" \
    --eval_accumulation_steps 1 \
    --eval_steps 0.04 \
    --save_strategy "steps" \
    --save_total_limit 1 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 0.001 \
    --gradient_checkpointing False \
    --dataloader_pin_memory True\
    --dataloader_num_workers 24 \
    --report_to tensorboard
