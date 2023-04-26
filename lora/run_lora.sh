#!/usr/bin/env bash

concept_names=("1" "2" "3" "4" "5" "6" "7" "8" "10" "11")
class_names=()
lora_rank=(16)

for r in "${!lora_rank[@]}"
do
    for c in "${!concept_names[@]}"
    do
        concept="${concept_names[c]}"
        #class="${class_names[c]}"
        rank="${lora_rank[r]}"

        export MODEL_NAME="runwayml/stable-diffusion-v1-5"
        export INSTANCE_DIR="/home/an28622/data/bmw10_multi/train/${concept}"
        export LORA_RANK=$rank
        export OUTPUT_DIR="./exps/rank-${rank}/${concept}"

        lora_pti \
          --pretrained_model_name_or_path=$MODEL_NAME \
          --instance_data_dir=$INSTANCE_DIR \
          --output_dir=$OUTPUT_DIR \
          --train_text_encoder \
          --resolution=512 \
          --train_batch_size=1 \
          --gradient_accumulation_steps=4 \
          --scale_lr \
          --learning_rate_unet=1e-4 \
          --learning_rate_text=1e-5 \
          --color_jitter \
          --lr_scheduler="linear" \
          --lr_warmup_steps=0 \
          --placeholder_tokens="sks" \
          --use_template="object" \
          --save_steps=200 \
          --max_train_steps_tuning=400 \
          --perform_inversion=False \
          --weight_decay_lora=0.001\
          --device="cuda:0" \
          --lora_rank=$LORA_RANK \
          --seed=42
          #--placeholder_tokens="sks" \
    done
done
