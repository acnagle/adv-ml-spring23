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
        export OUTPUT_DIR="./exps/rank-$LORA_RANK/${concept}"

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
          --learning_rate_ti=5e-4 \
          --color_jitter \
          --lr_scheduler="linear" \
          --lr_warmup_steps=0 \
          --placeholder_tokens="<${concept}-1>|<${concept}-2>" \
          --use_template="object" \
          --save_steps=100 \
          --max_train_steps_ti=2000 \
          --max_train_steps_tuning=100 \
          --perform_inversion=True \
          --clip_ti_decay \
          --weight_decay_ti=0.000 \
          --weight_decay_lora=0.001\
          --continue_inversion \
          --continue_inversion_lr=1e-4 \
          --device="cuda:1" \
          --lora_rank=$LORA_RANK \
          --seed=42
          #--initializer_tokens="$<rand-0.017>|<rand-0.017>" \
    done
done
