#!/usr/bin/env bash

# BMW model classes
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
        #export OUTPUT_DIR="./exps/${concept}"
        #export CLASS_DIR="prior_data/${concept}"

        #mkdir -p $CLASS_DIR

        # NOTE: different tokens must be used for each concept
        accelerate launch --mixed_precision fp16 train_dreambooth_lora.py \
          --pretrained_model_name_or_path=$MODEL_NAME \
          --instance_data_dir=$INSTANCE_DIR \
          --output_dir=$OUTPUT_DIR \
          --instance_prompt="A photo of an sks" \
          --resolution=512 \
          --train_batch_size=2 \
          --gradient_accumulation_steps=4 \
          --max_grad_norm=1 \
          --learning_rate=1e-4 \
          --lr_scheduler='constant' \
          --lr_warmup_steps=0 \
          --max_train_steps=200 \
          --checkpointing_steps=200 \
          --dataloader_num_workers=8 \
          --use_8bit_adam \
          --gradient_checkpointing \
          --lora_rank=$LORA_RANK \
          --seed=42
          #--class_data_dir=$CLASS_DIR \
          #--with_prior_preservation \
          #--instance_prompt="A <${concept}> ${class}" \
          #--class_prompt="A ${class}" \
          #--num_class_image=200 \
    done
done
