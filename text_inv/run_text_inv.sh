#!/usr/bin/env bash
# Adapted from https://huggingface.co/docs/diffusers/training/text_inversion

export CUDA_VISIBLE_DEVICES=1

concept_names=("backpack" "backpack_dog" "bear_plushie" "berry_bowl" "can" "candle" "cat" "cat2" "clock" "colorful_sneaker" "dog" "dog2" "dog3" "dog5" "dog6" "dog7" "dog8" "duck_toy" "fancy_boot" "grey_sloth_plushie" "monster_toy" "pink_sunglasses" "poop_emoji" "rc_car" "red_cartoon" "robot_toy" "shiny_sneaker" "teapot" "vase" "wolf_plushie")
class_names=("backpack" "backpack" "animal" "bowl" "can" "candle" "cat" "cat" "clock" "sneaker" "dog" "dog" "dog" "dog" "dog" "dog" "dog" "toy" "boot" "animal" "toy" "glasses" "toy" "toy" "cartoon" "toy" "sneaker" "teapot" "vase" "animal")

for c in "${!concept_names[@]}"
do
    concept="${concept_names[c]}"
    class="${class_names[c]}"

    export MODEL_NAME="runwayml/stable-diffusion-v1-5"
    export INSTANCE_DIR="/home/an28622/data/dreambooth-main/dataset/${concept}"
    export OUTPUT_DIR="./exps/${concept}"

    accelerate launch textual_inversion.py \
      --pretrained_model_name_or_path=$MODEL_NAME \
      --train_data_dir=$INSTANCE_DIR \
      --output_dir=$OUTPUT_DIR \
      --resolution=512 \
      --train_batch_size=1 \
      --gradient_accumulation_steps=4 \
      --scale_lr \
      --learning_rate=5.0e-04 \
      --lr_scheduler="linear" \
      --lr_warmup_steps=0 \
      --placeholder_token="<${concept}>" \
      --initializer_token="${class}" \
      --learnable_property="object" \
      --checkpointing_steps=500 \
      --max_train_steps=3000 \
      --adam_weight_decay=1e-2 \
      --seed=42
done
