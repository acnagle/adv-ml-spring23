
#!/usr/bin/env bash

class_names=("cat" "automobile" "bird" "airplane" "deer" "dog" "frog" "horse" "ship" "truck")

for c in "${!class_names[@]}"
do

	class="${class_names[c]}"
	echo $class
	export MODEL_NAME="runwayml/stable-diffusion-v1-5"
	export INSTANCE_DIR="/home/disha/Documents/lora/contents/CIFAR-10-images/train/${class}"

	export OUTPUT_DIR="/home/disha/Documents/lora/contents/training_output/${class}"
	export CLASS_DIR="/home/disha/Documents/lora/contents/cifar10_generated/train/${class}"

	mkdir -p $OUTPUT_DIR

	accelerate launch ../training_scripts/train_lora_dreambooth.py \
      --pretrained_model_name_or_path=$MODEL_NAME \
      --instance_data_dir=$INSTANCE_DIR \
      --output_dir=$OUTPUT_DIR \
      --class_data_dir=$CLASS_DIR \
      --instance_prompt="A sks ${class}" \
      --class_prompt="A ${class}" \
      --with_prior_preservation \
      --num_class_image=200 \
      --resolution=32 \
      --train_batch_size=2 \
      --gradient_accumulation_steps=4 \
      --max_grad_norm=1 \
      --learning_rate=5e-6 \
      --lr_scheduler='constant' \
      --lr_warmup_steps=0 \
      --max_train_steps=200 \
      --gradient_checkpointing \
      --seed=42
done
