import os
import shutil
from tqdm import tqdm

import torch
from lora_diffusion import monkeypatch_or_replace_lora, tune_lora_scale
from diffusers import StableDiffusionPipeline


def generate_images(classes, num_images):
    PRETRAINED_MODEL="runwayml/stable-diffusion-v1-5"
    MODEL_DIR = '/home/disha/Documents/lora/contents/training_output/'
    OUTPUT_DIR = '/home/disha/Documents/lora/contents/cifar0_finetuned_generated'
    for class_name in classes:
        model_path = os.path.join(MODEL_DIR, class_name)
        output_path = os.path.join(OUTPUT_DIR, class_name)
        os.mkdir(output_path)

        pipe = StableDiffusionPipeline.from_pretrained(PRETRAINED_MODEL, torch_dtype=torch.float16).to("cuda:0")
        monkeypatch_or_replace_lora(pipe.unet, torch.load(os.path.join(model_path, "lora_weight.pt")))

        pipe.safety_checker = None

        INFERENCE_PROMPT = 'a sks ' + class_name
        GUIDANCE = 5
        #tune_lora_scale(pipe.unet, LORA_SCALE_UNET)
        for i in range(num_images):
            image = pipe(INFERENCE_PROMPT, num_inference_steps=50, guidance_scale=GUIDANCE).images[0]
            image.save(output_path + "/image" + str(i) + ".png")

def main():
    cifar10_classes = ['airplane' , 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    cifar10_classes = ['deer']
    generate_images(cifar10_classes, 2000)

if __name__ == '__main__':
    main()

