from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler
from lora_diffusion import tune_lora_scale, patch_pipe, image_grid
import torch

import os
import argparse
import sys
sys.path.append('../')

from utils import seed_everything, dummy_filter

# Define constants
model_id = 'runwayml/stable-diffusion-v1-5'
exp_path = './exps-800'
num_repeats = 1000
rank = 16
seed = 42

classes = (1, 2, 3, 4, 5, 6, 7, 8, 10, 11)
prompts = ['A photo of an {}']

parser = argparse.ArgumentParser()
parser.add_argument('--overwrite', action='store_true', help='Whether to overwrite existing files')
args = parser.parse_args()


def main():
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to('cuda:0')
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.safety_checker = dummy_filter

    for c in classes:
        unq_token = 'sks'
        safetensor_path = os.path.join(exp_path, f'rank-{rank}', str(c), 'final_lora.safetensors')

        patch_pipe(
            pipe,
            safetensor_path,
            patch_text=True,
            patch_ti=True,
            patch_unet=True,
        )

        #for s_idx in range(len(lora_scale)):
        seed_everything(seed)   # reset the seed whe generating images for each subject (and for each scale for that subject)
        #tune_lora_scale(pipe.unet, lora_scale[s_idx])
        #tune_lora_scale(pipe.text_encoder, lora_scale[s_idx])

        #save_dir = os.path.join(exp_path, f'rank-{rank}', str(c), 'inference', f'scale{lora_scale[s_idx]}')
        save_dir = os.path.join(exp_path, f'rank-{rank}', str(c), 'inference')
        dir_exists = False
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            print(f'Directory {save_dir} created')
        else:
            print(f'Directory {save_dir} already exists')
            dir_exists = True
            if args.overwrite:
                print(f'Overwriting {save_dir}')

        if (not dir_exists) or (dir_exists and args.overwrite):
            imgs = []
            for p_idx in range(len(prompts)):
                for r in range(num_repeats):
                    prompt = prompts[p_idx].format(unq_token)
                    img = pipe(prompt, num_inference_steps=100, guidance_scale=7).images[0]
                    img.save(os.path.join(save_dir, f'bmw{c}_prompt{p_idx}{r}.png'))
#                            imgs.append(img)
#
#                        imgs = image_grid(imgs, 5, 5)
#                        imgs.save(os.path.join(save_dir, f'{concept_name}_grid.png'))


if __name__ == '__main__':
    main()
