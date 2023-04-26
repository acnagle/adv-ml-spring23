from diffusers import StableDiffusionPipeline
from lora_diffusion import image_grid
import torch

import os
import argparse
import sys
sys.path.append('../')

from utils import seed_everything, dummy_filter

# Define constants
model_id = 'runwayml/stable-diffusion-v1-5'
exp_path = './exps'
num_repeats = 50     # number of images to generate with each prompt
rank = 16
seed = 42

classes = (1, 2, 3, 4, 5, 6, 7, 8, 10, 11)
prompts = ['A photo of an {}']

parser = argparse.ArgumentParser()
parser.add_argument('--overwrite', action='store_true', help='Whether to overwrite existing files')
args = parser.parse_args()


def main():
    for c in classes:
        unq_token = 'sks'
        lora_path = os.path.join(exp_path, f'rank-{rank}', str(c))

        pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to('cuda')
        pipe.unet.load_attn_procs(lora_path)
        pipe.safety_checker = dummy_filter
        seed_everything(seed)

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


if __name__ == '__main__':
    main()
