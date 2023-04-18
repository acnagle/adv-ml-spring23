from diffusers import StableDiffusionPipeline
from lora_diffusion import image_grid
import torch

import os
import argparse
import sys
sys.path.append('../')

from prompts_and_classes import concept_dict, prompt_dict
from utils import seed_everything, dummy_filter

# Define constants
exp_path = './exps'
seed = 42

parser = argparse.ArgumentParser()
parser.add_argument('--overwrite', action='store_true', help='Whether to overwrite existing files')
args = parser.parse_args()


def main():
    for c_type in concept_dict.keys():
        for c_idx in range(len(concept_dict[c_type])):
            concept_name = concept_dict[c_type][c_idx][0]
            class_name = concept_dict[c_type][c_idx][1]
            unq_token = f'<{concept_name}>'
            model_id = os.path.join(exp_path, concept_name)

            pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to('cuda')
            pipe.safety_checker = dummy_filter
            seed_everything(seed)

            save_dir = os.path.join(exp_path, concept_name, 'inference')
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
                for p_idx in range(len(prompt_dict[c_type])):
                    prompt = prompt_dict[c_type][p_idx].format(unq_token, class_name)
                    img = pipe(prompt, num_inference_steps=50, guidance_scale=7).images[0]
                    img.save(os.path.join(save_dir, f'{concept_name}_prompt{p_idx}.png'))
                    imgs.append(img)

                imgs = image_grid(imgs, 5, 5)
                imgs.save(os.path.join(save_dir, f'{concept_name}_grid.png'))


if __name__ == '__main__':
    main()
