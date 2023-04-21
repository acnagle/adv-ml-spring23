
from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler
import torch
import os

model_id = "runwayml/stable-diffusion-v1-5"

pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda:0")

pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

torch.manual_seed(0)

cifar10_classes = ['airplane' , 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

output_dir = '/home/disha/Documents/lora/contents/cifar10/train/'

for i in range(4500):
    for prompt in cifar10_classes:
        image = pipe(prompt, num_inference_steps=50, guidance_scale=7).images[0]
        image.save(output_dir + prompt + "/image" +str(500+i) + ".jpeg")
