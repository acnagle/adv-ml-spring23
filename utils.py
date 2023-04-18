import os
import random

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision.transforms import ToTensor


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def dummy_filter(images, **kwargs):
    return images, False

def load_image(path, resample=False, size=(224, 224), mode='bicubic', antialias=False):
    img = ToTensor()(Image.open(path))
    if resample:
        img = img = F.interpolate(
            img.view(1, img.shape[0], img.shape[1], img.shape[2]),
            size=size,
            mode=mode,
            antialias=antialias
        )

    return img
