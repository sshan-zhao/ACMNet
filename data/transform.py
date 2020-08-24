import collections
import math
import numbers
import random

import numpy as np
import torch
from PIL import Image
import torchvision.transforms as transforms
random.seed(0)

class RandomImgAugment(object):
    """Randomly shift gamma"""

    def __init__(self, no_flip, no_augment, interpolation=Image.BILINEAR):

        self.flip = not no_flip
        self.augment = not no_augment
        self.interpolation = interpolation

    def __call__(self, inputs):

        img = inputs[0]
        gt_depth = inputs[1]
        s_depth = inputs[2]
        dataset = inputs[3]

        if self.augment:
            if random.random() < 0.5:	
                img = Image.fromarray(img)
                color_transform = transforms.Compose([transforms.ColorJitter(brightness=0.2,contrast=0.2,saturation=0.2,hue=0.0)])
                img = color_transform(img)
                img = np.array(img)

        if self.flip:
            flip_prob = random.random()
        else:
            flip_prob = 0.0

        if img is not None:
            img = img.astype(np.float32)
            if flip_prob >= 0.5:
                img = img[:, ::-1, :]
            img = img / 255.0
            img = np.transpose(img, (2, 0, 1))	

        if flip_prob >= 0.5:
            s_depth = s_depth[:, ::-1]
            if gt_depth is not None:
                gt_depth = gt_depth[:, ::-1]

        s_depth = np.expand_dims(s_depth, axis=0)
        s_depth = s_depth.astype(np.float32) / 256.0
        if gt_depth is not None:
            gt_depth = np.expand_dims(gt_depth, axis=0)
            gt_depth = gt_depth.astype(np.float32) / 256.0      


        return torch.tensor(img.copy()), torch.tensor(gt_depth.copy()), torch.tensor(s_depth.copy())
