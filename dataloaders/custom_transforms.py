# Code adapted from https://github.com/sail-sg/EditAnything/blob/main/utils/transforms.py
import torch
import cv2
import numpy as np
from typing import List, Tuple

class RandomHorizontalFlip:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, img_rgb, img_conditions):
        if np.random.random() < self.prob:
            return np.fliplr(img_rgb), np.fliplr(img_conditions)
        return img_rgb, img_conditions


class RandomResizeCrop:
    def __init__(self, size, scale=(0.08, 1.0), fit_to_new_size=False):
        self.size = size
        self.fit_to_new_size = fit_to_new_size
        if fit_to_new_size:
            # Adjust scale to 1.0 so we don't have images < size
            self.scale = (1.0, scale[1])
        else:
            self.scale = scale

    def scale_min_image_dim(self, img_rgb, img_conditions, size):
        height, width = img_rgb.shape[:2]
        min_dim = min(height, width)
        scale = size / min_dim
        with_scale = int(np.ceil(width*scale))
        height_scale = int(np.ceil(height*scale))
        img_rgb = cv2.resize(img_rgb, (with_scale, height_scale))
        img_conditions = cv2.resize(img_conditions, (with_scale, height_scale))

        return img_rgb, img_conditions

    def __call__(self, img_rgb, img_conditions):
        if self.fit_to_new_size:
            img_rgb, img_conditions = self.scale_min_image_dim(img_rgb, img_conditions, self.size)
        # random scale
        height, width = img_rgb.shape[:2]
        scale = np.random.uniform(self.scale[0], self.scale[1])

        with_scale = int(width*scale)
        height_scale = int(height*scale)
        img_rgb = cv2.resize(img_rgb, (with_scale, height_scale))
        img_conditions = cv2.resize(img_conditions, (with_scale, height_scale))

        # Crop
        height, width = img_rgb.shape[:2]
        x = np.random.randint(0, width - self.size + 1)
        y = np.random.randint(0, height - self.size + 1)

        img_rgb = img_rgb[y:y + self.size, x:x + self.size]
        img_conditions = img_conditions[y:y + self.size, x:x + self.size]

        return img_rgb, img_conditions


class ToTensor:
    def __call__(self, img_rgb: np.ndarray, img_conditions: np.ndarray):
        # img_rgb = img_rgb.transpose((2, 0, 1)).copy()
        # img_conditions = img_conditions.transpose((2, 0, 1)).copy()
        # return torch.from_numpy(img_rgb), torch.from_numpy(img_conditions)
        return img_rgb.copy(), img_conditions.copy()

class Compose(object):
    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, img_rgb, img_conditions):
        for t in self.transforms:
            img_rgb, img_conditions = t(img_rgb, img_conditions)

        return img_rgb, img_conditions