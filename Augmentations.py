import cv2
from albumentations import (Blur)
import random
import numpy as np

class DualCompose:
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, x):
        for t in self.transforms:
            x = t.apply(x[0], mask = x[1], regr = x[2])
            
        return x

class HorizontalFlip:
    def __init__(self, prob=0.5):
        self.prob = prob

    def apply(self, img, mask, regr):
        if random.random() < self.prob:
            img = cv2.flip(img, 1)
            mask = cv2.flip(mask, 1)
            regr = cv2.flip(regr, 1)

        return (img, mask, regr)


class VerticalFlip:
    def __init__(self, prob=0.5):
        self.prob = prob

    def apply(self, img, mask, regr):
        if random.random() < self.prob:
            img = cv2.flip(img, 0)
            mask = cv2.flip(mask, 0)
            regr = cv2.flip(regr, 0)
        
        return (img, mask, regr)

class RandomGamma:
    def __init__(self, limit=0.25, prob=0.5):
        self.limit = limit
        self.prob = prob

    def apply(self, img, mask, regr):
        if random.random() < self.prob:
            gamma = 1.0 + self.limit * random.uniform(-1, 1)

            img = img ** (1.0 / gamma)
            img = np.clip(img, 0, 1)

        return (img, mask, regr)

class RandomBrightnessContrast:
    def __init__(self, limit_alpha=0.3, limit_beta=0.3, prob=.1):
        self.limit_alpha = limit_alpha
        self.limit_beta = limit_beta
        self.prob = prob

    def apply(self, img, mask, regr, alpha=1., beta=0.):
        if random.random() < self.prob:
            alpha = 1.0 + random.uniform(-self.limit_alpha, self.limit_alpha)
        if random.random() < self.prob:
            beta  = 0.  + random.uniform(-self.limit_beta, self.limit_beta)

            img = np.clip(alpha * img + beta * np.mean(img), 0, 1)
        return (img, mask, regr)

class RBlur:
    def __init__(self, prob=0.5):
        self.prob = prob
        self.blur = Blur(blur_limit=7, p=1)

    def apply(self, img, mask, regr):
        
        if random.random() < self.prob:
            img = self.blur.apply(img)
        
        return (img, mask, regr)

train_transforms = DualCompose([
    RandomBrightnessContrast(prob=0.35),
    RandomGamma(prob=0.35),
    RBlur(prob=0.2)
])
valid_transforms = None
test_transforms = None