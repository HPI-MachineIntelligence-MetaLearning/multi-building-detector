import copy
import numpy as np
import random
import matplotlib

from chainercv import transforms
from chainercv.links.model.ssd import random_crop_with_bbox_constraints, \
    resize_with_random_interpolation


def random_distort(
        img,
        brightness_delta=32,
        contrast_low=0.5, contrast_high=1.5):

    cv_img = img[::-1].transpose((1, 2, 0)).astype(np.uint8)

    def convert(img, alpha=1, beta=0):
        img = img.astype(float) * alpha + beta
        img[img < 0] = 0
        img[img > 255] = 255
        return img.astype(np.uint8)

    def brightness(cv_img, delta):
        if random.randrange(2):
            return convert(
                cv_img,
                beta=random.uniform(-delta, delta))
        else:
            return cv_img

    def contrast(cv_img, low, high):
        if random.randrange(2):
            return convert(
                cv_img,
                alpha=random.uniform(low, high))
        else:
            return cv_img

    cv_img = brightness(cv_img, brightness_delta)

    if random.randrange(2):
        cv_img = contrast(cv_img, contrast_low, contrast_high)
    else:
        cv_img = contrast(cv_img, contrast_low, contrast_high)

    return cv_img.astype(np.float32).transpose((2, 0, 1))[::-1]


class ImageAugmentation:

    def __init__(self, coder, size, mean):
        # to send cpu, make a copy
        self.coder = copy.copy(coder)
        self.coder.to_cpu()

        self.size = size
        self.mean = mean

    def __call__(self, in_data):
        # There are five data augmentation steps
        # 1. Color augmentation
        # 2. Random expansion
        # 3. Random cropping
        # 4. Resizing with random interpolation
        # 5. Random horizontal flipping

        img, bbox, label = in_data

        # 1. Color augmentation
        img = random_distort(img)

        if len(bbox[0]) == 0:
            bbox = np.empty((0, 4))
            label = np.empty((0, 4))

        # 2. Random expansion
        if np.random.randint(2):
            img, param = transforms.random_expand(
                img, fill=self.mean, return_param=True)
            bbox = transforms.translate_bbox(
                bbox, y_offset=param['y_offset'], x_offset=param['x_offset'])

        # 3. Random cropping
        img, param = random_crop_with_bbox_constraints(
            img, bbox, return_param=True)
        bbox, param = transforms.crop_bbox(
            bbox, y_slice=param['y_slice'], x_slice=param['x_slice'],
            allow_outside_center=False, return_param=True)
        label = label[param['index']]

        # 4. Resizing with random interpolatation
        _, H, W = img.shape
        img = resize_with_random_interpolation(img, (self.size, self.size))
        bbox = transforms.resize_bbox(bbox, (H, W), (self.size, self.size))

        # 5. Random horizontal flipping
        img, params = transforms.random_flip(
            img, x_random=True, return_param=True)
        bbox = transforms.flip_bbox(
            bbox, (self.size, self.size), x_flip=params['x_flip'])

        # Preparation for SSD network
        img -= self.mean
        mb_loc, mb_label = self.coder.encode(bbox, label)

        return img, mb_loc, mb_label
