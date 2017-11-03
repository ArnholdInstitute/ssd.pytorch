from .voc0712 import VOCDetection, AnnotationTransform, detection_collate, VOC_CLASSES
from .config import *
import cv2
import numpy as np


def base_transform(image, size, mean):
    x = cv2.resize(image, (size, size)).astype(np.float32)
    # x = cv2.resize(np.array(image), (size, size)).astype(np.float32)
    x -= mean
    x = x.astype(np.float32)
    return x


class BaseTransform:
    def __init__(self, size, mean):
        self.size = size
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self, image, boxes=None, labels=None):
        new_img = base_transform(image, self.size, self.mean)
        if boxes is not None:
            boxes = boxes.astype('float32')
            boxes[:, (0, 2)] /= image.shape[1]
            boxes[:, (1, 3)] /= image.shape[0]

        return new_img, boxes, labels
