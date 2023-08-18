import json
import os.path
import shutil
from pathlib import Path
import random

import albumentations as a
import cv2
import numpy as np

annotation_directory = Path("D:/pycharmprojects/segmentation/dataset/train/images")
labels_directory = Path("D:/pycharmprojects/segmentation/dataset/train/labels")
annotation_file_path = Path("D:/pycharmprojects/segmentation/dataset/rect_gauge_coco.json")
post_process_directory = Path("D:/pycharmprojects/segmentation/dataset/output/images")
labels_post_process_directory = Path("D:/pycharmprojects/segmentation/dataset/output/labels")


class DataAugmentation:
    """
    Handles with various augmentations for dataset.
    """

    def __init__(self):
        pass

        # self.pool.apply_async(self.run_augmentations,
        #                       (annotation_directory, post_process_directory, filename, each_file))

    def brightness(self, img, low, high):
        value = random.uniform(low, high)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv = np.array(hsv, dtype=np.float64)
        hsv[:, :, 1] = hsv[:, :, 1] * value
        hsv[:, :, 1][hsv[:, :, 1] > 255] = 255
        hsv[:, :, 2] = hsv[:, :, 2] * value
        hsv[:, :, 2][hsv[:, :, 2] > 255] = 255
        hsv = np.array(hsv, dtype=np.uint8)
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return img

    def brightness_augmentation(self, image, low=0.5, high=1.5):
        brightness_factor = np.random.uniform(low, high)
        augmented_image = image * brightness_factor
        augmented_image = np.clip(augmented_image, 0, 255).astype(np.uint8)
        return augmented_image

    def contrast_augmentation(self, image, low=0.5, high=1.5):
        contrast_factor = np.random.uniform(low, high)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_image = np.clip(contrast_factor * gray_image, 0, 255).astype(np.uint8)
        augmented_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
        return augmented_image

    def gaussian_noise_augmentation(self, image, mean=0, std=25):
        noise = np.random.normal(mean, std, image.shape).astype(np.uint8)
        augmented_image = cv2.add(image, noise)
        augmented_image = np.clip(augmented_image, 0, 255).astype(np.uint8)
        return augmented_image

    def blur_augmentation(self, image, kernel_size=5):
        augmented_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        return augmented_image

    def process(self, annotation_directory, post_process_directory, labels_directory, labels_post_process_directory):
        assert os.path.exists(annotation_directory)
        if not os.path.exists(post_process_directory):
            os.mkdir(post_process_directory)
        for each_file, label_file in zip(os.listdir(annotation_directory), os.listdir(labels_directory)):
            filename, file_extension = os.path.splitext(each_file)
            label_filename, label_file_extension = os.path.splitext(label_file)
            print(filename, file_extension)
            print(label_filename, label_file_extension)

            if filename == label_filename:
                # if file_extension in ['.jpg', '.jpeg', '.png']:
                image = cv2.imread(os.path.join(annotation_directory, each_file))
                multi_images = (
                    self.blur_augmentation(image), self.gaussian_noise_augmentation(image),
                    self.brightness(image, 0.5, 3),
                    self.contrast_augmentation(image), self.brightness_augmentation(image))

                _file_name = 0
                for each_element in multi_images:
                    image = each_element

                    cv2.imwrite(
                        os.path.join(post_process_directory,
                                     f"{each_file[:-4]}" + "_" + f"{_file_name}" + f"{file_extension}"),
                        image)
                    shutil.copy(os.path.join(labels_directory, label_file),
                                os.path.join(labels_post_process_directory,
                                             f"{label_file[:-4]}" + "_" + f"{_file_name}" + f"{label_file_extension}"))

                    _file_name = _file_name + 1

    def combine_dataset(self, annotation_directory, post_process_directory, labels_directory,
                        labels_post_process_directory):
        for each_file, label_file in zip(os.listdir(annotation_directory), os.listdir(labels_directory)):
            shutil.copy(os.path.join(annotation_directory, each_file), post_process_directory)
            shutil.copy(os.path.join(labels_directory, label_file), labels_post_process_directory)


obj = DataAugmentation()
obj.process(annotation_directory, post_process_directory, labels_directory, labels_post_process_directory)
obj.combine_dataset(annotation_directory, post_process_directory, labels_directory, labels_post_process_directory)
