import cv2
import numpy as np


def image_to_vector(image_path, target_size=(512, 512)):
    """
    Loads a grayscale image, resizes it, and flattens it into a 1D vector.
    Returns a flattened image vector of size m
    """
    # Load image in grayscale:
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, target_size)
    # Normalize pixel values to range [0, 1] and flatten the matrix into a vector:
    img = img.astype(np.float32) / 255.0
    b = img.flatten()
    return b



"""
    def matrix_D(self, v):

        h, w = self.image_size*self.s, self.image_size*self.s
        assert h % self.s == 0 and w % self.s == 0

        new_h, new_w = h // self.s, w // self.s
        small_matrix = v.reshape(new_h, self.s, new_w, self.s).mean(axis=(1, 3))
        return small_matrix.reshape(new_h*new_w,)
"""