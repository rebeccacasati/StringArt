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