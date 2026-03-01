import cv2
"""
    Detect if an image is blur using the Laplacian variance method.

    Agrs:
        image (numpy.ndarray): The input image in grayscale format.
        threshold (float = 100.0): The default variance threshold which determines if the image is blur or not.
    Returns:
        bool: True if the image is blur, False otherwise.
        float: The actual variance of the Laplacian of the image.
"""

def is_blur(grayscale_image, threshold = 50) -> tuple[float, bool, float]:
    #Calculate the Laplacian of the image and then return the variance
    laplacian = cv2.Laplacian(grayscale_image, cv2.CV_64F)
    variance = laplacian.var()

    return laplacian, variance < threshold, variance
