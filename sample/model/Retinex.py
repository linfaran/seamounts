import numpy as np
import cv2 as cv

class RetinexProcessor:
    def __init__(self, image_path):
        self.image_path = image_path
        self.img = self._read_image()

    def _read_image(self):
        """
        Reads the image from the specified path.
        """
        img = cv.imread(self.image_path, 1)
        if img is None:
            raise ValueError("Image not found or invalid image path.")
        return np.float32(img) / 255.0

    def single_scale_retinex(self, sigma):
        """
        Single Scale Retinex.

        :param sigma: Gaussian kernel standard deviation
        :return: SSR result
        """
        temp = cv.GaussianBlur(self.img, (0, 0), sigma)
        gaussian = np.where(temp == 0, 0.01, temp)
        retinex = np.log10(self.img + 0.01) - np.log10(gaussian)
        return retinex

    def multi_scale_retinex(self, sigma_list):
        """
        Multi Scale Retinex.

        :param sigma_list: List of standard deviations for Gaussian kernel
        :return: MSR result
        """
        retinex = np.zeros_like(self.img)
        for sigma in sigma_list:
            retinex += self.single_scale_retinex(sigma)

        # Average
        retinex = retinex / len(sigma_list)
        return retinex

    def display_results(self, retinex_result, title):
        """
        Displays the original and processed images.

        :param retinex_result: The result of Retinex algorithm
        :param title: Title for the processed image window
        """
        retinex_result = cv.normalize(retinex_result, None, 0, 255, cv.NORM_MINMAX)
        retinex_result = np.uint8(retinex_result)

        cv.imshow('Original Image', self.img)
        cv.imshow(title, retinex_result)
        cv.imwrite('OriginalImage.jpg', retinex_result)
# Example usage:
# processor = RetinexProcessor('path_to_your_image.jpg')
# ssr_result = processor.single_scale_retinex(sigma=300)
# processor.display_results(ssr_result, 'Single Scale Retinex')
# msr_result = processor.multi_scale_retinex(sigma_list=[15, 80, 250])
# processor.display_results(msr_result, 'Multi Scale Retinex')