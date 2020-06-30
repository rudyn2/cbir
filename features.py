import numpy as np
from abc import abstractmethod
from typing import Tuple
import matplotlib.pyplot as plt
import cv2 as cv


class FeatureExtractor(object):

    def __init__(self):
        pass

    @classmethod
    def to_hsv(cls, input_img: np.array) -> np.array:
        return cv.cvtColor(input_img, cv.COLOR_BGR2HSV)

    @abstractmethod
    def divide(self, hsv_img: np.array):
        raise NotImplementedError

    @classmethod
    def generate_hists(cls, hsv_cell: np.array) -> Tuple[np.array, np.array, np.array]:
        raise NotImplementedError

    @classmethod
    def normalize_hist(cls, hist: np.array):
        raise NotImplementedError

    @classmethod
    def concat_hists(cls):
        raise NotImplementedError

    def __call__(self, input_img: np.array):
        """
        Calculates the feature vector of an input image.

        :param input_img:
            Input image as numpy array.
        :return:
            Feature vector as numpy array.
        """
        hsv_img = self.to_hsv(input_img)
        cells = self.divide(hsv_img)
        features = []
        for cell in cells:
            h_hist, s_hist, v_hist = self.generate_hists(cell)
            h_hist_ = self.normalize_hist(h_hist)
            s_hist_ = self.normalize_hist(s_hist)
            v_hist_ = self.normalize_hist(v_hist)
            features.append(np.stack([h_hist_, s_hist_, v_hist_]))
        return np.stack(features)


class Method1Extractor(FeatureExtractor):

    def __init__(self):
        super(FeatureExtractor, self).__init__()

    def divide(self, hsv_img: np.array):
        pass


if __name__ == '__main__':
    example = np.random.randint(low=0, high=255, size=(1024, 768))
    hsv_img = FeatureExtractor.to_hsv(example)
    plt.imshow(example, cmap='gray')
    plt.show()




