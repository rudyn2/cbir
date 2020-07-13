from abc import abstractmethod
import numpy as np


class DistanceMeasure(object):

    @abstractmethod
    def __call__(self, *args, **kwargs):
        raise NotImplementedError


class ChiSquareSimilarity(DistanceMeasure):

    def __call__(self, x: np.array, y: np.array) -> np.float:
        assert x.shape == y.shape, "x and y must be equal"
        distance = 0.5 * np.sum(np.divide((x-y)**2, x+y))
        return 1 / (1 + distance)


class EuclideanSimilarity(DistanceMeasure):

    def __call__(self, x: np.array, y: np.array) -> np.float:
        assert x.shape == y.shape, "x and y must be equal"
        distance = np.sqrt(np.sum((x-y)**2))
        return 1 / (1 + distance)


class CosineSimilarity(DistanceMeasure):

    def __call__(self, x: np.array, y: np.array) -> np.float:
        assert x.shape == y.shape, "x and y must be equal"
        return np.sum(x*y) / (np.sqrt(np.sum(x**2)) * np.sqrt(np.sum(y**2)))

