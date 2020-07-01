from abc import abstractmethod
import numpy as np


class DistanceMeasure(object):

    @abstractmethod
    def __call__(self, *args, **kwargs):
        raise NotImplementedError


class ChiSquareDistance(DistanceMeasure):

    def __call__(self, x: np.array, y: np.array) -> np.float:
        assert x.shape == y.shape, "x and y must be equal"
        return 0.5 * np.sum(np.divide((x-y)**2, x+y))


class EuclideanDistance(DistanceMeasure):

    def __call__(self, x: np.array, y: np.array) -> np.float:
        assert x.shape == y.shape, "x and y must be equal"
        return np.sqrt(np.sum((x-y)**2))


class CosineDistance(DistanceMeasure):

    def __call__(self, x: np.array, y: np.array) -> np.float:
        assert x.shape == y.shape, "x and y must be equal"
        return np.sum(x*y) / (np.sqrt(np.sum(x**2)) * np.sqrt(np.sum(y**2)))

