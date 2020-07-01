import numpy as np
from abc import abstractmethod
from typing import Tuple, List
import matplotlib.pyplot as plt
import cv2 as cv
import torchvision.models as models
import torch
from torchvision import transforms


class FeatureExtractor(object):

    @abstractmethod
    def __call__(self, *args, **kwargs):
        raise NotImplementedError


class HistogramFeatureExtractor(FeatureExtractor):

    @classmethod
    def to_hsv(cls, input_img: np.array) -> np.array:
        return cv.cvtColor(input_img, cv.COLOR_RGB2HSV)

    @abstractmethod
    def divide(self, hsv_img: np.array):
        raise NotImplementedError

    @classmethod
    def normalize_hist(cls, hist: np.array):
        # TODO: Check if this is the right way of normalizing an histogram
        bin_length = 256/len(hist)              # class width
        total = np.sum(hist)                    # number of observations
        for i in range(len(hist)):
            hist[i] /= bin_length*total
        return hist

    def __call__(self, input_img: np.array):
        """
        Calculates the feature vector of an RGB input image.

        :param input_img:
            Input image as numpy array.
        :return:
            Feature vector as numpy array.
        """
        hsv = self.to_hsv(input_img)
        feats = []
        masks = self.divide(input_img[:, :, 0])
        for mask in masks:
            h_hist_ = self.normalize_hist(cv.calcHist([hsv], [0], mask, [8], [0, 256]))
            s_hist_ = self.normalize_hist(cv.calcHist([hsv], [1], mask, [12], [0, 256]))
            v_hist_ = self.normalize_hist(cv.calcHist([hsv], [2], mask, [3], [0, 256]))
            stacked_hist = np.vstack([h_hist_, s_hist_, v_hist_])
            feats.extend(stacked_hist)
        return np.vstack(feats)


class Method1Extractor(HistogramFeatureExtractor):

    def __init__(self):
        super(HistogramFeatureExtractor, self).__init__()
        self.patch_generator = SquareImageGenerator(n_regions=3)

    def divide(self, image: np.array):
        return self.patch_generator(image)


class Method2Extractor(HistogramFeatureExtractor):
    def __init__(self):
        super(HistogramFeatureExtractor, self).__init__()
        self.patch_generator = SquareImageGenerator(n_regions=4)

    def divide(self, image: np.array):
        return self.patch_generator(image)


class Method3Extractor(HistogramFeatureExtractor):
    def __init__(self):
        super(HistogramFeatureExtractor, self).__init__()
        self.patch_generator = SquareImageGenerator(n_regions=6)

    def divide(self, image: np.array):
        return self.patch_generator(image)


class Method4Extractor(HistogramFeatureExtractor):
    def __init__(self):
        super(HistogramFeatureExtractor, self).__init__()
        self.patch_generator = CircularPatchGenerator(n_regions=2)

    def divide(self, image: np.array):
        return self.patch_generator(image)


class ImagePatchGenerator(object):

    def __init__(self, n_regions: int):
        self.n_regions = n_regions

    @abstractmethod
    def __call__(self, *args, **kwargs):
        raise NotImplementedError


class SquareImageGenerator(ImagePatchGenerator):
    def __init__(self, n_regions: int):
        super(SquareImageGenerator, self).__init__(n_regions)

    def __call__(self, image: np.array) -> List[np.array]:
        """
        Divides a 2D-image in patches and returns the mask of each patch in a list.
        :param image:
            Image that will be divided.
        :return:
            List of mask patches
        """
        assert len(image.shape) == 2, "The input image has more than 2 channels."
        width = image.shape[0]
        height = image.shape[1]
        i_step, j_step = int(width/self.n_regions), int(height/self.n_regions)
        i_idxs = [i_step*n for n in range(self.n_regions+1)]
        j_idxs = [j_step*n for n in range(self.n_regions+1)]
        masks = []
        for i_idx in range(self.n_regions):
            j_blocks = []
            actual_i, next_i = i_idxs[i_idx], i_idxs[i_idx + 1]

            if i_idx == self.n_regions - 1 and next_i != width:
                next_i = width

            for j_idx in range(self.n_regions):
                actual_j, next_j = j_idxs[j_idx], j_idxs[j_idx + 1]
                # check non-divisible heights
                if j_idx == self.n_regions - 1 and next_j != height:
                    next_j = height

                # extracts mask
                new_mask = np.zeros(shape=(width, height))
                new_mask[actual_i:next_i, actual_j:next_j] = 1
                j_blocks.append(new_mask.astype(np.uint8))

            masks.extend(j_blocks)
        return masks


class CircularPatchGenerator(ImagePatchGenerator):
    def __init__(self, n_regions: int):
        super(CircularPatchGenerator, self).__init__(n_regions)
        self.square_patch_gen = SquareImageGenerator(n_regions)

    @staticmethod
    def is_inside_circle(pix_x: int, pix_y: int, center_x: int, center_y: int, radius: int):
        if (pix_x - center_x)**2 + (pix_y - center_y)**2 <= radius**2:
            return 1
        return 0

    def __call__(self, image: np.array) -> List[np.array]:
        """
        Divides a 2D-image in a big circular patch and external patches and then returns
        the mask of each patch in a list.
        :param image:
            Image that will be divided.
        :return:
            List of mask patches
        """
        assert len(image.shape) == 2, "The input image has more than 2 channels."
        width = image.shape[0]
        height = image.shape[1]
        radius = int(min(width, height)*0.42)
        center_i = int(width/2)
        center_j = int(height/2)
        circular_mask = np.zeros(shape=image.shape)
        for i in range(width):
            for j in range(height):
                circular_mask[i, j] = self.is_inside_circle(i, j, center_i, center_j, radius)
        circular_mask = circular_mask.astype(np.uint8)

        square_masks = self.square_patch_gen(image)

        result = []
        for mask in square_masks:
            mask[circular_mask] = 1
            result.append(mask)
        result.append(circular_mask)
        return result


class CNNFeatureExtractor(FeatureExtractor):

    NORMALIZER = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])

    def prepare_image(self, images: np.array):

        assert images.dtype == np.uint8
        assert len(images.shape) == 4, "It has to be a Tensor in NCWH format."
        assert images.shape[1] == 3 or images.shape[3] == 3, "It needs to be a RGB Image"

        if images.shape[3] == 3:
            images = images.transpose(0, 3, 1, 2)

        # Applies transformations required by the model
        images = images.astype(np.float32) / 255
        images = torch.tensor(images)
        for idx in range(images.shape[0]):
            images[idx, :, :, :] = self.NORMALIZER(images[idx, :, :, :])
        images = images.float()

        return images

    def __call__(self, *args, **kwargs):
        raise NotImplementedError


class VGG16FeatureExtractor(CNNFeatureExtractor):

    def __init__(self, weight_dir: str):
        self.vgg16 = models.vgg16(pretrained=False)
        weights = torch.load(weight_dir)
        self.vgg16.load_state_dict(weights)
        self.vgg16.eval()

    def __call__(self, images: np.array):
        """
        Calculates the features of a tensor of images.

        :param images:
            Torch Tensor in NCHW or NHWC format.
        :return:
            Features in a numpy array.
        """
        image = self.prepare_image(images)
        feats = self.vgg16.avgpool(self.vgg16.features(image))
        feats = feats.flatten(1)
        feats = feats.detach().cpu().numpy()
        return [np.reshape(feats[idx, :], (feats.shape[1], 1)) for idx in range(feats.shape[0])]


if __name__ == '__main__':
    weight_dir = 'data/model_weights/vgg16-397923af.pth'
    single_img = cv.imread('data/jpg/100101.jpg', cv.IMREAD_COLOR)
    single_img = cv.cvtColor(single_img, cv.COLOR_BGR2RGB)
    single_img = single_img.astype(np.float) / 255

    example = np.random.randint(low=0, high=255, size=(10, 1024, 768, 3), dtype=np.uint8)
    f1 = VGG16FeatureExtractor(weight_dir=weight_dir)
    features = f1(example)

    plt.imshow(single_img)
    plt.show()


