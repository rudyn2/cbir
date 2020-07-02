import glob
import cv2 as cv
from tqdm import tqdm
from features import *
import pickle
import numpy as np
from distances import *
from typing import List
import torch
from collections import defaultdict
import matplotlib.pyplot as plt


class FeatureDB(object):

    def __init__(self, img_db: str):

        self.images = None
        self.img_db_path = img_db
        self.features: dict = {}
        self.feature_extractor: FeatureExtractor = None

    @staticmethod
    def _load_images(img_db: str):
        img_filenames = glob.glob(img_db + '/*.jpg')
        db = {}
        for image_filename in tqdm(img_filenames, "Loading images"):
            image_label = image_filename.split("/")[-1]
            db[image_label] = cv.cvtColor(cv.imread(image_filename, cv.IMREAD_COLOR), cv.COLOR_BGR2RGB)
        return db

    @classmethod
    def load_feature_db(cls, db_path: str):
        with open(db_path, 'rb') as f:
            feature_db = torch.load(f)
        return feature_db

    def load_images(self):
        self.images = self._load_images(self.img_db_path)

    def export_features(self, output_db_name: str, method: FeatureExtractor):
        """
        Exports features to a pickle object.

        :param output_db_name:
            Name of output database with features.
        :param method:
            Method used for feature extraction
        """
        db = self._load_images(self.img_db_path) if not self.images else self.images

        features = {}

        if isinstance(method, HistogramFeatureExtractor):
            for key, img in tqdm(db.items(), "Extracting features"):
                features[key] = method(img)
        elif isinstance(method, CNNFeatureExtractor):

            # groups images
            grouped_img = defaultdict(list)
            for key, img in db.items():
                shape = str(img.shape)
                grouped_img[shape].append((key, img))

            # stacks them into a tensor
            print("Extracting features...")
            for shape, images in grouped_img.items():
                img_list = [item[1] for item in images]
                img_names = [item[0] for item in images]
                img_stacked = np.stack(img_list)
                img_similarities = method(img_stacked)
                features.update(dict(zip(img_names, img_similarities)))

        self.features = features
        self.feature_extractor = method

        print(f"Saving db in {output_db_name}")
        with open(output_db_name, 'wb') as f:
            torch.save(self, f)

        print("Ready!")

    def __len__(self):
        if self.features:
            return len(self.features)
        elif self.images:
            return len(self.images)
        return None


class Ranker(object):

    def __init__(self, feature_db: FeatureDB, img_folder_path: str):
        self.feature_db: FeatureDB = feature_db
        self.img_folder_path = img_folder_path

    @staticmethod
    def extract_class(s: str):
        return int(s.split(".")[0][:3])

    def query(self, image: str, similarity_fn: DistanceMeasure) -> List[np.array]:

        image = cv.cvtColor(cv.imread(self.img_folder_path + '/' + image, cv.IMREAD_COLOR), cv.COLOR_BGR2RGB)
        feature_extractor = self.feature_db.feature_extractor
        if isinstance(feature_extractor, HistogramFeatureExtractor):
            image_feature = feature_extractor(image)
        else:
            # dirty adapter code
            image_feature = feature_extractor(torch.tensor(image).unsqueeze(0))

        similarities = []
        for key, other_feature in tqdm(self.feature_db.features.items(), "Searching in db"):
            img_name, similarity = key, similarity_fn(image_feature, other_feature)
            similarities.append((img_name, similarity))

        return sorted(similarities, key=lambda x: x[1], reverse=True)

    def query_rank(self, image: np.array, image_label: str, similarity_fn: DistanceMeasure, top_k: int = 10):

        query_class = self.extract_class(image_label)
        query_results = self.query(image, similarity_fn)
        top_query_results = query_results[:top_k]
        rank = 0
        total_query_in_class = 0
        for idx, (label, score) in enumerate(top_query_results):
            img_class = self.extract_class(label)
            if img_class == query_class:
                rank += idx + 1
                total_query_in_class += 1

        rank /= total_query_in_class
        return top_query_results, rank, total_query_in_class

    def query_normalized_rank(self, image: np.array, image_label: str, similarity_fn: DistanceMeasure, top_k: int = 10):

        top_query_results, rank, total_query_in_class = self.query_rank(image, image_label, similarity_fn, top_k)
        norm_rank = (rank - (total_query_in_class + 1) / 2) / len(self.feature_db)

        return top_query_results, norm_rank


class Visualizer(object):

    def __init__(self, img_folder_path: str):
        self.img_folder_path = img_folder_path

    def read_img(self, img_name: str):

        img_ = cv.cvtColor(cv.imread(self.img_folder_path + '/' + img_name, cv.IMREAD_COLOR), cv.COLOR_BGR2RGB)
        return img_

    def plot_image(self, img_name: str):

        img_ = self.read_img(img_name)
        plt.imshow(img_)
        plt.axis('off')
        plt.title(img_name)
        plt.show()

    def plot_k_best(self, query_img: str, img_results: list, k_best: int = 10):

        assert k_best <= 10 and k_best % 2 == 0, "Number of images to plot has to be an even number less than 10"
        top_k = img_results[:k_best]

        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(2, ncols=int(k_best/2) + 1)

        idx = 0
        for i in range(2):
            for j in range(1, int(k_best/2) + 1):
                ax = fig.add_subplot(gs[i, j])
                img_label_ = top_k[idx][0]
                similarity = top_k[idx][1]
                img_ = self.read_img(img_label_)
                ax.set(title=f"{similarity:.4f}")
                ax.axis('off')
                ax.imshow(img_)

                idx += 1

        query_ax = fig.add_subplot(gs[:, 0])
        query_ax.imshow(self.read_img(query_img))
        query_ax.set_title('Query image')
        query_ax.axis('off')
        plt.tight_layout()

        plt.show()

    def plot_query_best_3(self, query_img: List[str], img_results: List[List]):

        assert len(query_img) == len(img_results)
        assert len(query_img) <= 5

        fig, axs = plt.subplots(nrows=4, ncols=len(query_img), figsize=(10, 16))
        for i in range(4):
            for j in range(len(query_img)):
                # first row: query images
                ax = axs[i, j]
                if i == 0:
                    img_ = self.read_img(query_img[j])
                    ax.set(title="Query image")
                else:
                    img_ = self.read_img(img_results[j][i-1][0])
                    ax.set(title=f"{img_results[j][i-1][1]:.4f}")
                ax.axis('off')
                ax.imshow(img_)
        plt.show()


if __name__ == '__main__':
    img_path = 'data/jpg'
    db_path = 'data/dbs/method2features'
    img_1, img_2, img_3 = '100000.jpg', '101000.jpg', '102000.jpg'

    f = FeatureDB.load_feature_db(db_path=db_path)
    r = Ranker(feature_db=f, img_folder_path=img_path)
    d = CosineDistance()

    results1, results2, results3 = r.query(img_1, d), r.query(img_2, d), r.query(img_3, d)

    v = Visualizer(img_path)

    # v.plot_k_best(img_1, results1)
    # v.plot_image('108500.jpg')
    v.plot_query_best_3(query_img=[img_1, img_2, img_3], img_results=[results1, results2, results3])


