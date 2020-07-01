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
        for image_filename in tqdm(img_filenames[:5], "Loading images"):
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
            for shape, images in tqdm(grouped_img.items(), "Extracting features"):
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

    def __init__(self, feature_db: FeatureDB):
        self.feature_db: FeatureDB = feature_db

    @staticmethod
    def extract_class(s: str):
        return int(s.split(".")[0][:3])

    def query(self, image: np.array, similarity_fn: DistanceMeasure) -> List[np.array]:

        feature_extractor = self.feature_db.feature_extractor
        image_feature = feature_extractor(image)

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


if __name__ == '__main__':
    img_path = 'data/jpg'
    db_path = 'data/dbs/first_db'
    img_label = '100000.jpg'

    img = cv.cvtColor(cv.imread(f'data/jpg/{img_label}', cv.IMREAD_COLOR), cv.COLOR_BGR2RGB)
    f = FeatureDB(img_db='data/jpg')
    e1 = VGG16FeatureExtractor(weight_dir='data/model_weights/vgg16-397923af.pth')
    e2 = Method2Extractor()
    f.export_features(output_db_name='data/dbs/vgg16features', method=e1)
    # f = FeatureDB.load_feature_db(db_path)
    # r = Ranker(f)
    # d = CosineDistance()
    # results = r.query_normalized_rank(img, img_label, d)

