import glob
import cv2 as cv
from tqdm import tqdm
from features import Method1Extractor, Method2Extractor, Method3Extractor, Method4Extractor, FeatureExtractor
import pickle
import numpy as np
from distances import *
from typing import List


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
        for image_filename in tqdm(img_filenames, "Loading db"):
            image_label = image_filename.split("/")[-1]
            db[image_label] = cv.cvtColor(cv.imread(image_filename, cv.IMREAD_COLOR), cv.COLOR_BGR2RGB)
        return db

    @classmethod
    def load_feature_db(cls, db_path: str):
        with open(db_path, 'rb') as f:
            feature_db = pickle.load(f)
        return feature_db

    def load_images(self):
        self.images = self._load_images(self.img_db_path)

    def export_features(self, output_db_name: str, method: str):
        """
        Exports features in a pickle object.

        :param output_db_name:
            Name of output database with features.
        :param method:
            Method used for feature extraction
        """

        assert method in ['1', '2', '3', '4'], f"Expected method 1, 2, 3 or 4 but got {method}."

        db = self._load_images(self.img_db_path) if not self.images else self.images

        features = {}
        methods = {'1': Method1Extractor(),
                   '2': Method2Extractor(),
                   '3': Method3Extractor(),
                   '4': Method4Extractor()}

        for key, img in tqdm(db.items(), "Extracting features"):
            features[key] = methods[method](img)

        self.features = features
        self.feature_extractor = methods[method]

        print(f"Saving db in {output_db_name}")
        with open(output_db_name, 'wb') as f:
            pickle.dump(self, f)

        print("Ready!")


class Ranker(object):

    def __init__(self, feature_db: FeatureDB):
        self.feature_db: FeatureDB = feature_db

    def query(self, image: np.array, similarity_fn: DistanceMeasure) -> List[np.array]:

        feature_extractor = self.feature_db.feature_extractor
        image_feature = feature_extractor(image)

        similarities = []
        for key, other_feature in tqdm(self.feature_db.features.items(), "Searching in db"):
            img_name, similarity = key, similarity_fn(image_feature, other_feature)
            similarities.append((img_name, similarity))

        return sorted(similarities, key=lambda x: x[1], reverse=True)


if __name__ == '__main__':
    img_path = 'data/jpg'
    db_path = 'data/dbs/first_db'
    img = cv.cvtColor(cv.imread('data/jpg/100101.jpg', cv.IMREAD_COLOR), cv.COLOR_BGR2RGB)
    f = FeatureDB.load_feature_db(db_path)
    r = Ranker(f)
    d = CosineDistance()
    results = r.query(img, d)

