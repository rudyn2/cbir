import glob
from tqdm import tqdm
from features import *
from distances import *
from typing import List
import torch
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

plt.rcParams.update({'font.size': 25})


def zero_pad(obj, reference_shape: tuple):
    result = []
    for channel in range(obj.shape[2]):
        channel_padded = np.zeros(reference_shape)
        channel_padded[:obj.shape[0], :obj.shape[1]] = obj[:, :, channel]
        result.append(channel_padded)
    return np.stack(result, axis=-1).astype('uint8')


class FeatureDB(object):

    def __init__(self, img_db: str):

        self.images = None
        self.img_db_path = img_db
        self.features: dict = {}
        self.feature_extractor: FeatureExtractor = None

    @classmethod
    def load_feature_db(cls, db_path: str):
        with open(db_path, 'rb') as f:
            feature_db = torch.load(f, map_location=torch.device('cpu'))
        return feature_db

    def load_images(self):
        img_filenames = glob.glob(self.img_db_path + '/*.jpg')
        db = {}
        for image_filename in tqdm(img_filenames, "Loading images"):
            image_label = image_filename.split("/")[-1]
            db[image_label] = cv.cvtColor(cv.imread(image_filename, cv.IMREAD_COLOR), cv.COLOR_BGR2RGB)
        return db

    def export_features(self, output_db_name: str, method: FeatureExtractor):
        """
        Exports features to a pickle object.

        :param output_db_name:
            Name of output database with features.
        :param method:
            Method used for feature extraction
        """
        img_filenames = glob.glob(self.img_db_path + '/*.jpg')
        features = {}

        if isinstance(method, HistogramFeatureExtractor):
            for image_filename in tqdm(img_filenames, "Extracting features ..."):
                image_label = image_filename.split("/")[-1]
                img = cv.cvtColor(cv.imread(image_filename, cv.IMREAD_COLOR), cv.COLOR_BGR2RGB)
                features[image_label] = method(img)
        elif isinstance(method, HogExtractor):
            for image_filename in tqdm(img_filenames, "Extracting features ..."):
                image_label = image_filename.split("/")[-1]
                img = cv.cvtColor(cv.imread(image_filename, cv.IMREAD_COLOR), cv.COLOR_BGR2RGB)
                features[image_label] = method(zero_pad(img, reference_shape=(3888, 3888)))

        elif isinstance(method, CNNFeatureExtractor):

            # stacks them into a tensor
            for image_filename in tqdm(img_filenames, "Extracting features"):
                img_list = [cv.cvtColor(cv.imread(image_filename, cv.IMREAD_COLOR), cv.COLOR_BGR2RGB)]
                img_names = [image_filename.split("/")[-1]]
                img_stacked = np.stack(img_list)
                img_similarities = method(img_stacked)
                features.update(dict(zip(img_names, img_similarities)))

        self.features = features
        self.feature_extractor = method

        print(f"Saving db in {output_db_name}")
        with open(output_db_name, 'wb') as f:
            torch.save(self, f)

        print("Ready!")

    def set_img_db_path(self, img_db_path: str):
        self.img_db_path = img_db_path

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
        return int(s.split(".")[0][1:4])

    def query(self, image: str, similarity_fn: SimilarityMeasure) -> List[np.array]:
        """Makes a query to the DB and returns the sorted similarity between the query and each image
        stored in the DB."""

        image = cv.cvtColor(cv.imread(self.feature_db.img_db_path + '/' + image, cv.IMREAD_COLOR), cv.COLOR_BGR2RGB)
        feature_extractor = self.feature_db.feature_extractor
        if isinstance(feature_extractor, HistogramFeatureExtractor):
            image_feature = feature_extractor(image)
        else:
            # dirty adapter code
            image_feature = feature_extractor(np.stack([image]))[0]

        similarities = []
        for key, other_feature in self.feature_db.features.items():
            img_name, similarity = key, similarity_fn(image_feature, other_feature)
            similarities.append((img_name, similarity))

        return sorted(similarities, key=lambda x: x[1], reverse=True)

    def query_rank(self, image_label: str, similarity_fn: SimilarityMeasure, top_k: int = 10):
        """Makes a query to the DB and returns the top k results, the rank and number of total relevant images
        that belongs to the same class as the query."""

        query_results = self.query(image_label, similarity_fn)
        return self.rank_this(image_label, query_results, top_k, len(self.feature_db))

    @classmethod
    def rank_this(cls, image_label: str, query_results: list, top_k: int, total: int):

        query_class = cls.extract_class(image_label)
        top_query_results = query_results[:top_k]
        rank = 0
        total_query_in_class = 0
        for idx, (label, score) in enumerate(top_query_results):
            img_class = cls.extract_class(label)
            if img_class == query_class:
                rank += idx + 1
                total_query_in_class += 1
        rank /= total_query_in_class
        norm_rank = (rank - (total_query_in_class + 1) / 2) / (total - total_query_in_class)

        return top_query_results, total_query_in_class, rank, norm_rank


class IRP(object):
    """Auxiliary class for Implementation of Inverse Rank Position."""

    def __init__(self, feature_dbs: List[FeatureDB], similarity_fn):
        self.feature_dbs = feature_dbs
        self.similarity_fn = similarity_fn

    def sort(self, image: str):
        """Sorts common images using IRP algorithm based on all the feature dbs."""
        # make a query in all dbs
        images_ranked_by_fdb = []
        for fdb in self.feature_dbs:
            r = Ranker(fdb)
            images_ranked_by_fdb.append(r.query(image, self.similarity_fn)[:10])

        # find the intersection of the names
        img_names = set([img_name for img_name, _ in images_ranked_by_fdb[0]])
        for image_rank in images_ranked_by_fdb:
            new_names = [img_name for img_name, _ in image_rank]
            img_names.intersection(new_names)

        # calculates the IRP Score using of the common images based on all the db's scores
        results = []
        for image_name in img_names:
            results.append((image_name, self.irp_score(image_name, images_ranked_by_fdb)))
        return sorted(results, key=lambda x: x[1], reverse=False)

    def irp_score(self, image_to_compare: str, images_ranked: list):
        """Calculates the IRP Score for a single image using the given rankings."""

        total = 0
        for images_ranked in images_ranked:
            rank = self.get_rank(image_to_compare, images_ranked)
            if rank is None:
                continue
            total += 1 / rank
        if total == 0:
            raise ValueError("It can't rank the image")
        return 1 / total

    @staticmethod
    def get_rank(image: str, ranked_images: list):
        for idx, (img_name, _) in enumerate(ranked_images):
            if img_name == image:
                return idx + 1
        return None


class Visualizer(object):
    """Auxiliary class for CBIR results visualization."""

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

        fig = plt.figure(figsize=(32, 25))
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

    def plot_query_best_3(self, query_img: List[str], img_results: List[List], output_name: str = ''):

        assert len(query_img) == len(img_results)
        assert len(query_img) <= 7

        fig, axs = plt.subplots(nrows=4, ncols=len(query_img), figsize=(32, 34))
        for i in range(4):
            for j in range(len(query_img)):
                # first row: query images
                ax = axs[i, j]
                if i == 0:
                    img_ = self.read_img(query_img[j])
                    ax.set(title=f"q({query_img[j]})")
                else:
                    img_ = self.read_img(img_results[j][i-1][0])
                    ax.set(title=f"{img_results[j][i-1][1]:.4f}")
                ax.axis('off')
                ax.imshow(img_)
        plt.tight_layout()
        if output_name != '':
            plt.savefig(output_name, optimize=True)
        plt.show()
