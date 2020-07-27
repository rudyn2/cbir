from engine import FeatureDB, Ranker, IRP
from distances import *
from tqdm import tqdm

if __name__ == '__main__':
    img_path = 'data/jpg'
    db_path = 'data/dbs'
    eval_dbs = False
    eval_irp = True

    feat1 = FeatureDB.load_feature_db(db_path + '/method1features')
    feat1.set_img_db_path(img_path)
    feat2 = FeatureDB.load_feature_db(db_path + '/method2features')
    feat2.set_img_db_path(img_path)
    feat3 = FeatureDB.load_feature_db(db_path + '/method3features')
    feat3.set_img_db_path(img_path)
    feat4 = FeatureDB.load_feature_db(db_path + '/method4features')
    feat4.set_img_db_path(img_path)

    feat_dbs = [feat1, feat2, feat3, feat4]

    if eval_dbs:
        for sim in [CosineSimilarity(), EuclideanSimilarity(), ChiSquareSimilarity()]:
            print(f"similarity: {sim.__class__.__name__}")
            for idx, feat_db in enumerate(feat_dbs):
                feat_db.set_img_db_path(img_path)
                print(f"Results FDB {idx+1}")
                ranker = Ranker(feat_db)
                ranks = []
                norm_ranks = []
                for class_ in tqdm(range(500), "Querying"):
                    query_image_filename = f'1{str(class_).zfill(3)}00.jpg'
                    _, _, rank, norm_rank = ranker.query_rank(query_image_filename, sim)
                    ranks.append(rank)
                    norm_ranks.append(norm_rank)

                avg_rank = np.mean(ranks)
                std_rank = np.std(ranks)
                avg_norm_rank = np.mean(norm_ranks)
                std_norm_rank = np.std(norm_ranks)

                print(f"Mean rank: {avg_rank:.4f} +- {std_rank:.4f}")
                print(f"Mean normalized rank: {avg_norm_rank:.5f} +- {std_norm_rank:.5f}\n")

    if eval_irp:
        for sim in [CosineSimilarity(), EuclideanSimilarity(), ChiSquareSimilarity()]:

            print(f"Applying IRP Algorithm using {len(feat_dbs)} feature dbs and {sim.__class__.__name__}")
            irp = IRP(feature_dbs=feat_dbs, similarity_fn=sim)

            ranks = []
            norm_ranks = []

            for class_ in tqdm(range(500), "Querying"):
                query_image_filename = f'1{str(class_).zfill(3)}00.jpg'
                results = irp.sort(image=query_image_filename)
                _, _, rank, norm_rank = Ranker.rank_this(query_image_filename, results, 10, len(feat1))
                ranks.append(rank)
                norm_ranks.append(norm_rank)

            avg_rank = np.mean(ranks)
            std_rank = np.std(ranks)
            avg_norm_rank = np.mean(norm_ranks)
            std_norm_rank = np.std(norm_ranks)

            print(f"Mean rank: {avg_rank:.4f} +- {std_rank:.4f}")
            print(f"Mean normalized rank: {avg_norm_rank:.5f} +- {std_norm_rank:.5f}\n")

