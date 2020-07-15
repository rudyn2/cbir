from engine import Visualizer, Ranker, FeatureDB
from distances import CosineSimilarity


if __name__ == '__main__':
    img_path = 'data/jpg'

    f = FeatureDB.load_feature_db('data/dbs/method3features')
    f.set_img_db_path(img_path)

    r = Ranker(f)
    d = CosineSimilarity()
    v = Visualizer(img_path)

    query_images = ['101000.jpg', '100100.jpg', '107200.jpg', '102300.jpg', '100200.jpg', '130400.jpg', '111400.jpg']
    query_results = [r.query(q_img, similarity_fn=d) for q_img in query_images]
    v.plot_query_best_3(query_images, query_results, output_name='example3.jpg')

    print("")
