from engine import FeatureDB
from features import *

if __name__ == '__main__':
    img_path = 'data/jpg'
    db_path = 'data/dbs'

    feature_extractor = HogExtractor(32)
    FeatureDB(img_path).export_features(output_db_name='data/dbs/hogfeatures32', method=feature_extractor)