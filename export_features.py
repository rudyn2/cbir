from engine import FeatureDB
from features import *

if __name__ == '__main__':
    img_path = 'data/jpg'
    db_path = 'data/dbs'

    feature_extractor = Method4Extractor()
    FeatureDB(img_path).export_features(output_db_name='data/dbs/method4features', method=feature_extractor)