import sys
import os
from src.exception import CustomException
from src.utils import load_object
from keras.models import load_model


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path=os.path.join("artifacts","model.h5")
            model_weights_path=os.path.join("artifacts","model.h5")
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            print("Before Loading")
            model=load_model(model_path)
            model.load_weights(model_weights_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled=preprocessor.transform(features)
            data_scaled = data_scaled[:, 1:]
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)