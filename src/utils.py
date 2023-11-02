import os
import sys
import pickle
from src.exception import CustomException
from sklearn.metrics import r2_score
from keras.models import load_model


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            file_obj.save(file_path.h5)
            return load_model('./models/file_path.h5')

    except Exception as e:
        raise CustomException(e, sys)
    