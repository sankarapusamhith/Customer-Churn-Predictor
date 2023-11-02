import sys
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from src.exception import CustomException
from src.logger import logging
from config import TransformationConfig
from src.utils import save_object

class Transformation:
    def __init__(self):
        self.data_transformation_config=TransformationConfig()
        
    def get_data_transformer_object(self,data):
        
        try:

            num_col = list(data.select_dtypes(include=['float64','int64']).columns)
            cat_col = list(data.select_dtypes(include=['object']).columns)
            num_col.pop()

            num_pipeline= Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="median")),
                ("scaler",StandardScaler())

                ]
            )

            cat_pipeline=Pipeline(

                steps=[
                ("imputer",SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder",OneHotEncoder()),
                ("scaler",StandardScaler(with_mean=False))
                ]

            )

            logging.info(f"Categorical columns: {cat_col}")
            logging.info(f"Numerical columns: {num_col}")

            preprocessor=ColumnTransformer(
                [
                ("num_pipeline",num_pipeline,num_col),
                ("cat_pipelines",cat_pipeline,cat_col)
                ]
            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)

    def initiate_data_transformation(self,train_path,test_path):

        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            
            train_df=train_df.iloc[:,2:12]
            test_df=test_df.iloc[:,2:12]
            
            logging.info("Read train and test data completed")
            logging.info("Obtaining preprocessing object")

            preprocessing_obj=self.get_data_transformer_object(train_df)

            target_column_name="Exited"

            x_train_df=train_df.drop(columns=[target_column_name],axis=1)
            y_train=train_df[target_column_name].values

            x_test_df=test_df.drop(columns=[target_column_name],axis=1)
            y_test=test_df[target_column_name].values

            logging.info("Applying preprocessing object on training dataframe and testing dataframe.")

            x_train=preprocessing_obj.fit_transform(x_train_df)
            x_test=preprocessing_obj.transform(x_test_df)
            
            x_train = x_train[:, 1:]
            x_test = x_test[:, 1:]

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            
            logging.info(f"Saved preprocessing object.")

            return (
                x_train,
                x_test,
                y_train,
                y_test,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        
        except Exception as e:
            raise CustomException(e,sys)