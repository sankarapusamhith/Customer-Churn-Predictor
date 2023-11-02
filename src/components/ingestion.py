import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from src.components.model_trainer import ModelTrainer
from src.components.transformer import Transformation
from src.exception import CustomException
from src.logger import logging as lg
from config import IngestionConfig

class Ingestion:
    def __init__(self):
        self.ingestion_config=IngestionConfig()

    def initiate_data_ingestion(self):

        lg.info("Entered the data ingestion method ")

        try:
            df=pd.read_csv('data/loan_approval.csv')
            lg.info('Read the dataset as dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            lg.info("Train test split initiated")

            train_df,test_df=train_test_split(df,test_size=0.2,random_state=10)
            train_df.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_df.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            lg.info("Ingestion of the data is completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path

            )
        except Exception as e:
            raise CustomException(e,sys)

if __name__=="__main__":
    obj=Ingestion()
    train_data,test_data=obj.initiate_data_ingestion()
    lg.info("completed the data ingestion method ")

    data_transformation=Transformation()
    X_train, X_test, y_train, y_test,_=data_transformation.initiate_data_transformation(train_data,test_data)
    lg.info("completed the data transformaton method ")
    
    modeltrainer=ModelTrainer()
    accuracy=modeltrainer.initiate_model_trainer(X_train,y_train,X_test,y_test)
    lg.info("completed the model training  ")
    lg.info(f"Accuracy = {accuracy}")
    print("Accuracy = ",accuracy)