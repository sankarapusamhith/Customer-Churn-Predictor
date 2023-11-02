import sys
import keras
from keras.models import Sequential 
from keras.layers import Dense
from sklearn.metrics import confusion_matrix,accuracy_score
from src.exception import CustomException
from src.logger import logging
from config import ModelTrainerConfig
from src.utils import save_object

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()


    def initiate_model_trainer(self,X_train,y_train,X_test,y_test):
        try:
            classifier = Sequential()
            
            #Add the hidden layers one by one using the dense function.
            classifier.add(Dense(units = 6, kernel_initializer = 'uniform',activation = 'relu', input_dim = 9))
            
            #Add the second layer
            classifier.add(Dense(units = 6, kernel_initializer = 'uniform',activation = 'relu'))
          
            #Add the output layer
            classifier.add(Dense(units = 1, kernel_initializer = 'uniform',activation = 'sigmoid'))

            #Compile the ANN
            classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
        
            #Fit the ANN to the Training Set
            classifier.fit(X_train, y_train, batch_size = 10, epochs = 25)
            
            y_pred = classifier.predict(X_test)
            y_pred = (y_pred > 0.5)

            cm = confusion_matrix(y_test, y_pred)
            
            accuracy = accuracy_score(y_test, y_pred)
            
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=classifier
            )
            
            logging.info("Saved the model pickle file ")

            return accuracy

        except Exception as e:
            raise CustomException(e,sys)
        








