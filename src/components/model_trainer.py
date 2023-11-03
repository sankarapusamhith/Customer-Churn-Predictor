import sys
import keras
from keras.models import Sequential 
from keras.layers import Dense
from sklearn.metrics import confusion_matrix,accuracy_score
from src.exception import CustomException
from src.logger import logging
from config import ModelTrainerConfig
from src.utils import save_object
from keras.initializers import GlorotNormal
from keras.activations import swish
from keras.callbacks import EarlyStopping

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()


    def initiate_model_trainer(self,X_train,y_train,X_test,y_test):
        try:
            classifier = Sequential()
            
            #Add the hidden layers one by one using the dense function.
            classifier.add(Dense(units = 6, kernel_initializer = GlorotNormal(),activation = swish, input_dim = 9))
            
            #Add the second layer
            classifier.add(Dense(units = 6, kernel_initializer = GlorotNormal(),activation = swish))
            
            #Add the output layer
            classifier.add(Dense(units = 1, kernel_initializer = GlorotNormal(),activation = 'sigmoid'))

            #Compile the ANN
            classifier.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['accuracy'])
        
            #Fit the ANN to the Training Set
            early_stopping = EarlyStopping(monitor='val_loss', patience=3)
            classifier.fit(
                X_train,
                y_train,
                validation_data=(X_test, y_test),
                epochs=25,
                callbacks=[early_stopping],
                batch_size=5,
            )
            
            y_pred = classifier.predict(X_test)
            y_pred = (y_pred > 0.5)

            cm = confusion_matrix(y_test, y_pred)
            print(cm)
            
            accuracy = accuracy_score(y_test, y_pred)
            
            classifier.save('artifacts/model.h5')
            classifier.save_weights('artifacts/model_weights.h5')
            
            logging.info("Saved the model pickle file ")

            return accuracy

        except Exception as e:
            raise CustomException(e,sys)
        








