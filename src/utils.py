import os
import sys
import pickle
import numpy as np
import pandas as pd

from src.exception import CustomException
from src.logger import logging

# import necessary libraries, modules, packages




# define a function according to need -- Mr. Shayan Kr: and also do not forget to use try,exception block and add log function also to save the logging info.

## define a function to save the transformations and models in pickle format:
def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file=file_path, mode="wb") as file_obj:
            pickle.dump(obj=obj, file=file_obj)

    except Exception as e:
        raise CustomException(e, sys)





# define a function according to need -- Mr. Shayan Kr: and also do not forget to use try,exception block and add log function also to save the logging info.

## define a function to load objects/models:
def load_object(file_path):
    try:
        with open(file=file_path, mode="rb") as file_obj:
            return pickle.load(file=file_obj)
        
    except Exception as e:
        logging.info('Exception occured in load_object functionalities!')
        raise CustomException(e, sys)




# define a function according to need -- Mr. Shayan Kr: and also do not forget to use try,exception block and add log function also to save the logging info.

## define a function to evaluate model:
def evaluate_model(X_train, y_train, X_test, models):
    try:
        pass
    except Exception as e:
        logging.info("Model Evaluation Error Occured!")
        raise CustomException(e, sys)






# define a function according to need -- Mr. Shayan Kr: and also do not forget to use try,exception block and add log function also to save the logging info.






# define a function according to need -- Mr. Shayan Kr: and also do not forget to use try,exception block and add log function also to save the logging info.






# define a function according to need -- Mr. Shayan Kr: and also do not forget to use try,exception block and add log function also to save the logging info.


