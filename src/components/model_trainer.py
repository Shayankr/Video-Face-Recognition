# Importing necessary libraries and models
import numpy as np
import pandas as pd


from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_model

from dataclasses import dataclass
import os,sys

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')


class ModelTrainer:
    def _init_(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self, train_array, test_array):
        try:
            logging.info("Splitting Dependent and Independent Variables from Train and Test Data:")
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models = {
                'LogisticRegression': LogisticRegression(),
                'DecisionTree' : DecisionTreeClassifier(),
                'RandomForest' : RandomForestClassifier(n_estimators=100, max_depth=4),
                'SVM' : SVC(),
                'XGBoost' : XGBClassifier(max_depth = 4)
            }

            model_report:dict = evaluate_model(X_train=X_train, X_test=X_test, 
                                               models=models)
            logging.info(f'Model Report: {model_report}')

            ## To get best Model Score from dictionary:
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]
            logging.info(f'Best Model found, Model Name is: {best_model_name} with score: {best_model_score}')

            ## Save BEST Model:
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj= best_model
                )

        except Exception as e:
            logging.info("Error in Model Training Steps!")
            raise CustomException(e, sys)
        
#