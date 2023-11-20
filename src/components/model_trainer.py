## Basic Imports
import numpy as np 
import pandas as pd
from sklearn.linear_model import LinearRegression,Ridge,Lasso,ElasticNet
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor 
from sklearn.ensemble import AdaBoostRegressor
from src.logger import logging
from src.exception import CustomException
from src.util import save_object,evaluate_model
from dataclasses import dataclass
import sys
import os

@dataclass
class ModelTrainerconfig:
    trained_model_file_path=os.path.join('artifacts','model.pkl')


class ModelTrainer():
    def __init__(self):
        self.model_trainer_config = ModelTrainerconfig()

    def initiate_model_training(self,train_array,test_array):
        try:
            logging.info("Splitting dependent and independent variables from train and test arrays")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models={
            'LinearRegression':LinearRegression(),
            'Lasso':Lasso(),
            'Ridge':Ridge(),
            'Elasticnet':ElasticNet(),
            'DecisionTreeRegressor':DecisionTreeRegressor(max_depth=50),
            'GradientBoostingRegressor':GradientBoostingRegressor(
                n_estimators = 100,  # Number of boosting stages or trees
                learning_rate = 0.1,  # Shrinkage or step size
                max_depth = 3 , # Maximum depth of each tree
                subsample = 1.0 , # Fraction of samples used for training each tree
                min_samples_split = 2 , # Minimum number of samples required to split an internal node
                min_samples_leaf = 1 , # Minimum number of samples required at a leaf node
                max_features = None 
            ),
            'RandomForestRegressor':RandomForestRegressor(n_estimators=100,max_depth=50),
            'AdaBoostRegressor':AdaBoostRegressor()
            }
            model_report=evaluate_model(X_train,y_train,X_test,y_test,models)
            print(model_report)
            print("\n============================================================")
            logging.info(f"Model report: {model_report}")
            #To get best model score from dictionary
            best_model_score =max(sorted(model_report.values()))
            best_model_name=list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model=models[best_model_name]
            
            print(f"Best model found : {best_model_name},R2 Score: {best_model_score}")
            print("\n============================================================")
            
            logging.info(f"Best model found : {best_model_name},R2 Score: {best_model_score}") 

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

        except Exception as e:
            logging.info("Error in model training")
            raise CustomException(e,sys)

