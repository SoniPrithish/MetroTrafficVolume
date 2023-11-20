import os
import sys
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer ## HAndling Missing Values
from sklearn.preprocessing import StandardScaler # HAndling Feature Scaling
from sklearn.preprocessing import OrdinalEncoder # Ordinal Encoding
## pipelines
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from src.util import save_object

@dataclass 
class DataTransformationconfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.DataTransformationconfig=DataTransformationconfig()

    def get_data_transformation(self):
        try:
            logging.info('Data transformation Started')
            numerical_cols = ['temp', 'rain_1h', 'snow_1h', 'clouds_all']
            categorical_cols = ['holiday', 'weather_main', 'time', 'month', 'day']
            
            logging.info('Pipeline initiated')
            ## Numerical Pipeline
            num_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler())
                ])

            # Categorigal Pipeline
            cat_pipeline= Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OrdinalEncoder())
                ])


            preprocessor=ColumnTransformer([
            ('num_pipeline',num_pipeline,numerical_cols),
            ('cat_pipeline',cat_pipeline,categorical_cols)
            ])
            logging.info('Pipeline Completed')
            return preprocessor 


        except Exception as e:
            logging.info('Error getting data transformation')
            raise CustomException(e,sys)

    def initiate_data_transformation(self,train_path,test_path):
        try:
            #Reading training asnd testing data
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            ## Logging Information
            logging.info('Reading of Train and Test Data completed')
            logging.info(f'Train Dataframe Head : \n{train_df.head().to_string()}')
            logging.info(f'Test Dataframe Head  : \n{test_df.head().to_string()}')
    
            logging.info("Getting PreProcessing Object")

            preprocessing_obj=self.get_data_transformation()
            
            target_column_name='traffic_volume'
            drop_columns=[target_column_name,'date_time', 'weather_description','year']

            input_feature_train_df=train_df.drop(columns=drop_columns,axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=drop_columns,axis=1)
            target_feature_test_df=test_df[target_column_name]

            
            ## Transforming using pre processor object
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)
            logging.info('Applying preprocessing object on train and test data')

            train_arr=np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr=np.c_[input_feature_test_arr,np.array(target_feature_test_df)]

            save_object(
                file_path=self.DataTransformationconfig.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            logging.info("preprocessing pickle file saved")
            return(
                train_arr,
                test_arr,
                self.DataTransformationconfig.preprocessor_obj_file_path
            )


        except Exception as e:
            logging.info("Error occured in initiate data transformation")
            raise CustomException(e,sys)