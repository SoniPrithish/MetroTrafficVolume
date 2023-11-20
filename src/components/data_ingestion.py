import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

## Initialize Data Ingestion Configuration
@dataclass
class DataIngestionconfig:
    train_data_path:str=os.path.join('artifacts','train.csv')
    test_data_path:str=os.path.join('artifacts','test.csv')
    raw_data_path:str=os.path.join('artifacts','raw.csv')

## Create a Class For Data Ingestion
class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionconfig
    
    def initiate_data_ingestion(self):
        logging.info('Data Ingestion initialized')
        try:
            df=pd.read_csv('notebooks\data\Metro_Interstate_Traffic_Volume.csv')
            logging.info('Data Set Read as pandas data frame')

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False)
            # convert the date_time column to datetime type
            df['date_time'] = pd.to_datetime(df['date_time'])
            df['time'] = df['date_time'].dt.hour
            df['month'] =df['date_time'].dt.month
            df['year'] =df['date_time'].dt.year
            df['day'] = df['date_time'].dt.day_name()
            df = df[df['temp'] != 0]
            df = df[df.rain_1h < 100]
            z = lambda x: False if x == 'None' else True
            df['holiday'] = df['holiday'].apply(z)
            logging.info('Splitting Data into Train and Test Data')
            train_set,test_set=train_test_split(df,test_size=0.25)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info('Ingestion of Data is Completed')

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )



        except Exception as e:
            logging.info('Exception in Data Ingestion Stage')
            raise CustomException(e,sys)   

