import sys
import os
from src.exception import CustomException
from src.logger import logging
from src.util import load_object
import pandas as pd


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            model_path=os.path.join('artifacts','model.pkl')

            preprocessor=load_object(preprocessor_path)
            model=load_object(model_path)

            data_scaled=preprocessor.transform(features)

            pred=model.predict(data_scaled)
            return pred
            

        except Exception as e:
            logging.info("Exception occured in prediction")
            raise CustomException(e,sys)
        
class CustomData:
    def __init__(self,
                 Temp:float,
                 Rain:float,
                 Snow:float,
                 Clouds:float,
                 Holiday:bool,
                 Weather:float,
                 Time:str,
                 Month:str,
                 Day:str):
        
        self.Temp = Temp
        self.Rain =Rain
        self.Snow =Snow
        self.Clouds =Clouds
        self.Holiday =Holiday
        self.Weather =Weather
        self.Time =Time
        self.Month = Month
        self.Day = Day

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'temp':[self.Temp],
                'rain_1h':[self.Rain],
                'snow_1h':[self.Snow],
                'clouds_all':[self.Clouds],
                'holiday':[self.Holiday],
                'weather_main':[self.Weather],
                'time':[self.Time],
                'month':[self.Month],
                'day':[self.Day]
            }
            df = pd.DataFrame(custom_data_input_dict)
            logging.info('Dataframe Gathered')
            logging.info(df.head())
            return df
        except Exception as e:
            logging.info('Exception Occured in prediction pipeline')
            raise CustomException(e,sys)