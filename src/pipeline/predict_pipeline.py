import sys
import os
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path=os.path.join("artifacts","model.pkl")
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def femalepredict(self,features):
        try:
            female_model_path=os.path.join("artifacts","female_model.pkl")
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')

            print("Before Loading Female Model")
            model=load_object(file_path=female_model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading Female Model")
            data_scaled=preprocessor.transform(features)
            # same 4 feature indices used in female trainer
            feature_indices=[0,4,5,11]  
            data_scaled=data_scaled[:,feature_indices]
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)


class CustomData:
    def __init__(  self,
        Age: int,
        Sex: str,
        ChestPainType: str,
        RestingBP: int,
        Cholesterol: int,
        FastingBS: int,
        RestingECG: str,
        MaxHR: int,
        ExerciseAngina: str,
        Oldpeak: float,
        ST_Slope: str):

        self.Age = Age
        self.Sex = Sex
        self.ChestPainType = ChestPainType
        self.RestingBP = RestingBP
        self.Cholesterol = Cholesterol
        self.FastingBS = FastingBS
        self.RestingECG = RestingECG
        self.MaxHR = MaxHR
        self.ExerciseAngina = ExerciseAngina
        self.Oldpeak = Oldpeak
        self.ST_Slope = ST_Slope

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "Age": [self.Age],
                "Sex": [self.Sex],
                "ChestPainType": [self.ChestPainType],
                "RestingBP": [self.RestingBP],
                "Cholesterol": [self.Cholesterol],
                "FastingBS": [self.FastingBS],
                "RestingECG": [self.RestingECG],
                "MaxHR": [self.MaxHR],
                "ExerciseAngina": [self.ExerciseAngina],
                "Oldpeak": [self.Oldpeak],
                "ST_Slope": [self.ST_Slope]
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)