# This has common functionality which the entire project can use

import sys #Python runtime - CLI scripts, debugging imports, exiting
import os #interface between my script and operating system - Files, folders, environment variables
import dill
import pickle
import numpy as np
import pandas as pd

from src.exception import CustomException
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)

    
