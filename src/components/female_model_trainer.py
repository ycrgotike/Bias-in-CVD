import os
import sys
from dataclasses import dataclass

from catboost import CatBoostClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models


@dataclass
class FemaleModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "female_model.pkl")


class FemaleModelTrainer:

    def __init__(self):
        self.model_trainer_config = FemaleModelTrainerConfig()

    def initiate_female_model_trainer(self, train_array, test_array):

        try:
            logging.info("Split training and test input data")

            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            # ---- FEMALE FILTER ----
            # assuming Sex_F column exists after OneHotEncoding
            sex_female_index = 7  # adjust based on encoder order

            female_train_mask = X_train[:, sex_female_index] == 1
            female_test_mask = X_test[:, sex_female_index] == 1

            X_train = X_train[female_train_mask]
            y_train = y_train[female_train_mask]

            X_test = X_test[female_test_mask]
            y_test = y_test[female_test_mask]

            # ---- SELECT 4 FEATURES ----
            # indexes must match transformed feature positions
            feature_indices = [0, 4, 5, 11]  # Age, Cholesterol, FastingBS, ST_Slope

            X_train = X_train[:, feature_indices]
            X_test = X_test[:, feature_indices]

            models = {
                "Random Forest": RandomForestClassifier(),
                "Decision Tree": DecisionTreeClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(),
                "Logistic Regression": LogisticRegression(),
                "XGBClassifier": XGBClassifier(),
                "CatBoosting Classifier": CatBoostClassifier(verbose=False),
                "AdaBoost Classifier": AdaBoostClassifier(),
                "Support Vector Classifier": SVC(),
                "K-NN Classifier": KNeighborsClassifier(),
            }

            parameters = {

                "Decision Tree": {
                    "criterion": ["gini", "entropy", "log_loss"],
                },

                "Random Forest": {
                    "criterion": ["gini", "entropy", "log_loss"],
                },

                "Gradient Boosting": {
                    "learning_rate": [0.1, 0.01, 0.05, 0.001],
                    "subsample": [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                    "n_estimators": [8, 16, 32, 64, 128, 256],
                },

                "Logistic Regression": {
                    "C": [0.01, 0.1, 1, 10, 100],
                    "solver": ["saga"],
                    "l1_ratio": [0, 0.5, 1],
                    "max_iter": [1000, 2000, 5000],
                },

                "XGBClassifier": {
                    "learning_rate": [0.1, 0.01, 0.05, 0.001],
                    "n_estimators": [8, 16, 32, 64, 128, 256],
                },

                "CatBoosting Classifier": {
                    "depth": [6, 8, 10],
                    "learning_rate": [0.01, 0.05, 0.1],
                    "iterations": [30, 50, 100],
                },

                "AdaBoost Classifier": {
                    "learning_rate": [0.1, 0.01, 0.5, 0.001],
                    "n_estimators": [8, 16, 32, 64, 128, 256],
                },

                "Support Vector Classifier": {
                    "C": [0.1, 1, 10, 100],
                    "kernel": ["linear", "rbf", "poly", "sigmoid"],
                },

                "K-NN Classifier": {
                    "n_neighbors": [3, 5, 7, 9, 11],
                    "p": [1, 2],
                },

            }

            model_report: dict = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                param=parameters,
            )

            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found")

            logging.info("Best female model found")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model,
            )

            predicted = best_model.predict(X_test)

            recall = recall_score(y_test, predicted)

            logging.info(f"Best Female Model: {best_model_name} | Recall Score: {recall:.4f}")
            print(f"Best Female Model: {best_model_name} | Recall Score: {recall:.4f}")

            return best_model_name, recall

        except Exception as e:
            raise CustomException(e, sys)