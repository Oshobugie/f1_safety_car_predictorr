import sys
import os
import mlflow
import mlflow.xgboost
import numpy as np
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, roc_auc_score, accuracy_score, classification_report

from src.logger import logging
from src.exception import CustomException
from src.entity.config_entity import ModelTrainerConfig
from src.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact


class ModelTrainer:
    """Trains, evaluates, and logs the XGBoost Safety Car classifier using MLflow."""

    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def initiate_model_trainer(self, transformation_artifact: DataTransformationArtifact) -> ModelTrainerArtifact:
        """Splits transformed data, handles class imbalance, logs metrics to MLflow, and exports the model."""
        try:
            logging.info("Loading transformed data matrix...")
            transformed_arr = np.load(transformation_artifact.transformed_data_file_path)

            # Unpack scaled feature columns and the trailing binary target column
            X = transformed_arr[:, :-1]
            y = transformed_arr[:, -1]

            # Stratify the split to preserve the rare Safety Car class ratio in both sets
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            logging.info(f"Data split: Train shape {X_train.shape}, Test shape {X_test.shape}")

            # Compute class weight strictly from training labels to offset heavy green-flag imbalance
            neg_count = np.sum(y_train == 0)
            pos_count = np.sum(y_train == 1)
            scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
            logging.info(f"Calculated scale_pos_weight: {scale_pos_weight:.2f}")

            mlflow.set_experiment("F1_Safety_Car_Prediction")

            with mlflow.start_run():
                params = {
                    "n_estimators": 100,
                    "max_depth": 5,
                    "learning_rate": 0.1,
                    "scale_pos_weight": scale_pos_weight,
                    "random_state": 42,
                    "eval_metric": "logloss"
                }

                mlflow.log_params(params)

                model = XGBClassifier(**params)
                model.fit(X_train, y_train)

                y_pred = model.predict(X_test)
                y_proba = model.predict_proba(X_test)[:, 1]

                # Prioritize ROC-AUC and Log Loss over raw accuracy to assess probability calibration under imbalance
                test_log_loss = log_loss(y_test, y_proba)
                test_roc_auc = roc_auc_score(y_test, y_proba)
                test_acc = accuracy_score(y_test, y_pred)

                mlflow.log_metric("log_loss", test_log_loss)
                mlflow.log_metric("roc_auc", test_roc_auc)
                mlflow.log_metric("accuracy", test_acc)

                logging.info(f"Model Evaluation -> ROC-AUC: {test_roc_auc:.4f}, Log Loss: {test_log_loss:.4f}")

                # Log model artifact to MLflow registry and export native XGBoost binary for FastAPI serving
                mlflow.xgboost.log_model(model, "model")

                model_dir = Path(self.config.trained_model_file_path).parent
                os.makedirs(model_dir, exist_ok=True)
                model.save_model(str(self.config.trained_model_file_path))
                logging.info(f"Model saved locally at: {self.config.trained_model_file_path}")

            return ModelTrainerArtifact(
                trained_model_file_path=self.config.trained_model_file_path,
                metric_artifact={"roc_auc": test_roc_auc, "log_loss": test_log_loss}
            )

        except Exception as e:
            raise CustomException(e, sys)