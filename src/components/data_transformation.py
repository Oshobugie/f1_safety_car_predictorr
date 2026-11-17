import sys
import os
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from src.logger import logging
from src.exception import CustomException
from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import DataIngestionArtifact, DataTransformationArtifact

# Ground truth mapping of 2023 Grand Prix rounds to confirmed Safety Car lap numbers
SAFETY_CAR_LAPS = {
    1: [],
    2: [18, 19, 20],
    3: [12, 13, 14, 54, 55, 56],
    4: [14, 15],
    5: [],
    6: [55, 56, 57],
    7: [],
    8: [12, 13, 14, 15],
    9: [1, 2, 3],
    10: [33, 34, 35, 36, 37, 38],
    11: [1, 2],
    12: [1, 2],
    13: [16, 17, 42, 43],
    14: [41, 42, 43],
    15: [20, 21, 60, 61, 62],
    16: [1, 2],
    17: [1, 2, 3, 4, 11, 12],
    18: [1, 2, 3, 4, 5],
    19: [1, 2, 3, 4, 33, 34, 35, 36, 37, 38, 39],
    20: [1, 2, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71],
    21: [1, 2, 3, 26, 27, 28],
    22: []
}


class DataTransformation:
    """Cleans raw lap telemetry, engineers target labels, and fits the preprocessing pipeline."""

    def __init__(self, config: DataTransformationConfig):
        self.config = config

    @staticmethod
    def convert_time_to_seconds(time_str: str) -> float | None:
        """Parses Ergast lap time strings ('M:SS.sss') into total elapsed seconds."""
        if pd.isna(time_str) or not isinstance(time_str, str):
            return None
        try:
            minutes, seconds = time_str.split(":")
            return int(minutes) * 60 + float(seconds)
        except Exception:
            return None

    @staticmethod
    def check_safety_car(row: pd.Series) -> int:
        """Assigns binary ground-truth label (1 = Safety Car active, 0 = Green Flag)."""
        race_round = row["round"]
        lap = row["lap_number"]
        if race_round in SAFETY_CAR_LAPS and lap in SAFETY_CAR_LAPS.get(race_round, []):
            return 1
        return 0

    def get_data_transformer_object(self) -> Pipeline:
        """Builds a scikit-learn preprocessing pipeline for numerical telemetry features."""
        try:
            # Median imputation protects against extreme outlier lap times (e.g., red flags or pit stops)
            num_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ])
            return num_pipeline
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, ingestion_artifact: DataIngestionArtifact) -> DataTransformationArtifact:
        """Transforms ingested CSV data into scaled feature arrays and persists the fitted preprocessor."""
        try:
            logging.info("Starting Data Transformation stage...")

            df = pd.read_csv(ingestion_artifact.ingested_data_file_path)

            # Parse formatted lap durations and label rows against historical Safety Car logs
            df["lap_time_sec"] = df["time"].apply(self.convert_time_to_seconds)
            df["is_safety_car"] = df.apply(self.check_safety_car, axis=1)

            # Strip identifiers and calendar keys ('round', 'lap_number') after labeling to prevent target leakage
            df.drop(
                columns=["driverId", "time", "raceName", "season", "round", "lap_number"],
                inplace=True,
                errors="ignore"
            )

            # Separate model input features [position, lap_time_sec] from the binary target
            target_column = "is_safety_car"
            X = df.drop(columns=[target_column])
            y = df[target_column]

            # Fit and apply scaling transformations
            preprocessor = self.get_data_transformer_object()
            X_transformed = preprocessor.fit_transform(X)

            # Combine transformed features and target into a single NumPy array for downstream training
            transformed_arr = np.c_[X_transformed, np.array(y)]

            # Persist the transformed dataset and fitted preprocessor so inference uses identical scaling
            os.makedirs(os.path.dirname(self.config.transformed_data_file_path), exist_ok=True)
            np.save(self.config.transformed_data_file_path, transformed_arr)
            joblib.dump(preprocessor, self.config.preprocessor_obj_file_path)

            logging.info("Data Transformation completed successfully.")

            return DataTransformationArtifact(
                transformed_data_file_path=self.config.transformed_data_file_path,
                preprocessor_obj_file_path=self.config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)