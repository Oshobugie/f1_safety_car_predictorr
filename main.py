"""End-to-end training pipeline entrypoint for the F1 Safety Car Detector."""

import sys
from src.logger import logging
from src.exception import CustomException
from src.config.configuration import ConfigurationManager
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


def main():
    """Orchestrates sequential execution of ingestion, transformation, and model training."""
    try:
        logging.info(">>>>>> F1 Safety Car Training Pipeline Started <<<<<<")

        # Load centralized YAML parameters and provision artifact paths
        config_manager = ConfigurationManager()

        # Stage 1: Fetch raw 2023 lap telemetry from Ergast API and flatten to CSV
        logging.info(">>> Stage 1: Data Ingestion <<<")
        ingestion_config = config_manager.get_data_ingestion_config()
        data_ingestion = DataIngestion(config=ingestion_config)
        ingestion_artifact = data_ingestion.initiate_data_ingestion()

        # Stage 2: Engineer target labels, strip leakage keys, and fit preprocessing pipeline
        logging.info(">>> Stage 2: Data Transformation <<<")
        transform_config = config_manager.get_data_transformation_config()
        data_transformation = DataTransformation(config=transform_config)
        transform_artifact = data_transformation.initiate_data_transformation(ingestion_artifact)

        # Stage 3: Train imbalanced XGBoost classifier, log metrics to MLflow, and export model
        logging.info(">>> Stage 3: Model Training <<<")
        trainer_config = config_manager.get_model_trainer_config()
        model_trainer = ModelTrainer(config=trainer_config)
        trainer_artifact = model_trainer.initiate_model_trainer(transform_artifact)

        logging.info(">>>>>> Training Pipeline Completed Successfully <<<<<<")

    except Exception as e:
        logging.error(f"Pipeline execution failed: {e}")
        raise CustomException(e, sys)


if __name__ == "__main__":
    main()