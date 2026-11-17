import yaml
from pathlib import Path
from src.entity.config_entity import (
    DataIngestionConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
)


class ConfigurationManager:
    """Loads pipeline parameters from YAML and provisions typed configuration dataclasses."""

    def __init__(self, params_filepath: Path = Path("params.yaml")):
        """Reads the central YAML configuration file into memory."""
        with open(params_filepath, "r") as f:
            self.params = yaml.safe_load(f)

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        """Prepares artifact directories and returns settings for the data ingestion stage."""
        config = self.params["data_ingestion"]

        # Pre-create output folders so downstream file writes never fail on missing paths
        Path(config["raw_data_dir"]).mkdir(parents=True, exist_ok=True)
        Path(config["ingested_data_dir"]).mkdir(parents=True, exist_ok=True)

        return DataIngestionConfig(
            dataset_download_url=config["dataset_download_url"],
            raw_data_dir=Path(config["raw_data_dir"]),
            ingested_data_dir=Path(config["ingested_data_dir"])
        )

    def get_data_transformation_config(self) -> DataTransformationConfig:
        """Prepares artifact directories and returns settings for preprocessing and feature scaling."""
        config = self.params["data_transformation"]

        Path(config["transformed_data_dir"]).mkdir(parents=True, exist_ok=True)

        return DataTransformationConfig(
            transformed_data_dir=Path(config["transformed_data_dir"]),
            transformed_data_file_path=Path(config["transformed_data_file_path"]),
            preprocessor_obj_file_path=Path(config["preprocessor_obj_file_path"])
        )

    def get_model_trainer_config(self) -> ModelTrainerConfig:
        """Prepares artifact directories and returns settings for model training and export."""
        config = self.params["model_trainer"]

        Path(config["trained_model_dir"]).mkdir(parents=True, exist_ok=True)

        return ModelTrainerConfig(
            trained_model_dir=Path(config["trained_model_dir"]),
            trained_model_file_path=Path(config["trained_model_file_path"])
        )