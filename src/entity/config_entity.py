from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    dataset_download_url: str
    raw_data_dir: Path
    ingested_data_dir: Path

@dataclass(frozen=True)
class DataTransformationConfig:
    transformed_data_dir: Path
    transformed_data_file_path: Path
    preprocessor_obj_file_path: Path

@dataclass(frozen=True)
class ModelTrainerConfig:
    trained_model_dir: Path
    trained_model_file_path: Path