from dataclasses import dataclass
from pathlib import Path
from typing import Dict

@dataclass(frozen=True)
class DataIngestionArtifact:
    raw_data_file_path: Path
    ingested_data_file_path: Path

@dataclass(frozen=True)
class DataTransformationArtifact:
    transformed_data_file_path: Path
    preprocessor_obj_file_path: Path

@dataclass(frozen=True)
class ModelTrainerArtifact:
    trained_model_file_path: Path
    metric_artifact: Dict[str, float]