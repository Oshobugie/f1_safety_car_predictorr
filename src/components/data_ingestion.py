import sys
import json
import time
import requests
import pandas as pd
from pathlib import Path
from src.logger import logging
from src.exception import CustomException
from src.entity.config_entity import DataIngestionConfig
from src.entity.artifact_entity import DataIngestionArtifact


class DataIngestion:
    """Handles extracting historical F1 lap telemetry and persisting structured tabular data."""

    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_race_data(self) -> Path:
        """Fetches raw lap-by-lap race JSON payloads from the Ergast F1 API."""
        try:
            logging.info("Starting lap data ingestion from Ergast F1 API (2023 Season)...")
            raw_file_path = self.config.raw_data_dir / "races_raw.json"

            all_laps_data = []

            # Iterate across all 22 Grand Prix rounds in the 2023 calendar
            for round_num in range(1, 23):
                url = f"{self.config.dataset_download_url}/f1/2023/{round_num}/laps.json?limit=1000"
                logging.info(f"Fetching data for Round {round_num}...")

                response = requests.get(url, timeout=20)

                # Throttle requests to stay within Ergast API rate limits
                time.sleep(2)

                response.raise_for_status()
                race_data = response.json()

                # Unwrap the standard Ergast MRData response envelope
                races_list = race_data.get("MRData", {}).get("RaceTable", {}).get("Races", [])

                if races_list:
                    all_laps_data.extend(races_list)

            # Persist raw JSON payload before transformation for reproducibility
            with open(raw_file_path, "w") as f:
                json.dump(all_laps_data, f)

            logging.info(f"Raw lap data successfully saved to {raw_file_path}")
            return raw_file_path

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        """Executes download and normalizes nested lap timings into a flat CSV artifact."""
        try:
            raw_path = self.download_race_data()

            with open(raw_path, "r") as f:
                races_list = json.load(f)

            logging.info("Normalizing nested JSON lap data...")

            # Flatten hierarchical race -> lap -> driver timing records into a tabular schema
            df = pd.json_normalize(
                races_list,
                record_path=["Laps", "Timings"],  # Expand per-driver lap timing entries
                meta=[
                    "season",
                    "round",
                    "raceName",
                    ["Laps", "number"]  # Retain parent lap number context on each row
                ],
                errors="ignore"
            )

            # Standardize nested column naming
            df.rename(columns={"Laps.number": "lap_number"}, inplace=True)

            processed_path = self.config.ingested_data_dir / "races_ingested.csv"
            df.to_csv(processed_path, index=False)

            logging.info(f"Data Ingestion Stage completed successfully. Grid shape: {df.shape}")

            return DataIngestionArtifact(
                raw_data_file_path=raw_path,
                ingested_data_file_path=processed_path
            )
        except Exception as e:
            raise CustomException(e, sys)