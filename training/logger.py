"""Experiment result logging and serialization."""

import os
import json
import time
from datetime import datetime
from typing import Dict, Any, List
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class ExperimentLogger:
    """Manages recording and exporting experiment results to CSV and JSON."""

    def __init__(self, results_dir: str = "./results", dataset_name: str = "experiment"):
        self.results_dir = results_dir
        self.raw_dir = os.path.join(results_dir, "raw")
        self.logs_dir = os.path.join(results_dir, "logs")
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        
        self.dataset_name = dataset_name
        self.experiment_id = self._generate_unique_experiment_id()
        self.records: List[Dict[str, Any]] = []

    def _generate_unique_experiment_id(self) -> str:
        """Create a collision-free experiment identifier."""
        now = datetime.now()
        date_str = now.strftime("%Y_%m_%d")
        
        # Check existing files to determine sequence number
        existing = [
            f for f in os.listdir(self.raw_dir)
            if f.startswith(f"{self.dataset_name}_{date_str}") and f.endswith(".csv")
        ]
        seq = len(existing) + 1
        return f"{self.dataset_name}_{date_str}_{seq:03d}"

    def log_run(self, record: Dict[str, Any]) -> None:
        """Add a single run result dictionary to the logger."""
        record_with_id = {"experiment_id": self.experiment_id, **record}
        self.records.append(record_with_id)
        logger.info(
            "Logged run: Method=%s, SubsetRatio=%.2f, Seed=%s, TestAcc=%.4f, TotalTime=%.2fs",
            record.get("method"),
            record.get("subset_ratio", 1.0),
            record.get("seed"),
            record.get("accuracy", 0.0),
            record.get("total_time", 0.0),
        )

    def save(self, config: Dict[str, Any]) -> str:
        """Save all logged records to CSV and full metadata to JSON.
        
        Returns:
            The filepath of the saved CSV.
        """
        if not self.records:
            logger.warning("No records to save.")
            return ""

        df = pd.DataFrame(self.records)
        csv_path = os.path.join(self.raw_dir, f"{self.experiment_id}.csv")
        json_path = os.path.join(self.raw_dir, f"{self.experiment_id}.json")

        df.to_csv(csv_path, index=False)
        
        meta = {
            "experiment_id": self.experiment_id,
            "timestamp": datetime.now().isoformat(),
            "config": config,
            "results_count": len(self.records),
            "records": self.records,
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        logger.info("Saved experiment results to CSV: %s", csv_path)
        logger.info("Saved experiment metadata to JSON: %s", json_path)
        return csv_path
