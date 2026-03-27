"""
XG3 Speedway GP — Training script.

Usage:
    python train.py

Trains on FIM Speedway GP data 1995-2025 and saves models to models/r0/.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def main() -> None:
    data_dir = "D:/codex/Data/motorsports/tier1/curated/speedway"
    models_dir = "models/r0"

    logger.info("Starting Speedway GP model training ...")
    logger.info("Data dir: %s", data_dir)
    logger.info("Models dir: %s", models_dir)

    from ml.trainer import train_and_save

    metrics = train_and_save(data_dir=data_dir, models_dir=models_dir)

    # Print summary
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(json.dumps(metrics, indent=2))
    print("=" * 60)

    # Save metrics to file
    metrics_path = Path(models_dir) / "training_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Metrics saved to %s", metrics_path)


if __name__ == "__main__":
    main()
