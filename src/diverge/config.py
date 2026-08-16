from dataclasses import dataclass
import os
@dataclass(frozen=True)
class Settings:
    data_source: str = os.getenv("DIVERGE_DATA_SOURCE", "synthetic")
    random_seed: int = int(os.getenv("DIVERGE_RANDOM_SEED", "42"))
    model_path: str = os.getenv("DIVERGE_MODEL_PATH", "artifacts/model.joblib")
    threshold: float = float(os.getenv("DIVERGE_THRESHOLD", "0.50"))
