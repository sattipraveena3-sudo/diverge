import json
from pathlib import Path

from diverge.pipeline import train_test_demo

bundle = train_test_demo()
result = {
    "data_source": "synthetic",
    "divergence": bundle.divergence.to_dict(),
    "baseline": bundle.baseline.to_dict(),
    "lead_time_advantage_s": bundle.lead_time_advantage_s,
}
Path("artifacts").mkdir(exist_ok=True)
Path("artifacts/evaluation.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
