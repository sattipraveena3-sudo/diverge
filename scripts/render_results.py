import json
from pathlib import Path

p = Path("artifacts/evaluation.json")
if not p.exists():
    raise SystemExit("Run scripts/evaluate.py first")
r = json.loads(p.read_text())
d = r["divergence"]
b = r["baseline"]
lines = [
    "# Results",
    "",
    "## Reproducible synthetic benchmark",
    "",
    (
        "These values come from the deterministic synthetic benchmark, not from a clinical "
        "cohort. They validate the engineering and evaluation path but do **not** establish "
        "clinical effectiveness."
    ),
    "",
    "| Method | Precision | Recall | F1 | Mean lead time (s) |",
    "|---|---:|---:|---:|---:|",
    (
        f"| DiVerge | {d['precision']:.3f} | {d['recall']:.3f} | {d['f1']:.3f} | "
        f"{d['lead_time_s']:.1f} |"
    ),
    (
        f"| Single-signal threshold | {b['precision']:.3f} | {b['recall']:.3f} | "
        f"{b['f1']:.3f} | {b['lead_time_s']:.1f} |"
    ),
    "",
    f"Lead-time advantage: **{r['lead_time_advantage_s']:.1f} seconds**.",
]
Path("docs/RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
