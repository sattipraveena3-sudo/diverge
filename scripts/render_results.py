import json
from pathlib import Path
p=Path("artifacts/evaluation.json")
if not p.exists(): raise SystemExit("Run scripts/evaluate.py first")
r=json.loads(p.read_text()); d=r["divergence"]; b=r["baseline"]
text=f"""# Results\n\n## Reproducible synthetic benchmark\n\nThese values come from the deterministic synthetic benchmark, not from a clinical cohort. They validate the engineering and evaluation path but do **not** establish clinical effectiveness.\n\n| Method | Precision | Recall | F1 | Mean lead time (s) |\n|---|---:|---:|---:|---:|\n| DiVerge | {d['precision']:.3f} | {d['recall']:.3f} | {d['f1']:.3f} | {d['lead_time_s']:.1f} |\n| Single-signal threshold | {b['precision']:.3f} | {b['recall']:.3f} | {b['f1']:.3f} | {b['lead_time_s']:.1f} |\n\nLead-time advantage: **{r['lead_time_advantage_s']:.1f} seconds**.\n"""; Path("docs/RESULTS.md").write_text(text)
