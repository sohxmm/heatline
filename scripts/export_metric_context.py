"""Publish metric definitions and baselines calculated from saved confusion counts."""
from datetime import datetime, timezone
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def classification_context(row):
    tp, tn, fp, fn = (int(row[key]) for key in ("tp", "tn", "fp", "fn"))
    total = tp + tn + fp + fn
    if total == 0 or min(tp, tn, fp, fn) < 0:
        raise ValueError("Nonnegative counts and at least one observation required")
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    return dict(days=total, positive_days=tp + fn, tp=tp, tn=tn, fp=fp, fn=fn,
                accuracy=(tp + tn) / total,
                precision=tp / (tp + fp) if tp + fp else 0.0,
                recall=recall, specificity=specificity,
                balanced_accuracy=(recall + specificity) / 2,
                always_negative=dict(accuracy=(tn + fp) / total, recall=0.0,
                                     balanced_accuracy=0.5 if tp + fn and tn + fp else None),
                fair_coin=dict(expected_accuracy=0.5, expected_recall=0.5,
                               expected_balanced_accuracy=0.5,
                               note="Analytical expectation for independent fair binary guesses; not a fitted model or an observed run."))


def export():
    directory = ROOT / "artifacts/heatline"
    model = json.loads((directory / "model.json").read_text(encoding="utf-8"))
    nasa = json.loads((directory / "provider_benchmark.json").read_text(encoding="utf-8"))
    context = dict(schema_version="1.0", source="Saved Heatline v2 confusion counts",
                   mode="retrospective metric audit", issued_at=datetime.now(timezone.utc).isoformat(),
                   target_date=None, model_version=model["model_version"], availability="available",
                   timezone="Asia/Kolkata", years="May 2019–2026",
                   definitions=dict(accuracy="(TP + TN) / all days",
                                    precision="TP / (TP + FP): fraction of alerts that were correct",
                                    recall="TP / (TP + FN): fraction of positive days detected",
                                    balanced_accuracy="Mean of sensitivity and specificity"),
                   era5=classification_context(model["metrics"][0]),
                   nasa_f1=classification_context(nasa["metrics"][0]["balanced"]),
                   interpretation="29.4% is ERA5 alert precision, not accuracy. Accuracy alone conceals the rare positive class. NASA and ERA5 use different daily products and target labels; their scores do not establish which source matches station truth.")
    for target in (directory, ROOT / "frontend/public/data"):
        (target / "metric_context.json").write_text(json.dumps(context, indent=2) + "\n", encoding="utf-8")
    return context


if __name__ == "__main__":
    export()
