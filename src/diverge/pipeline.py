from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import generate_synthetic_record
from .detection import future_event_target, predict_detector, threshold_baseline, train_detector
from .evaluation import Metrics, evaluate_predictions
from .features import build_features


@dataclass
class EvaluationBundle:
    divergence: Metrics
    baseline: Metrics
    lead_time_advantage_s: float | None


def make_dataset(seeds: Iterable[int] | None = None):
    if seeds is None:
        seeds = range(30)
    return [build_features(generate_synthetic_record(seed=int(s)).frame) for s in seeds]


def train_test_demo(
    train_seeds: Iterable[int] | None = None,
    test_seeds: Iterable[int] | None = None,
):
    if train_seeds is None:
        train_seeds = range(20)
    if test_seeds is None:
        test_seeds = range(20, 30)
    train = pd.concat(make_dataset(train_seeds), ignore_index=True)
    model = train_detector(train)
    div_metrics = []
    base_metrics = []
    for seed in test_seeds:
        rec = generate_synthetic_record(seed=int(seed))
        frame = build_features(rec.frame)
        y = future_event_target(frame)
        div = predict_detector(model, frame)
        base = threshold_baseline(frame)
        div_metrics.append(
            evaluate_predictions(frame.time_s.to_numpy(), y, div.risk, rec.event_start)
        )
        base_metrics.append(
            evaluate_predictions(frame.time_s.to_numpy(), y, base.risk, rec.event_start)
        )

    def mean_metric(items):
        def avg(name):
            vals = [getattr(x, name) for x in items if getattr(x, name) is not None]
            return float(np.mean(vals)) if vals else None

        return Metrics(
            avg("precision") or 0.0,
            avg("recall") or 0.0,
            avg("f1") or 0.0,
            avg("alert_time_s"),
            avg("lead_time_s"),
        )

    dm, bm = mean_metric(div_metrics), mean_metric(base_metrics)
    adv = (
        None
        if dm.lead_time_s is None or bm.lead_time_s is None
        else dm.lead_time_s - bm.lead_time_s
    )
    return EvaluationBundle(dm, bm, adv)
