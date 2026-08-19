import numpy as np

from diverge.evaluation import evaluate_predictions, first_alert_time


def test_first_alert_requires_consecutive_points():
    t = np.arange(6)
    risk = np.array([0, 1, 0, 1, 1, 1], dtype=bool)
    assert first_alert_time(t, risk, min_consecutive=3) == 3.0


def test_lead_time_calculation():
    t = np.arange(10)
    y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    risk = np.array([0, 0, 1, 1, 1, 1, 1, 1, 1, 1], dtype=bool)
    m = evaluate_predictions(t, y, risk, event_start_s=8)
    assert m.alert_time_s == 2.0
    assert m.lead_time_s == 6.0
