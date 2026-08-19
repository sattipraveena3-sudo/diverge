from diverge.data import generate_synthetic_record
from diverge.detection import predict_detector, threshold_baseline, train_detector
from diverge.features import build_features


def test_model_returns_probabilities():
    frame = build_features(generate_synthetic_record(seed=3).frame)
    model = train_detector(frame)
    pred = predict_detector(model, frame)
    assert pred.confidence.min() >= 0
    assert pred.confidence.max() <= 1
    assert len(pred.risk) == len(frame)


def test_baseline_shape():
    frame = build_features(generate_synthetic_record(seed=4).frame)
    pred = threshold_baseline(frame)
    assert len(pred.risk) == len(frame)
