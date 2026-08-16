from diverge.data import generate_synthetic_record
from diverge.features import build_features
from diverge.detection import train_detector,predict_detector,threshold_baseline
def test_model_returns_probabilities():
    frame=build_features(generate_synthetic_record(seed=3).frame); model=train_detector(frame); pred=predict_detector(model,frame); assert pred.confidence.min()>=0; assert pred.confidence.max()<=1; assert len(pred.risk)==len(frame)
def test_baseline_shape():
    frame=build_features(generate_synthetic_record(seed=4).frame); pred=threshold_baseline(frame); assert len(pred.risk)==len(frame)
