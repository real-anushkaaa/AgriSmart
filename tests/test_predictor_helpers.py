import pytest

from app.ml.disease_predictor import load_disease_model, predict_disease
from app.ml.errors import InvalidImageError, ModelUnavailableError
from app.ml.yield_predictor import predict_crop_yield


def test_disease_predictor_rejects_empty_image_before_tensorflow():
    with pytest.raises(InvalidImageError, match="empty"):
        predict_disease(b"", "missing-model.keras", "missing-classes.json")


def test_disease_model_loader_reports_missing_artifacts():
    with pytest.raises(ModelUnavailableError, match="missing"):
        load_disease_model("/missing/model.keras", "/missing/classes.json")


def test_yield_predictor_strips_categorical_inputs(monkeypatch):
    seen = {}

    class Model:
        def predict(self, frame):
            seen["frame"] = frame
            return [3.25]

    monkeypatch.setattr("app.ml.yield_predictor.get_yield_model", lambda: Model())

    prediction = predict_crop_yield(
        crop=" Coconut ",
        crop_year=2020,
        season=" Whole Year ",
        state=" Kerala ",
        area=1,
        annual_rainfall=1,
        fertilizer=1,
        pesticide=1,
    )

    assert prediction == 3.25
    frame = seen["frame"]
    assert frame.loc[0, "Crop"] == "Coconut"
    assert frame.loc[0, "Season"] == "Whole Year"
    assert frame.loc[0, "State"] == "Kerala"
