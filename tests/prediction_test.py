from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def mock_model():
    import numpy as np

    mock_model_instance = MagicMock()
    # predict_proba returns 2D array: [[prob_class_0, prob_class_1]]
    mock_model_instance.predict_proba.return_value = np.array([[0.3, 0.7]])
    return mock_model_instance


@pytest.fixture
def mock_mlflow_client():
    mock_client = MagicMock()
    mock_version_details = MagicMock()
    mock_version_details.run_id = "test_run_id_12345"
    mock_client.get_model_version.return_value = mock_version_details
    return mock_client


@pytest.fixture
def mock_mlflow_server(mock_model):
    mock_mlflow_server = MagicMock()
    mock_mlflow_server.sklearn.load_model.return_value = mock_model
    # Patch mlflow with the mock_mlflow_server (not mock_model!)
    with patch("src.router.predict.mlflow", mock_mlflow_server):
        yield mock_mlflow_server


def test_get_model(mock_mlflow_server, mock_mlflow_client, mock_model):
    # Mock MlflowClient to avoid network calls
    with patch("src.router.predict.MlflowClient", return_value=mock_mlflow_client):
        from src.router.predict import get_model

        # get_model() returns a tuple: (model, run_id)
        model, run_id = get_model()

        assert model == mock_model
        assert run_id == "test_run_id_12345"

        # Test that the model's predict method works
        import numpy as np

        result = model.predict_proba(
            [
                [
                    25,
                    "Engineer",
                    "Male",
                    "iOS",
                    "viettel",
                    0,
                    0,
                    1,
                    0,
                    0,
                    0,
                    0,
                    0,
                    1,
                    2,
                    0,
                ]
            ]
        )
        assert np.array_equal(result, np.array([[0.3, 0.7]]))


def test_predict_endpoint(mock_mlflow_server, mock_mlflow_client, mock_model):
    # Mock MlflowClient to avoid network calls
    with patch("src.router.predict.MlflowClient", return_value=mock_mlflow_client):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from src.router.predict import router

        app = FastAPI()
        app.include_router(router)
        client = TestClient(app)
        # response body
        response = client.post(
            "/credit_risk/predict",
            json={
                "Age": 30,
                "occupation": "student",
                "gender": "male",
                "operating_system": "iOS",
                "phone_provider": "viettel",
                "ENQ_3M": 0,
                "has_group2_debt_12m": 0,
                "NUM_CC_NON_BANK": 1,
                "NUM_NEW_LOAN_12M": 0,
                "MID_TERM_COUNT_NON_BANK": 0,
                "NUM_NEW_LOAN_6M": 0,
                "OUTS_BAL_LOAN_M1": 0,
                "LONG_TERM_AMOUNT": 0,
                "ENQ_9M": 1,
                "NUM_CC_BANK": 2,
                "NUM_NEW_LOAN_3M": 0,
            },
        )

        assert response.status_code == 200
        response_data = response.json()
        assert response_data["FPD10_plus_probability"] == 0.7
        assert response_data["model_name"] == "credit_risk_FPD10_plus"
        assert response_data["model_version"] == "3"
        assert response_data["model_run_id"] == "test_run_id_12345"
