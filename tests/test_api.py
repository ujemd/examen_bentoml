"""Tests for the admission prediction service.

They call the running service over HTTP (the Docker container or
`bentoml serve`), so they do not need the source code.

Optional environment variables:
  API_URL          base URL of the service (default http://127.0.0.1:3000)
  JWT_SECRET_KEY   secret used by the service (default: the service's default)
"""
import os
from datetime import datetime, timedelta, timezone

import jwt
import pytest
import requests

BASE_URL = os.getenv("API_URL", "http://127.0.0.1:3000")
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "change-me-exercise-only-secret-key")
JWT_ALGORITHM = "HS256"
TIMEOUT = 10

VALID_USER = {"username": "user123", "password": "password123"}
VALID_INPUT = {
    "gre_score": 337,
    "toefl_score": 118,
    "university_rating": 4,
    "sop": 4.5,
    "lor": 4.5,
    "cgpa": 9.65,
    "research": 1,
}
WEAK_INPUT = {
    "gre_score": 290,
    "toefl_score": 95,
    "university_rating": 1,
    "sop": 1.5,
    "lor": 2.0,
    "cgpa": 6.8,
    "research": 0,
}


def make_token(secret=JWT_SECRET_KEY, expires_in=timedelta(hours=1), user="user123"):
    payload = {"sub": user, "exp": datetime.now(timezone.utc) + expires_in}
    return jwt.encode(payload, secret, algorithm=JWT_ALGORITHM)


def login(credentials):
    return requests.post(
        f"{BASE_URL}/login", json={"credentials": credentials}, timeout=TIMEOUT
    )


def predict(payload, token=None, authorization=None):
    headers = {}
    if authorization is not None:
        headers["Authorization"] = authorization
    elif token is not None:
        headers["Authorization"] = f"Bearer {token}"
    return requests.post(
        f"{BASE_URL}/predict",
        json={"input_data": payload},
        headers=headers,
        timeout=TIMEOUT,
    )


@pytest.fixture(scope="session", autouse=True)
def service_is_running():
    try:
        login(VALID_USER)
    except requests.exceptions.ConnectionError:
        pytest.exit(f"The service is not reachable at {BASE_URL}. Start it first.", returncode=1)


@pytest.fixture(scope="session")
def valid_token():
    response = login(VALID_USER)
    assert response.status_code == 200
    return response.json()["token"]


# --- Login API ---------------------------------------------------------------

def test_login_returns_valid_jwt_for_correct_credentials():
    response = login(VALID_USER)
    assert response.status_code == 200
    token = response.json()["token"]
    payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
    assert payload["sub"] == "user123"
    assert payload["exp"] > datetime.now(timezone.utc).timestamp()


def test_login_rejects_wrong_password():
    response = login({"username": "user123", "password": "wrong"})
    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid credentials"


def test_login_rejects_unknown_user():
    response = login({"username": "nobody", "password": "password123"})
    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid credentials"


def test_login_rejects_missing_fields():
    response = login({"username": "user123"})
    assert response.status_code in (400, 422)


# --- JWT authentication ------------------------------------------------------

def test_predict_without_token_is_rejected():
    response = predict(VALID_INPUT)
    assert response.status_code == 401
    assert "token" in response.json()["detail"].lower()


@pytest.mark.parametrize("header", ["abc", "Bearer", "Basic abc123"])
def test_predict_with_malformed_authorization_header_is_rejected(header):
    response = predict(VALID_INPUT, authorization=header)
    assert response.status_code == 401


def test_predict_with_invalid_token_is_rejected():
    response = predict(VALID_INPUT, token="not.a.valid-token")
    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid token"


def test_predict_with_token_signed_with_wrong_secret_is_rejected():
    response = predict(VALID_INPUT, token=make_token(secret="another-secret-key-not-used-by-service"))
    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid token"


def test_predict_with_expired_token_is_rejected():
    response = predict(VALID_INPUT, token=make_token(expires_in=timedelta(seconds=-60)))
    assert response.status_code == 401
    assert response.json()["detail"] == "Token has expired"


def test_predict_with_valid_token_succeeds(valid_token):
    response = predict(VALID_INPUT, token=valid_token)
    assert response.status_code == 200


# --- Prediction API ----------------------------------------------------------

def test_prediction_has_expected_format(valid_token):
    response = predict(VALID_INPUT, token=valid_token)
    assert response.status_code == 200
    prediction = response.json()["prediction"]
    assert isinstance(prediction, list) and len(prediction) == 1
    assert isinstance(prediction[0], float)
    assert 0.0 <= prediction[0] <= 1.0


def test_strong_candidate_scores_higher_than_weak_candidate(valid_token):
    strong = predict(VALID_INPUT, token=valid_token).json()["prediction"][0]
    weak = predict(WEAK_INPUT, token=valid_token).json()["prediction"][0]
    assert strong > weak


@pytest.mark.parametrize(
    "field, value",
    [
        ("gre_score", 400),
        ("toefl_score", -5),
        ("university_rating", 0),
        ("cgpa", 11),
        ("research", 2),
        ("gre_score", "not a number"),
    ],
)
def test_invalid_input_returns_an_error(valid_token, field, value):
    payload = {**VALID_INPUT, field: value}
    response = predict(payload, token=valid_token)
    assert response.status_code in (400, 422)
    assert "prediction" not in response.json()


def test_missing_field_returns_an_error(valid_token):
    payload = {k: v for k, v in VALID_INPUT.items() if k != "cgpa"}
    response = predict(payload, token=valid_token)
    assert response.status_code in (400, 422)
    assert "prediction" not in response.json()