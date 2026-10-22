# Admission Prediction Service

BentoML service that predicts the chance of admission of a student to a university
(linear regression), protected by JWT authentication.

Endpoints (port 3000):
- `POST /login`: body `{"credentials": {"username": "user123", "password": "password123"}}`, returns `{"token": "..."}`
- `POST /predict`: header `Authorization: Bearer <token>`, body `{"input_data": {"gre_score": 337, "toefl_score": 118, "university_rating": 4, "sop": 4.5, "lor": 4.5, "cgpa": 9.65, "research": 1}}`, returns `{"prediction": [0.95...]}`

Demo users: `user123` / `password123` and `user456` / `password456`.

## Contents of this archive

- `admission_prediction_service.tar`: Docker image of the service
- `requirements.txt`: dependencies (used here for the tests)
- `README.md`: this file
- `tests/`: pytest tests

## Prerequisites and versions

- Docker, with port 3000 free
- `curl`
- Python 3.10, 3.11 or 3.12 with the `venv` module, to run the tests. Python 3.13 or
  later is not supported, because `scikit-learn==1.3.2` has no wheels for it.

The service itself runs inside the Docker image with Python 3.10, so the Python version
of the host only matters for the tests.

Pinned versions (see `requirements.txt`): bentoml 1.4.35, scikit-learn 1.3.2,
pandas 2.3.3, numpy 1.26.4, pydantic 2.12.x, pyjwt 2.10.1, pytest 8.3.3,
requests 2.32.5. The model was trained with scikit-learn 1.3.2, so this version must
not be changed. Only pytest, requests and pyjwt are needed to run the tests.

## Commands

Run these commands in order, from the folder where the archive was extracted.

1. Load the Docker image:

```bash
docker load -i admission_prediction_service.tar
```

2. Run the service:

```bash
docker run -d --name admission_prediction_service -p 3000:3000 admission_prediction_service:latest
```

3. Wait until the service is ready:

```bash
until curl -sf http://127.0.0.1:3000/readyz > /dev/null; do sleep 2; done
```

4. Create a virtual environment and install the dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

5. Run the tests:

```bash
pytest -v
```

6. Stop and remove the container:

```bash
docker stop admission_prediction_service
docker rm admission_prediction_service
```

The tests call the running service over HTTP at `http://127.0.0.1:3000`.