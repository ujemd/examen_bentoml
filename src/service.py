"""BentoML service: JWT-protected prediction of the chance of admission."""
import os
import secrets
from datetime import datetime, timedelta, timezone

import bentoml
import jwt
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

# The secret can be overridden with an environment variable. The default is for
# this exercise only: never hardcode a real secret in production.
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "change-me-exercise-only-secret-key")
JWT_ALGORITHM = "HS256"
TOKEN_LIFETIME = timedelta(hours=1)

# Demo users (plain text on purpose).
USERS = {
    "user123": "password123",
    "user456": "password456",
}

# Same order as the columns used for training.
FEATURES = [
    "gre_score", "toefl_score", "university_rating", "sop", "lor", "cgpa", "research",
]


class Credentials(BaseModel):
    username: str
    password: str


class AdmissionInput(BaseModel):
    gre_score: int = Field(ge=0, le=340)
    toefl_score: int = Field(ge=0, le=120)
    university_rating: int = Field(ge=1, le=5)
    sop: float = Field(ge=0, le=5)
    lor: float = Field(ge=0, le=5)
    cgpa: float = Field(ge=0, le=10)
    research: int = Field(ge=0, le=1)


def create_jwt_token(user_id: str) -> str:
    payload = {"sub": user_id, "exp": datetime.now(timezone.utc) + TOKEN_LIFETIME}
    return jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)


class JWTAuthMiddleware(BaseHTTPMiddleware):
    """Runs in front of every route; only /predict requires a valid token."""

    async def dispatch(self, request, call_next):
        if request.url.path.endswith("/predict"):
            scheme, _, token = request.headers.get("Authorization", "").partition(" ")
            if scheme.lower() != "bearer" or not token:
                return JSONResponse(
                    status_code=401,
                    content={"detail": "Missing or malformed authentication token"},
                )
            try:
                payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
            except jwt.ExpiredSignatureError:
                return JSONResponse(status_code=401, content={"detail": "Token has expired"})
            except jwt.InvalidTokenError:
                return JSONResponse(status_code=401, content={"detail": "Invalid token"})
            request.state.user = payload.get("sub")
        return await call_next(request)


@bentoml.service
class AdmissionService:
    bento_model = bentoml.models.get("admission_lr:latest")

    def __init__(self) -> None:
        self.model = bentoml.sklearn.load_model(self.bento_model)

    @bentoml.api(route="/login")
    def login(self, credentials: Credentials):
        expected = USERS.get(credentials.username)
        if expected is not None and secrets.compare_digest(expected, credentials.password):
            return {"token": create_jwt_token(credentials.username)}
        return JSONResponse(status_code=401, content={"detail": "Invalid credentials"})

    @bentoml.api(route="/predict")
    def predict(self, input_data: AdmissionInput) -> dict:
        features = pd.DataFrame([input_data.model_dump()], columns=FEATURES)
        prediction = self.model.predict(features)
        # A chance is a probability, so keep it inside [0, 1].
        return {"prediction": [float(np.clip(p, 0.0, 1.0)) for p in prediction]}


AdmissionService.add_asgi_middleware(JWTAuthMiddleware)