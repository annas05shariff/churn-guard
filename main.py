from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import Literal
from pathlib import Path
import joblib
import pandas as pd

# ---------------------------------------------------------------------------
# Startup: load model once
# ---------------------------------------------------------------------------

MODEL_PATH = Path(__file__).parent / "churn_xgboost_model.pkl"
model = joblib.load(MODEL_PATH)

app = FastAPI(
    title="Telco Churn Decision-Support API",
    description="Predicts customer churn probability and recommends targeted retention actions.",
    version="1.0.0",
)

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SERVICE_COLS = [
    "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
]

MEDIAN_MONTHLY_CHARGE = 65.0   # approximate dataset median

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

YesNo         = Literal["Yes", "No"]
YesNoNoSvc    = Literal["Yes", "No", "No phone service", "No internet service"]


class CustomerInput(BaseModel):
    gender:           Literal["Male", "Female"]
    SeniorCitizen:    int   = Field(..., ge=0, le=1, description="1 = senior citizen")
    Partner:          YesNo
    Dependents:       YesNo
    tenure:           int   = Field(..., ge=0, description="Months with the company")
    PhoneService:     YesNo
    MultipleLines:    YesNoNoSvc
    InternetService:  Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity:   YesNoNoSvc
    OnlineBackup:     YesNoNoSvc
    DeviceProtection: YesNoNoSvc
    TechSupport:      YesNoNoSvc
    StreamingTV:      YesNoNoSvc
    StreamingMovies:  YesNoNoSvc
    Contract:         Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: YesNo
    PaymentMethod:    Literal[
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)",
    ]
    MonthlyCharges:   float = Field(..., gt=0)
    TotalCharges:     float = Field(..., ge=0)

    model_config = {
        "json_schema_extra": {
            "example": {
                "gender": "Female", "SeniorCitizen": 0,
                "Partner": "Yes", "Dependents": "No", "tenure": 5,
                "PhoneService": "Yes", "MultipleLines": "No",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No", "OnlineBackup": "No",
                "DeviceProtection": "No", "TechSupport": "No",
                "StreamingTV": "No", "StreamingMovies": "No",
                "Contract": "Month-to-month", "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 70.35, "TotalCharges": 351.75,
            }
        }
    }


class Recommendation(BaseModel):
    cause:  str
    action: str


class ChurnPrediction(BaseModel):
    churn_probability:      float
    risk_level:             Literal["Low", "Medium", "High"]
    churn_drivers:          list[str]
    recommendations:        list[Recommendation]
    recommendation_summary: str

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_dataframe(data: CustomerInput) -> pd.DataFrame:
    row = data.model_dump()
    total_services = sum(1 for col in SERVICE_COLS if row[col] == "Yes")
    row["TotalServices"]     = total_services
    row["ChargesPerService"] = row["MonthlyCharges"] / (total_services + 1)
    return pd.DataFrame([row])


def risk_level(prob: float) -> str:
    if prob >= 0.60:
        return "High"
    if prob >= 0.35:
        return "Medium"
    return "Low"


def analyze(data: CustomerInput, prob: float) -> tuple[list[str], list[Recommendation], str]:
    row            = data.model_dump()
    drivers        = []
    recommendations: list[Recommendation] = []
    total_services = sum(1 for col in SERVICE_COLS if row[col] == "Yes")

    # --- Contract type -------------------------------------------------
    if row["Contract"] == "Month-to-month":
        drivers.append("Month-to-month contract (low switching cost)")
        recommendations.append(Recommendation(
            cause="Month-to-month contract",
            action="Offer a 1- or 2-year contract with a loyalty discount",
        ))

    # --- High monthly charges ------------------------------------------
    if row["MonthlyCharges"] > MEDIAN_MONTHLY_CHARGE * 1.2:
        drivers.append("Monthly charges above average")
        discount = "15%" if prob >= 0.70 else "10%"
        recommendations.append(Recommendation(
            cause="High monthly charges",
            action=f"Offer {discount} discount on the next 3 months of billing",
        ))

    # --- Low service engagement ----------------------------------------
    if total_services <= 2:
        drivers.append("Low service engagement (few add-ons)")
        recommendations.append(Recommendation(
            cause="Low service engagement",
            action="Offer a bundled plan with OnlineSecurity + TechSupport free for 2 months",
        ))

    # --- Fiber optic with no security/support --------------------------
    if (row["InternetService"] == "Fiber optic"
            and row["OnlineSecurity"] == "No"
            and row["TechSupport"] == "No"):
        drivers.append("Fiber optic with no security or support add-ons")
        recommendations.append(Recommendation(
            cause="Unprotected Fiber optic plan",
            action="Offer a free 3-month trial of OnlineSecurity + TechSupport",
        ))

    # --- Short tenure --------------------------------------------------
    if row["tenure"] <= 12:
        drivers.append("Short tenure — within early churn risk window")
        recommendations.append(Recommendation(
            cause="Short customer tenure",
            action="Assign a dedicated onboarding specialist and schedule a check-in call",
        ))

    # --- Risky payment method ------------------------------------------
    if row["PaymentMethod"] == "Electronic check":
        drivers.append("Payment via electronic check (highest churn correlation)")
        recommendations.append(Recommendation(
            cause="Manual payment method",
            action="Incentivise switch to auto-pay with a $5/month bill credit",
        ))

    # --- Low stickiness (no partner/dependents) ------------------------
    if row["Partner"] == "No" and row["Dependents"] == "No":
        drivers.append("Single-user account with no partner or dependents (less sticky)")

    # --- Fallback -------------------------------------------------------
    if not drivers:
        drivers.append("No dominant individual risk factor identified")
        recommendations.append(Recommendation(
            cause="General retention",
            action="Send a proactive satisfaction survey with a loyalty reward offer",
        ))

    summary = " | ".join(r.action for r in recommendations[:2])
    return drivers, recommendations, summary

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
def landing():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/form", include_in_schema=False)
def predict_form():
    return FileResponse(STATIC_DIR / "predict.html")


@app.get("/health", tags=["Health"])
def health():
    return {"status": "healthy"}


@app.post("/predict", response_model=ChurnPrediction, tags=["Prediction"])
def predict(customer: CustomerInput):
    df   = build_dataframe(customer)
    prob = float(model.predict_proba(df)[:, 1][0])

    drivers, recommendations, summary = analyze(customer, prob)

    return ChurnPrediction(
        churn_probability      = round(prob, 4),
        risk_level             = risk_level(prob),
        churn_drivers          = drivers,
        recommendations        = recommendations,
        recommendation_summary = summary,
    )
