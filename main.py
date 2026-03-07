from fastapi import FastAPI, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import Literal, Optional
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
import urllib.request
import json as _json

# ---------------------------------------------------------------------------
# Startup: load model + demo data once
# ---------------------------------------------------------------------------

MODEL_PATH = Path(__file__).parent / "churn_xgboost_model.pkl"
DATA_PATH  = Path(__file__).parent / "data" / "telco.csv"

model = joblib.load(MODEL_PATH)

app = FastAPI(
    title="ChurnGuard API",
    description="Churn prediction, batch scanning, and retention recommendations.",
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

MEDIAN_MONTHLY_CHARGE = 65.0

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

YesNo      = Literal["Yes", "No"]
YesNoNoSvc = Literal["Yes", "No", "No phone service", "No internet service"]


class CustomerInput(BaseModel):
    gender:           Literal["Male", "Female"]
    SeniorCitizen:    int   = Field(..., ge=0, le=1)
    Partner:          YesNo
    Dependents:       YesNo
    tenure:           int   = Field(..., ge=0)
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


class BatchCustomer(CustomerInput):
    """CustomerInput extended with a client-side customer identifier."""
    customer_id: str


class Recommendation(BaseModel):
    cause:  str
    action: str


class ChurnPrediction(BaseModel):
    churn_probability:      float
    risk_level:             Literal["Low", "Medium", "High"]
    churn_drivers:          list[str]
    recommendations:        list[Recommendation]
    recommendation_summary: str


class CustomerResult(BaseModel):
    customer_id:            str
    churn_probability:      float
    risk_level:             Literal["Low", "Medium", "High"]
    top_driver:             str
    recommendation_summary: str
    recommendations:        list[Recommendation]
    monthly_charges:        float
    tenure:                 int
    contract:               str


class BatchResponse(BaseModel):
    total_scanned:    int
    high_risk_count:  int
    medium_risk_count: int
    low_risk_count:   int
    customers:        list[CustomerResult]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_dataframe(data: CustomerInput) -> pd.DataFrame:
    row = data.model_dump()
    total_services       = sum(1 for col in SERVICE_COLS if row[col] == "Yes")
    row["TotalServices"] = total_services
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
    drivers: list[str]               = []
    recommendations: list[Recommendation] = []
    total_services = sum(1 for col in SERVICE_COLS if row[col] == "Yes")

    if row["Contract"] == "Month-to-month":
        drivers.append("Month-to-month contract (low switching cost)")
        recommendations.append(Recommendation(
            cause="Month-to-month contract",
            action="Offer a 1- or 2-year contract with a loyalty discount",
        ))

    if row["MonthlyCharges"] > MEDIAN_MONTHLY_CHARGE * 1.2:
        drivers.append("Monthly charges above average")
        discount = "15%" if prob >= 0.70 else "10%"
        recommendations.append(Recommendation(
            cause="High monthly charges",
            action=f"Offer {discount} discount on the next 3 months of billing",
        ))

    if total_services <= 2:
        drivers.append("Low service engagement (few add-ons)")
        recommendations.append(Recommendation(
            cause="Low service engagement",
            action="Offer a bundled plan with OnlineSecurity + TechSupport free for 2 months",
        ))

    if (row["InternetService"] == "Fiber optic"
            and row["OnlineSecurity"] == "No"
            and row["TechSupport"] == "No"):
        drivers.append("Fiber optic with no security or support add-ons")
        recommendations.append(Recommendation(
            cause="Unprotected Fiber optic plan",
            action="Offer a free 3-month trial of OnlineSecurity + TechSupport",
        ))

    if row["tenure"] <= 12:
        drivers.append("Short tenure — within early churn risk window")
        recommendations.append(Recommendation(
            cause="Short customer tenure",
            action="Assign a dedicated onboarding specialist and schedule a check-in call",
        ))

    if row["PaymentMethod"] == "Electronic check":
        drivers.append("Payment via electronic check (highest churn correlation)")
        recommendations.append(Recommendation(
            cause="Manual payment method",
            action="Incentivise switch to auto-pay with a $5/month bill credit",
        ))

    if row["Partner"] == "No" and row["Dependents"] == "No":
        drivers.append("Single-user account with no partner or dependents (less sticky)")

    if not drivers:
        drivers.append("No dominant individual risk factor identified")
        recommendations.append(Recommendation(
            cause="General retention",
            action="Send a proactive satisfaction survey with a loyalty reward offer",
        ))

    summary = " | ".join(r.action for r in recommendations[:2])
    return drivers, recommendations, summary


def row_to_customer_input(row: pd.Series) -> CustomerInput:
    """Convert a raw Telco DataFrame row into a CustomerInput object."""
    return CustomerInput(
        gender           = row["gender"],
        SeniorCitizen    = int(row["SeniorCitizen"]),
        Partner          = row["Partner"],
        Dependents       = row["Dependents"],
        tenure           = int(row["tenure"]),
        PhoneService     = row["PhoneService"],
        MultipleLines    = row["MultipleLines"],
        InternetService  = row["InternetService"],
        OnlineSecurity   = row["OnlineSecurity"],
        OnlineBackup     = row["OnlineBackup"],
        DeviceProtection = row["DeviceProtection"],
        TechSupport      = row["TechSupport"],
        StreamingTV      = row["StreamingTV"],
        StreamingMovies  = row["StreamingMovies"],
        Contract         = row["Contract"],
        PaperlessBilling = row["PaperlessBilling"],
        PaymentMethod    = row["PaymentMethod"],
        MonthlyCharges   = float(row["MonthlyCharges"]),
        TotalCharges     = float(row["TotalCharges"]),
    )


def build_customer_result(customer_id: str, customer: CustomerInput, prob: float) -> CustomerResult:
    drivers, recommendations, summary = analyze(customer, prob)
    return CustomerResult(
        customer_id            = customer_id,
        churn_probability      = round(prob, 4),
        risk_level             = risk_level(prob),
        top_driver             = drivers[0] if drivers else "Unknown",
        recommendation_summary = summary,
        recommendations        = recommendations,
        monthly_charges        = customer.MonthlyCharges,
        tenure                 = customer.tenure,
        contract               = customer.Contract,
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
def landing():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/form", include_in_schema=False)
def predict_form():
    return FileResponse(STATIC_DIR / "predict.html")


@app.get("/dashboard", include_in_schema=False)
def dashboard_page():
    return FileResponse(STATIC_DIR / "dashboard.html")


@app.get("/health", tags=["Health"])
def health():
    return {"status": "healthy"}


# ---------------------------------------------------------------------------
# Single prediction
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Batch prediction  (client sends their own customer list)
# ---------------------------------------------------------------------------

@app.post("/predict_batch", response_model=BatchResponse, tags=["Batch"])
def predict_batch(customers: list[BatchCustomer]):
    """
    Accept a list of customers with IDs, return predictions for all of them
    sorted by churn probability descending.
    """
    results: list[CustomerResult] = []

    for c in customers:
        df   = build_dataframe(c)
        prob = float(model.predict_proba(df)[:, 1][0])
        results.append(build_customer_result(c.customer_id, c, prob))

    results.sort(key=lambda x: x.churn_probability, reverse=True)

    high   = sum(1 for r in results if r.risk_level == "High")
    medium = sum(1 for r in results if r.risk_level == "Medium")
    low    = sum(1 for r in results if r.risk_level == "Low")

    return BatchResponse(
        total_scanned     = len(results),
        high_risk_count   = high,
        medium_risk_count = medium,
        low_risk_count    = low,
        customers         = results,
    )


# ---------------------------------------------------------------------------
# Demo scan  (loads Telco dataset, runs predictions, returns at-risk customers)
# ---------------------------------------------------------------------------

@app.get("/high_risk_customers", response_model=BatchResponse, tags=["Dashboard"])
def high_risk_customers(
    threshold: float = Query(default=0.5, ge=0.0, le=1.0, description="Minimum churn probability"),
    limit:     int   = Query(default=5000, ge=1, le=5000, description="Max customers to return"),
):
    """
    Scans the demo Telco dataset and returns all customers above the
    churn probability threshold, sorted by risk descending.
    """
    # Load & clean
    df = pd.read_csv(DATA_PATH)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna().reset_index(drop=True)

    customer_ids = df["customerID"].tolist()
    X = df.drop(columns=["customerID", "Churn"])

    # Engineered features
    X["TotalServices"]     = X[SERVICE_COLS].apply(lambda r: sum(1 for v in r if v == "Yes"), axis=1)
    X["ChargesPerService"] = X["MonthlyCharges"] / (X["TotalServices"] + 1)

    # Batch predict (fast — single matrix operation)
    probs = model.predict_proba(X)[:, 1]

    # Filter to at-risk only, then build results
    results: list[CustomerResult] = []
    for i, (cid, prob) in enumerate(zip(customer_ids, probs)):
        if prob >= threshold:
            customer = row_to_customer_input(X.iloc[i])
            results.append(build_customer_result(cid, customer, float(prob)))

    results.sort(key=lambda x: x.churn_probability, reverse=True)

    # Count across ALL results before applying display limit
    high   = sum(1 for r in results if r.risk_level == "High")
    medium = sum(1 for r in results if r.risk_level == "Medium")
    low    = sum(1 for r in results if r.risk_level == "Low")

    return BatchResponse(
        total_scanned     = len(df),
        high_risk_count   = high,
        medium_risk_count = medium,
        low_risk_count    = low,
        customers         = results[:limit],
    )


# ---------------------------------------------------------------------------
# LLM Explanation  (Ollama llama3.2:3b)
# ---------------------------------------------------------------------------

class LLMExplainRequest(BaseModel):
    """Flexible input — works from both the predict form and the dashboard."""
    risk_level:        str
    churn_probability: float
    churn_drivers:     list[str]
    recommendations:   list[Recommendation]
    # Optional customer profile fields
    tenure:            Optional[int]   = None
    contract:          Optional[str]   = None
    monthly_charges:   Optional[float] = None
    internet_service:  Optional[str]   = None
    payment_method:    Optional[str]   = None
    senior_citizen:    Optional[int]   = None
    partner:           Optional[str]   = None
    dependents:        Optional[str]   = None


class LLMExplainResponse(BaseModel):
    explanation: str
    model:       str = "llama3.2:3b"


@app.post("/llm_explain", response_model=LLMExplainResponse, tags=["AI"])
def llm_explain(request: LLMExplainRequest):
    """
    Calls local Ollama (llama3.2:3b) to produce a plain-English explanation
    of why this customer is at risk. The LLM reasons within the constraints
    of the rule-based drivers and actions — it never invents new ones.
    """
    r = request

    # Build customer profile text from whatever fields are available
    profile_lines = []
    if r.tenure           is not None: profile_lines.append(f"- Tenure: {r.tenure} months")
    if r.contract         is not None: profile_lines.append(f"- Contract: {r.contract}")
    if r.monthly_charges  is not None: profile_lines.append(f"- Monthly Charges: ${r.monthly_charges:.2f}")
    if r.internet_service is not None: profile_lines.append(f"- Internet Service: {r.internet_service}")
    if r.payment_method   is not None: profile_lines.append(f"- Payment Method: {r.payment_method}")
    if r.partner          is not None: profile_lines.append(f"- Partner: {r.partner}")
    if r.dependents       is not None: profile_lines.append(f"- Dependents: {r.dependents}")
    if r.senior_citizen   is not None: profile_lines.append(f"- Senior Citizen: {'Yes' if r.senior_citizen else 'No'}")

    profile_text  = "\n".join(profile_lines) if profile_lines else "- Profile not available"
    drivers_text  = "\n".join(f"- {d}" for d in r.churn_drivers)
    actions_text  = "\n".join(f"- {rec.cause}: {rec.action}" for rec in r.recommendations)

    prompt = f"""You are a customer retention advisor. A churn prediction model has flagged this telecom customer.

Customer Profile:
{profile_text}

Churn Risk: {r.risk_level} ({r.churn_probability:.0%} probability)

Risk Drivers identified:
{drivers_text}

Recommended retention actions:
{actions_text}

Give exactly 3 bullet points explaining why this specific customer is at risk and why the recommended actions make sense. Format each bullet as:
• [point here]

Be specific to this customer's profile. Do NOT invent new recommendations or contradict the actions above. No intro sentence, just the 3 bullets."""

    try:
        payload = _json.dumps({
            "model":   "llama3.2:3b",
            "prompt":  prompt,
            "stream":  False,
            "options": {"temperature": 0.3, "num_predict": 200},
        }).encode()

        req = urllib.request.Request(
            "http://localhost:11434/api/generate",
            data    = payload,
            headers = {"Content-Type": "application/json"},
            method  = "POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = _json.loads(resp.read())
        explanation = result["response"].strip()

    except Exception as e:
        explanation = f"AI explanation unavailable — make sure Ollama is running (`ollama serve`). Error: {e}"

    return LLMExplainResponse(explanation=explanation)
