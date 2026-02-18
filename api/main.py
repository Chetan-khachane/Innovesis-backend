import os
import pandas as pd
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from twilio.rest import Client

app = FastAPI()

# ---------------- CORS ----------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- ENV ----------------
TWILIO_SID = os.getenv("TWILIO_SID")
TWILIO_AUTH = os.getenv("TWILIO_AUTH")
TWILIO_NUMBER = os.getenv("TWILIO_NUMBER")  # WhatsApp sandbox number

twilio_client = Client(TWILIO_SID, TWILIO_AUTH) if TWILIO_SID else None

# Only these numbers will receive in hackathon demo
ALLOWED_NUMBERS = [
    "+918329391715",
    "+919867542729",
    "+918097783653",
    "+919152274885"
]

cluster_memory = {}

# ---------------- PREPROCESS ----------------
def preprocess(df):
    df = df.drop_duplicates()
    df["email"] = df["email"].astype(str).str.lower().str.strip()

    platform_cols = [
        "whatsapp_usage_minutes_per_week",
        "facebook_usage_minutes_per_week",
        "instagram_usage_minutes_per_week",
        "telegram_usage_minutes_per_week",
        "gmail_usage_minutes_per_week",
        "sms_usage_minutes_per_week"
    ]

    df["total_engagement"] = df[platform_cols].sum(axis=1)
    df["most_active_platform"] = df[platform_cols].idxmax(axis=1)

    return df

# ---------------- EXECUTE CAMPAIGN ----------------
@app.post("/execute-campaign")
async def execute_campaign(file: UploadFile = File(...)):

    df = pd.read_csv(file.file)
    df = preprocess(df)

    # Encode
    le_ins = LabelEncoder()
    le_event = LabelEncoder()

    df["insurance_encoded"] = le_ins.fit_transform(df["insurance_type"])
    df["life_event_encoded"] = le_event.fit_transform(df["life_event"])

    # Cluster
    cluster_features = df[["age", "income_lpa", "total_engagement"]]
    scaler_cluster = StandardScaler()
    cluster_scaled = scaler_cluster.fit_transform(cluster_features)

    kmeans = KMeans(n_clusters=4, random_state=42)
    df["cluster"] = kmeans.fit_predict(cluster_scaled)

    # Propensity
    model_features = df[
        ["age", "income_lpa", "total_engagement",
         "insurance_encoded", "life_event_encoded"]
    ]

    y = df["purchased"]

    scaler_model = StandardScaler()
    X_scaled = scaler_model.fit_transform(model_features)

    model = LogisticRegression(max_iter=500)
    model.fit(X_scaled, y)

    df["purchase_probability"] = model.predict_proba(X_scaled)[:, 1]

    segments = []

    for cid in df["cluster"].unique():

        segment_df = df[df["cluster"] == cid]

        dominant_insurance = (
            segment_df["insurance_type"]
            .value_counts()
            .idxmax()
        )

        customers = segment_df[
            ["name", "email", "phone_number", "purchase_probability"]
        ].to_dict(orient="records")

        cluster_memory[int(cid)] = customers

        segments.append({
            "cluster_id": int(cid),
            "customer_count": len(segment_df),
            "average_purchase_probability":
                round(float(segment_df["purchase_probability"].mean()), 4),
            "insurance_type": dominant_insurance,
            "recommended_channel": "whatsapp",
            "customers_preview": customers[:5]
        })

    return {
        "total_customers": len(df),
        "overall_expected_conversion":
            round(float(df["purchase_probability"].mean()), 4),
        "segments": segments
    }

# ---------------- SEND CAMPAIGN ----------------
@app.post("/send-campaign")
async def send_campaign(payload: dict):

    cluster_id = int(payload.get("cluster_id"))
    message = payload.get("message")
    channel = payload.get("channel", "whatsapp")

    if cluster_id not in cluster_memory:
        return {"error": "Cluster not found"}

    customers = cluster_memory[cluster_id]
    sent = 0
    failed = 0

    for c in customers:
        phone = str(c["phone_number"]).strip()

        if phone not in ALLOWED_NUMBERS:
            continue

        if not twilio_client:
            continue

        try:
            twilio_client.messages.create(
                body=message,
                from_=f"whatsapp:{TWILIO_NUMBER}",
                to=f"whatsapp:{phone}"
            )
            sent += 1
        except Exception:
            failed += 1

    return {
        "cluster_id": cluster_id,
        "messages_sent": sent,
        "failed": failed
    }

@app.get("/")
def root():
    return {"status": "Backend Running"}
