import pandas as pd
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from twilio.rest import Client
import os

app = FastAPI()

# --------------------------------------------------
# CORS
# --------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------
# Twilio ENV (Sandbox)
# --------------------------------------------------
TWILIO_SID = os.getenv("TWILIO_SID")
TWILIO_AUTH = os.getenv("TWILIO_AUTH")
TWILIO_NUMBER = os.getenv("TWILIO_NUMBER")  # Must be +14155238886

twilio_client = Client(TWILIO_SID, TWILIO_AUTH)

# --------------------------------------------------
# ONLY THESE NUMBERS WILL RECEIVE MESSAGE
# --------------------------------------------------
ALLOWED_NUMBERS = [
    "+918329391715",
    "+919867542729",
    "+918097783653",
    "+919152274885"
]

# --------------------------------------------------
# Memory Storage
# --------------------------------------------------
cluster_memory = {}

# --------------------------------------------------
# Preprocess
# --------------------------------------------------
def preprocess(df):
    df = df.drop_duplicates()

    df["email"] = df["email"].astype(str).str.lower().str.strip()
    df["phone_number"] = df["phone_number"].astype(str).str.strip()

    # Ensure +91 format
    df["phone_number"] = df["phone_number"].apply(
        lambda x: x if x.startswith("+") else "+91" + x
    )

    platform_cols = [
        "whatsapp_usage_minutes_per_week",
        "facebook_usage_minutes_per_week",
        "instagram_usage_minutes_per_week",
        "telegram_usage_minutes_per_week",
        "gmail_usage_minutes_per_week",
        "sms_usage_minutes_per_week"
    ]

    df["total_engagement"] = df[platform_cols].sum(axis=1)

    return df

# --------------------------------------------------
# EXECUTE CAMPAIGN
# --------------------------------------------------
@app.post("/execute-campaign")
async def execute_campaign(file: UploadFile = File(...)):

    df = pd.read_csv(file.file)
    df = preprocess(df)

    # Encode
    df["insurance_encoded"] = LabelEncoder().fit_transform(df["insurance_type"])
    df["life_event_encoded"] = LabelEncoder().fit_transform(df["life_event"])

    # Clustering
    cluster_features = df[["age", "income_lpa", "total_engagement"]]
    cluster_scaled = StandardScaler().fit_transform(cluster_features)

    kmeans = KMeans(n_clusters=4, random_state=42)
    df["cluster"] = kmeans.fit_predict(cluster_scaled)

    # Propensity
    model_features = df[[
        "age",
        "income_lpa",
        "total_engagement",
        "insurance_encoded",
        "life_event_encoded"
    ]]

    X_scaled = StandardScaler().fit_transform(model_features)

    model = LogisticRegression(max_iter=500)
    model.fit(X_scaled, df["purchased"])

    df["purchase_probability"] = model.predict_proba(X_scaled)[:, 1]

    segments = []

    for cid in sorted(df["cluster"].unique()):

        segment_df = df[df["cluster"] == cid]

        customers = segment_df[[
            "name",
            "email",
            "phone_number",
            "purchase_probability"
        ]].to_dict(orient="records")

        cluster_memory[int(cid)] = customers

        segments.append({
            "cluster_id": int(cid),
            "customer_count": len(segment_df),
            "average_purchase_probability":
                round(float(segment_df["purchase_probability"].mean()), 4),
            "recommended_channel": "whatsapp",
            "customers_preview": customers[:5]  # preview only
        })

    return {
        "total_customers": len(df),
        "segments": segments
    }

# --------------------------------------------------
# SEND CAMPAIGN (SAFE MODE)
# --------------------------------------------------
@app.post("/send-campaign")
async def send_campaign(payload: dict):

    cluster_id = payload.get("cluster_id")
    message = payload.get("message")

    if cluster_id not in cluster_memory:
        return {"error": "Cluster not found"}

    customers = cluster_memory[cluster_id]

    sent = 0
    failed = 0

    # ONLY send to teammate numbers
    for phone in ALLOWED_NUMBERS:

        try:
            msg = twilio_client.messages.create(
                body=message,
                from_="whatsapp:" + TWILIO_NUMBER,
                to="whatsapp:" + phone
            )

            print("Sent to:", phone, "SID:", msg.sid)
            sent += 1

        except Exception as e:
            print("Error:", str(e))
            failed += 1

    return {
        "cluster_id": cluster_id,
        "messages_sent": sent,
        "failed": failed,
        "channel": "whatsapp",
        "note": "Only teammate numbers targeted"
    }

# --------------------------------------------------
# HEALTH
# --------------------------------------------------
@app.get("/")
def root():
    return {"status": "Innovesis Hackathon Backend Live"}
