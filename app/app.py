import streamlit as st
import requests
import os

st.set_page_config(page_title="AI Fraud Detection", layout="centered")

st.title("AI-Powered Fraud Detection System")
st.markdown("### Enter Transaction Details")

# Read API URL from env var, fall back to localhost for local dev
API_URL = os.getenv("FRAUD_API_URL", "http://127.0.0.1:8001")

EXPECTED_FEATURES = 30  # time(1) + V1-V28(28) + amount(1)

# -------------------------
# USER INPUTS
# -------------------------
amount = st.number_input("Transaction Amount", min_value=0.0, value=100.0)
time = st.number_input("Transaction Time (seconds from first)", min_value=0.0, value=0.0)

transaction_type = st.selectbox("Transaction Type", ["Online", "POS", "ATM"])
country = st.selectbox("Country", ["India", "USA", "UK", "Other"])


# -------------------------
# FEATURE GENERATION
# -------------------------
def generate_features(amount: float, time: float) -> list[float]:
    """
    Produces a 30-element feature vector matching the trained model.
    V1-V28 are PCA-transformed in production; here we simulate them.
    Replace this logic with your real PCA transform if available.
    """
    v_features = [0.0] * 28

    # Simple heuristic for demo realism — replace with real PCA
    if amount > 2000:
        v_features[0] = -2.0
        v_features[3] = 3.5

    feature_vector = [time] + v_features + [amount]

    # Guard: surface mismatches immediately instead of sending bad data
    assert len(feature_vector) == EXPECTED_FEATURES, (
        f"Feature vector length {len(feature_vector)} != {EXPECTED_FEATURES}"
    )

    return feature_vector


# -------------------------
# PREDICT
# -------------------------
if st.button("Analyze Transaction"):
    try:
        features = generate_features(amount, time)
    except AssertionError as e:
        st.error(f"Feature generation error: {e}")
        st.stop()

    try:
        response = requests.post(
            f"{API_URL}/predict",
            json={"features": features},
            timeout=10,
        )
        response.raise_for_status()          # raises on 4xx / 5xx
        result = response.json()

    except requests.exceptions.ConnectionError:
        st.error(f"Cannot reach API at {API_URL}. Is the FastAPI server running?")
        st.stop()
    except requests.exceptions.Timeout:
        st.error("Request timed out. The API took too long to respond.")
        st.stop()
    except requests.exceptions.HTTPError as e:
        # Surface the detail message returned by FastAPI
        detail = response.json().get("detail", str(e))
        st.error(f"API error {response.status_code}: {detail}")
        st.stop()

    fraud = result["fraud"]
    probability = result.get("probability", 0.5)

    st.markdown("## Result")

    if fraud == 1:
        st.error("High Risk: Fraudulent Transaction Detected")
    else:
        st.success("Low Risk: Transaction Appears Legitimate")

    st.markdown("### Risk Score")
    st.progress(int(probability * 100))
    st.write(f"Fraud probability: **{probability:.2%}**")