import numpy as np
import streamlit as st

# ─────────────────────────────────────────────
# REAL MODEL WEIGHTS FROM YOUR COLAB TRAINING
# ─────────────────────────────────────────────

# Learned by gradient descent
# [bias, age, total_purchase, account_manager, years, num_sites]
THETA = np.array([
    -1.4463355800539957,
     0.11715754109057566,
     0.04344411351146206,
     0.11952931874647747,
     0.3060937201286444,
     0.9163588787998899
])

# What StandardScaler learned from your training data
SCALER_MEAN = np.array([
    41.59861111111111,
    10086.20125,
    0.4777777777777778,
    5.272736111111111,
    8.615277777777777
])

SCALER_SCALE = np.array([
    5.977387785450625,
    2444.2491297026268,
    0.4995059287330894,
    1.2693524759263803,
    1.7608520445623779
])

# ─────────────────────────────────────────────
# HELPER FUNCTIONS (same as your notebook)
# ─────────────────────────────────────────────

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def predict_proba(features):
    # Step 1: Scale exactly like training data
    scaled = (features - SCALER_MEAN) / SCALER_SCALE
    # Step 2: Add bias term (the column of 1s from np.c_)
    with_bias = np.concatenate([[1], scaled])
    # Step 3: Predict using learned weights
    prob = sigmoid(np.dot(with_bias, THETA))
    return prob

# ─────────────────────────────────────────────
# STREAMLIT UI
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📡",
    layout="centered"
)

st.title("📡 Telecom Customer Churn Predictor")
st.markdown("Enter customer details below to predict whether they are likely to churn.")
st.divider()

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=80, value=42,
                          help="Customer's age in years")
    total_purchase = st.number_input("Total Purchase ($)", min_value=0.0,
                                     max_value=50000.0, value=10000.0, step=100.0,
                                     help="Total amount spent by the customer")
    account_manager = st.selectbox("Has Account Manager?", options=[0, 1],
                                   format_func=lambda x: "Yes" if x == 1 else "No",
                                   help="Whether customer has a dedicated account manager")

with col2:
    years = st.number_input("Years as Customer", min_value=0.0, max_value=15.0,
                            value=5.0, step=0.1,
                            help="How long they have been a customer")
    num_sites = st.number_input("Number of Sites", min_value=1, max_value=20,
                                value=8, help="Number of sites the customer uses")

st.divider()

if st.button("🔍 Predict Churn Risk", use_container_width=True, type="primary"):

    features = np.array([age, total_purchase, account_manager, years, num_sites])
    prob = predict_proba(features)
    prediction = 1 if prob >= 0.5 else 0

    st.subheader("Prediction Result")

    col_result, col_prob = st.columns(2)

    with col_result:
        if prediction == 1:
            st.error("⚠️ HIGH CHURN RISK")
            st.markdown("This customer is **likely to churn.**")
        else:
            st.success("✅ LOW CHURN RISK")
            st.markdown("This customer is **likely to stay.**")

    with col_prob:
        st.metric(label="Churn Probability", value=f"{prob:.1%}")

    st.progress(float(prob), text=f"Churn probability: {prob:.1%}")

    st.divider()
    st.subheader("What This Means")

    if prob >= 0.7:
        st.warning("🔴 Very high risk. Consider immediate retention — offer discount or dedicated support.")
    elif prob >= 0.5:
        st.warning("🟠 Moderate-high risk. Proactive outreach recommended.")
    elif prob >= 0.3:
        st.info("🟡 Low-moderate risk. Standard check-in recommended.")
    else:
        st.info("🟢 Low risk. Customer appears stable.")

    # Key driver callout
    st.divider()
    st.subheader("Key Risk Drivers")
    st.caption("Based on your model's learned weights:")
    d1, d2, d3 = st.columns(3)
    d1.metric("Num Sites Impact", f"{num_sites} sites", 
              delta="High risk" if num_sites > 10 else "Normal")
    d2.metric("Tenure Impact", f"{years:.1f} yrs",
              delta="High risk" if years > 7 else "Normal")
    d3.metric("Age Impact", f"{age} yrs",
              delta="Slightly higher" if age > 50 else "Normal")

st.divider()
st.caption("Built with Logistic Regression trained from scratch using NumPy · Deployed on Streamlit")
