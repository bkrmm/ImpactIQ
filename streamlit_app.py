# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import requests
from io import BytesIO
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

# Set up the page layout and title
st.set_page_config(layout="wide")
st.title("ImpactIQ - Customer Acquisition Attribution Analysis")
st.markdown("By Bikramjeet Singh Bedi, In Synapses'25 Hackathon by IIT Roorkee")

@st.cache_resource(show_spinner=True)
def train_model():
    try:
        st.info("Downloading dataset from Criteo...")
        # Use HTTPS if available
        url = "https://go.criteo.net/criteo-research-uplift-v2.1.csv.gz"

        with st.spinner('Downloading dataset...'):
            response = requests.get(url, stream=True)
            response.raise_for_status()  # Ensure we got a valid response
            
            # Limit download size by reading a limited number of rows (sample to speed up training)
            data = pd.read_csv(BytesIO(response.content), compression="gzip", nrows=100000)
            st.success("✅ Dataset downloaded successfully!")

        # Drop unneeded columns and missing rows
        data = data.drop(columns=["visit"], errors="ignore")
        data = data.dropna()

        # If needed, further reduce data for deployment speed (uncomment the next line if necessary)
        # data = data.sample(n=50000, random_state=42)

        # Split into label and features
        y = data["conversion"]
        X = data.drop(columns=["conversion"])

        # Scale features
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

        # Split into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, stratify=y, random_state=42
        )

        # Train XGBoost model with safe settings
        model = xgb.XGBClassifier(
            use_label_encoder=False, 
            eval_metric="logloss",
            scale_pos_weight=0.1
        )
        model.fit(X_train, y_train)
        
        # Create a SHAP explainer on a sample of training data
        try:
            sample_size = min(1000, len(X_train))
            explainer = shap.TreeExplainer(model, data=X_train.sample(sample_size, random_state=1))
        except Exception as e:
            st.warning(f"SHAP explainer creation failed: {str(e)}")
            explainer = None

        # Get feature names and calculate feature ranges for UI sliders
        feature_names = X.columns.tolist()
        feature_ranges = {}
        for col in X.columns:
            try:
                min_val = float(X[col].min())
                max_val = float(X[col].max())
                if min_val == max_val:
                    min_val -= 1
                    max_val += 1
                feature_ranges[col] = (min_val, max_val)
            except Exception as e:
                feature_ranges[col] = (-1, 1)  # Fallback range

        # Calculate AUC on the test set and display
        if model is not None:
            auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
            st.success(f"✅ Model trained. Test AUC = {auc:.4f}")
        
        return model, scaler, explainer, feature_names, feature_ranges

    except requests.RequestException as e:
        st.error(f"Failed to download dataset: {str(e)}")
        return None, StandardScaler(), None, [], {}
    except Exception as e:
        st.error(f"Error in model training: {str(e)}")
        return None, StandardScaler(), None, [], {}

# Attempt to train and unpack the model and components with error handling
try:
    model, scaler, explainer, feature_names, feature_ranges = train_model()
    if model is None:
        st.error("Model training failed. Please check your data and try again.")
        st.stop()
except Exception as e:
    st.error(f"Unexpected error during initialization: {str(e)}")
    st.stop()

# -------------------------------
# Sidebar: User Input for Attribution
# -------------------------------
st.sidebar.header("🔧 Input Features")

# Reset button for sliders
if st.sidebar.button("Reset All Features"):
    for k in st.session_state.keys():
        if k.startswith("slider_"):
            del st.session_state[k]

user_input = {}
for feature in feature_names:
    try:
        min_val, max_val = feature_ranges.get(feature, (-1, 1))
        default = float((min_val + max_val) / 2)
        # Ensure values are within bounds
        min_val = max(-1e6, min_val)
        max_val = min(1e6, max_val)
        default = np.clip(default, min_val, max_val)
        user_input[feature] = st.sidebar.slider(
            feature,
            min_value=float(min_val),
            max_value=float(max_val),
            value=float(default),
            key=f"slider_{feature}",
            step=0.01 * (max_val - min_val)
        )
    except Exception as e:
        st.sidebar.error(f"Error with feature '{feature}': {str(e)}")
        user_input[feature] = 0.0

st.sidebar.markdown("---")
auto_update = st.sidebar.checkbox("Enable realtime updates", value=True)

# -------------------------------
# Main Area: Model Prediction
# -------------------------------
input_df = pd.DataFrame([user_input])
try:
    input_scaled = pd.DataFrame(scaler.transform(input_df), columns=feature_names)
except Exception as e:
    st.error(f"Error scaling input features: {str(e)}")
    st.stop()

pred_prob = model.predict_proba(input_scaled)[0, 1]
st.subheader("🎯 Predicted Conversion Probability")
col1, col2 = st.columns([1, 2])
with col1:
    st.metric(
        label="Conversion Probability",
        value=f"{pred_prob:.4f}",
        delta=f"{pred_prob - 0.0025:.4f}",
        delta_color="inverse"
    )

# -------------------------------
# SHAP Attribution for Input
# -------------------------------
st.subheader("📊 Feature Attribution (SHAP Values)")
try:
    shap_values = explainer.shap_values(input_scaled)
    shap_df = pd.DataFrame({
        "Feature": feature_names,
        "SHAP Value": shap_values[0]
    }).sort_values("SHAP Value", key=abs, ascending=False)
    st.dataframe(shap_df.style.background_gradient(cmap="coolwarm", subset=["SHAP Value"]), use_container_width=True)
except Exception as e:
    st.error(f"Error computing SHAP values: {str(e)}")

# -------------------------------
# Global SHAP Summary Plot
# -------------------------------
st.subheader("🌐 Global Feature Importance (SHAP Summary)")
try:
    background_data = explainer.data if hasattr(explainer, "data") else None
    if background_data is not None:
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.summary_plot(explainer.shap_values(background_data), background_data, plot_type="bar", show=False)
        st.pyplot(fig)
    else:
        st.warning("Global SHAP summary not available.")
except Exception as e:
    st.error(f"Error generating global SHAP summary plot: {str(e)}")
