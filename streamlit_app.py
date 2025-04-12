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

st.set_page_config(layout="wide")
st.title("ImapactIQ - Customer Acquisition Attribution Analysis")
st.markdown("By Bikramjeet Singh Bedi, In Synapses'25 Hackathon by IIT Roorkee")

@st.cache_resource
def train_model():
    try:
        st.info("Downloading dataset from Criteo...")
        # URL of the dataset
        url = "http://go.criteo.net/criteo-research-uplift-v2.1.csv.gz"
        
        # Download with progress bar
        with st.spinner('Downloading dataset...'):
            response = requests.get(url, stream=True)
            response.raise_for_status()  # Ensure we got a valid response
            
            # Load the data directly from the response content
            data = pd.read_csv(BytesIO(response.content), compression="gzip")
            st.success("✅ Dataset downloaded successfully!")

        # Drop columns
        data = data.sample(n=1000000, random_state=42)
        data = data.drop(columns=["visit"], errors="ignore")

        # Drop missing rows (or fillna if you prefer)
        data = data.dropna()

        # Limit size for speed
        #data = data.sample(n=1000000, random_state=42)

        # Split features and label
        y = data["conversion"]
        X = data.drop(columns=["conversion"])

        # Scale features
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

        # Split
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=42)

        # Initialize all return values with proper error handling
        model = None
        explainer = None
        feature_names = []
        feature_ranges = {}

        # Process data
        if not data.empty:
            # Train model with safeguards
            model = xgb.XGBClassifier(
                use_label_encoder=False, 
                eval_metric="logloss",
                scale_pos_weight=0.1
            )
            model.fit(X_train, y_train)
            
            # Create explainer safely
            try:
                explainer = shap.TreeExplainer(model, data=X_train.sample(min(1000, len(X_train)), random_state=1))
            except Exception as e:
                st.warning(f"SHAP explainer creation failed: {str(e)}")
                explainer = None
            
            # Get feature names
            feature_names = X.columns.tolist()
            
            # Calculate feature ranges safely
            for col in X.columns:
                try:
                    min_val = float(X[col].min())
                    max_val = float(X[col].max())
                    if min_val == max_val:
                        min_val -= 1
                        max_val += 1
                    feature_ranges[col] = (min_val, max_val)
                except Exception as e:
                    feature_ranges[col] = (-1, 1)  # fallback range
            
            # Calculate AUC
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

# Add error handling for unpacking
try:
    model, scaler, explainer, feature_names, feature_ranges = train_model()
    if model is None:
        st.error("Model training failed. Please check your data and try again.")
        st.stop()
except ValueError as e:
    st.error(f"Error unpacking model components: {str(e)}")
    st.stop()

# -------------------------------
# User Input for Attribution
# -------------------------------
st.sidebar.header("🔧 Input Features")

# Add reset button
if st.sidebar.button("Reset All Features"):
    for k in st.session_state.keys():
        if k.startswith("slider_"):
            del st.session_state[k]

user_input = {}
for feature in feature_names:
    try:
        min_val, max_val = feature_ranges[feature]
        default = float((min_val + max_val) / 2)
        
        # Ensure values are within reasonable bounds
        min_val = max(-1e6, min_val)
        max_val = min(1e6, max_val)
        default = np.clip(default, min_val, max_val)

        user_input[feature] = st.sidebar.slider(
            feature,
            min_value=float(min_val),
            max_value=float(max_val),
            value=float(default),
            key=f"slider_{feature}",
            step=0.01 * (max_val - min_val)  # Add appropriate step size
        )
    except Exception as e:
        st.sidebar.error(f"Error with feature {feature}: {str(e)}")
        user_input[feature] = 0.0

# Enable realtime updates
st.sidebar.markdown("---")
auto_update = st.sidebar.checkbox("Enable realtime updates", value=True)

input_df = pd.DataFrame([user_input])
input_scaled = pd.DataFrame(scaler.transform(input_df), columns=feature_names)

# -------------------------------
# Prediction
# -------------------------------
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
# SHAP Attribution
# -------------------------------
st.subheader("📊 Feature Attribution (SHAP Values)")
shap_values = explainer.shap_values(input_scaled)

shap_df = pd.DataFrame({
    "Feature": feature_names,
    "SHAP Value": shap_values[0]
}).sort_values("SHAP Value", key=abs, ascending=False)

st.dataframe(shap_df.style.background_gradient(cmap="coolwarm", subset=["SHAP Value"]), use_container_width=True)

# -------------------------------
# Global SHAP Summary Plot
# -------------------------------
st.subheader("🌐 Global Feature Importance (SHAP Summary)")
background_data = explainer.data if hasattr(explainer, "data") else None

if background_data is not None:
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(explainer.shap_values(background_data), background_data, plot_type="bar", show=False)
    st.pyplot(fig)
else:
    st.warning("Global SHAP summary not available.")
