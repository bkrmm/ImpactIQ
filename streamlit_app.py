# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import requests
from io import BytesIO
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

st.set_page_config(layout="wide")
st.title("ImpactIQ - Customer Acquisition Attribution Analysis")
st.markdown("By Bikramjeet Singh Bedi, In Synapses'25 Hackathon by IIT Roorkee")

@st.cache_resource
def train_model():
    try:
        # First try to load from local file in the data directory
        data_path = Path(__file__).parent / "data" / "criteo-uplift-v2.1.csv"
        local_gz_path = Path(__file__).parent / "criteo-uplift-v2.1.csv.gz"
        
        if data_path.exists():
            st.info("Loading dataset from local file...")
            with st.spinner('Loading dataset...'):
                data = pd.read_csv(data_path)
                st.success("✅ Dataset loaded successfully from local file!")
        elif local_gz_path.exists():
            st.info("Loading compressed dataset from local file...")
            with st.spinner('Loading compressed dataset...'):
                data = pd.read_csv(local_gz_path, compression="gzip")
                st.success("✅ Compressed dataset loaded successfully from local file!")
        else:
            # Fallback to downloading a smaller sample of the dataset from URL
            st.info("Local dataset not found. Downloading a smaller sample of the dataset from Criteo...")
            # URL of the dataset
            url = "http://go.criteo.net/criteo-research-uplift-v2.1.csv.gz"
            
            # Download with progress bar
            with st.spinner('Downloading dataset sample...'):
                try:
                    # Set up headers to download only a portion of the file
                    # This will download approximately 30MB instead of 300MB
                    headers = {'Range': 'bytes=0-30000000'}
                    response = requests.get(url, stream=True, timeout=30, headers=headers)
                    response.raise_for_status()  # Ensure we got a valid response
                    
                    # Load the data directly from the response content
                    # Read only the first chunk of data to create a smaller sample
                    data = pd.read_csv(BytesIO(response.content), compression="gzip")
                    
                    # Take a random sample to ensure we have a representative dataset
                    # This further reduces the size while maintaining data distribution
                    sample_size = min(100000, len(data))  # Limit to 100k rows max
                    data = data.sample(n=sample_size, random_state=42)
                    
                    st.success(f"✅ Dataset sample downloaded successfully! Sample size: {len(data)} rows")
                    
                    # Save for future use
                    try:
                        if not local_gz_path.parent.exists():
                            local_gz_path.parent.mkdir(parents=True, exist_ok=True)
                        # Save the sampled data instead of the full file
                        data.to_csv(local_gz_path, compression="gzip", index=False)
                        st.success("✅ Dataset sample saved locally for future use!")
                    except Exception as save_error:
                        st.warning(f"Could not save dataset locally: {str(save_error)}")
                except Exception as download_error:
                    st.error(f"Failed to download dataset: {str(download_error)}")
                    # Try to use a small sample dataset as fallback
                    st.warning("Using a small synthetic dataset as fallback...")
                    # Create a small synthetic dataset
                    data = pd.DataFrame({
                        'f0': np.random.normal(0, 1, 1000),
                        'f1': np.random.normal(0, 1, 1000),
                        'f2': np.random.normal(0, 1, 1000),
                        'treatment': np.random.randint(0, 2, 1000),
                        'conversion': np.random.randint(0, 2, 1000),
                    })

        # Drop columns
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
            
            # Create explainer safely with proper background data
            try:
                # Ensure we have enough background data for SHAP
                background_sample_size = min(5000, len(X_train))
                background_data = X_train.sample(background_sample_size, random_state=1)
                
                # Create explainer with explicit background data
                explainer = shap.TreeExplainer(model, data=background_data)
                
                # Store background data for global summary plot
                explainer.background_data = background_data
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

# Use the background data we explicitly stored in the explainer
if explainer is not None and hasattr(explainer, "background_data") and explainer.background_data is not None:
    try:
        # Create a new figure for the summary plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Calculate SHAP values for background data
        background_shap_values = explainer.shap_values(explainer.background_data)
        
        # Generate the summary plot
        shap.summary_plot(
            background_shap_values, 
            explainer.background_data, 
            plot_type="bar", 
            max_display=20,  # Limit to top 20 features for clarity
            show=False
        )
        
        # Display the plot
        st.pyplot(fig)
    except Exception as e:
        st.warning(f"Error generating SHAP summary plot: {str(e)}")
        # Fallback to simpler visualization if the summary plot fails
        try:
            # Create feature importance from model directly as fallback
            importance_fig, importance_ax = plt.subplots(figsize=(10, 6))
            xgb.plot_importance(model, ax=importance_ax, max_num_features=20)
            importance_ax.set_title("Feature Importance (XGBoost)")
            st.pyplot(importance_fig)
        except:
            st.error("Could not generate feature importance visualization.")
else:
    st.warning("Global SHAP summary not available. SHAP explainer was not properly initialized.")
    # Fallback to model's feature importance
    if model is not None:
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            xgb.plot_importance(model, ax=ax, max_num_features=20)
            ax.set_title("Feature Importance (XGBoost)")
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Could not generate feature importance: {str(e)}")
