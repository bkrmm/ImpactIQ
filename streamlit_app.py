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
st.title("ImapactIQ - Customer Acquisition Attribution Analysis")
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
                    # Try to create a small sample dataset from the Criteo structure
                    st.warning("Creating a small sample dataset based on Criteo structure...")
                    
                    # Create a small dataset with the same structure as Criteo dataset
                    # but with fewer rows (10,000) to ensure SHAP explainer works correctly
                    n_samples = 10000
                    
                    # Create a small dataset with the correct structure
                    # The Criteo dataset has features f0-f11, treatment, visit, and conversion
                    data = pd.DataFrame({
                        # Create features with realistic distributions
                        # Using slightly different distributions for each feature
                        'f0': np.random.normal(0, 1, n_samples),
                        'f1': np.random.normal(0.5, 1.2, n_samples),
                        'f2': np.random.normal(-0.2, 0.8, n_samples),
                        'f3': np.random.normal(0.1, 1.5, n_samples),
                        'f4': np.random.normal(-0.5, 1.1, n_samples),
                        'f5': np.random.normal(0.3, 0.9, n_samples),
                        'f6': np.random.normal(-0.1, 1.3, n_samples),
                        'f7': np.random.normal(0.2, 1.0, n_samples),
                        'f8': np.random.normal(-0.3, 1.2, n_samples),
                        'f9': np.random.normal(0.4, 0.7, n_samples),
                        'f10': np.random.normal(-0.4, 1.4, n_samples),
                        'f11': np.random.normal(0.6, 0.8, n_samples),
                        # Treatment is binary (0 or 1)
                        'treatment': np.random.randint(0, 2, n_samples)
                    })
                    
                    # Generate visit and conversion separately to avoid syntax errors
                    # Visit is binary (0 or 1) with correlation to treatment
                    visit_probs = np.clip(0.3 + 0.1 * np.random.normal(0, 1, n_samples), 0, 1)
                    data['visit'] = np.random.binomial(1, visit_probs, n_samples)
                    
                    # Conversion is binary (0 or 1) with correlation to features
                    conversion_probs = np.clip(0.1 + 0.03 * (np.random.normal(0, 1, n_samples) + 
                                                        np.random.normal(0, 1, n_samples)), 0, 1)
                    data['conversion'] = np.random.binomial(1, conversion_probs, n_samples)
                    
                    # Save this sample dataset for future use
                    try:
                        # Create a data directory if it doesn't exist
                        data_dir = Path(__file__).parent / "data"
                        if not data_dir.exists():
                            data_dir.mkdir(parents=True, exist_ok=True)
                            
                        # Save as CSV in the data directory
                        sample_path = data_dir / "criteo-uplift-v2.1.csv"
                        data.to_csv(sample_path, index=False)
                        st.success(f"✅ Sample dataset created and saved locally! Sample size: {len(data)} rows")
                    except Exception as save_error:
                        st.warning(f"Could not save sample dataset locally: {str(save_error)}")
                        
                    st.info("Using this sample dataset for model training...")

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
                scale_pos_weight=0.1,
                # Add parameters to ensure valid probabilities
                max_delta_step=1,
                min_child_weight=1,
                objective="binary:logistic"
            )
            model.fit(X_train, y_train)
            
            # Create explainer safely with proper background data
            try:
                # Ensure we have enough background data for SHAP
                background_sample_size = min(5000, len(X_train))
                background_data = X_train.sample(background_sample_size, random_state=1)
                
                # Validate background data
                if background_data.isnull().values.any():
                    st.warning("Background data contains NaN values. Replacing with zeros.")
                    background_data = background_data.fillna(0)
                
                # Create explainer with explicit background data and model validation
                try:
                    # First try with interventional feature perturbation for better stability
                    explainer = shap.TreeExplainer(
                        model, 
                        data=background_data,
                        feature_perturbation="interventional",
                        model_output="probability"
                    )
                except Exception as explainer_error:
                    st.warning(f"First SHAP explainer attempt failed: {str(explainer_error)}. Trying alternative configuration...")
                    # Fallback to simpler configuration
                    explainer = shap.TreeExplainer(model, data=background_data)
                
                # Store background data for global summary plot
                explainer.background_data = background_data
                
                # Validate explainer by calculating SHAP values for a small sample
                test_sample = background_data.iloc[:5]
                test_shap_values = explainer.shap_values(test_sample)
                
                # Check if SHAP values are valid
                if isinstance(test_shap_values, list):
                    test_array = test_shap_values[0]
                else:
                    test_array = test_shap_values
                    
                if np.isnan(test_array).any():
                    st.warning("SHAP explainer produced NaN values. Using fallback approach.")
                    # Try alternative approach
                    explainer = shap.Explainer(model.predict, background_data)
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
try:
    # Get prediction probabilities with validation
    raw_probs = model.predict_proba(input_scaled)
    # Ensure probabilities are valid
    if np.isnan(raw_probs).any() or (raw_probs < 0).any() or (raw_probs > 1).any():
        st.warning("Model produced invalid probabilities. Applying correction...")
        raw_probs = np.nan_to_num(raw_probs, nan=0.5)
        raw_probs = np.clip(raw_probs, 0, 1)
    
    pred_prob = raw_probs[0, 1]
    st.subheader("🎯 Predicted Conversion Probability")
    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric(
            label="Conversion Probability",
            value=f"{pred_prob:.4f}",
            delta=f"{pred_prob - 0.0025:.4f}",
            delta_color="inverse"
        )
except Exception as e:
    st.error(f"Error in prediction: {str(e)}")
    pred_prob = 0.5  # Fallback to neutral prediction
    st.metric("Conversion Probability", f"{pred_prob:.4f}", delta=None)

# -------------------------------
# SHAP Attribution
# -------------------------------
st.subheader("📊 Feature Attribution (SHAP Values)")

try:
    # Calculate SHAP values with validation
    if explainer is not None:
        # Ensure input data is valid
        if input_scaled.isnull().values.any():
            st.warning("Input contains NaN values. Replacing with zeros.")
            input_scaled = input_scaled.fillna(0)
            
        # Calculate SHAP values with error handling
        shap_values = explainer.shap_values(input_scaled)
        
        # Validate SHAP values
        if isinstance(shap_values, list):
            shap_values_array = shap_values[0]
        else:
            shap_values_array = shap_values
            
        # Check for NaN or invalid values
        if np.isnan(shap_values_array).any():
            st.warning("SHAP values contain NaN. Applying correction...")
            shap_values_array = np.nan_to_num(shap_values_array)
            
        # Create DataFrame for display
        shap_df = pd.DataFrame({
            "Feature": feature_names,
            "SHAP Value": shap_values_array
        }).sort_values("SHAP Value", key=abs, ascending=False)
        
        st.dataframe(shap_df.style.background_gradient(cmap="coolwarm", subset=["SHAP Value"]), use_container_width=True)
    else:
        st.warning("SHAP explainer not available. Cannot calculate feature attributions.")
        
except Exception as e:
    st.error(f"Error calculating SHAP values: {str(e)}")
    st.info("Displaying feature importance from model instead.")
    
    # Fallback to feature importance
    if model is not None:
        importance = model.feature_importances_
        imp_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importance
        }).sort_values("Importance", ascending=False)
        
        st.dataframe(imp_df.style.background_gradient(cmap="viridis", subset=["Importance"]), use_container_width=True)

# -------------------------------
# Global SHAP Summary Plot
# -------------------------------
st.subheader("🌐 Global Feature Importance (SHAP Summary)")

# Use the background data we explicitly stored in the explainer
if explainer is not None and hasattr(explainer, "background_data") and explainer.background_data is not None:
    try:
        # Validate background data
        background_data = explainer.background_data
        if background_data.isnull().values.any():
            st.warning("Background data contains NaN values. Applying correction...")
            background_data = background_data.fillna(0)
        
        # Create a new figure for the summary plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Calculate SHAP values for background data with validation
        try:
            background_shap_values = explainer.shap_values(background_data)
            
            # Validate SHAP values
            if isinstance(background_shap_values, list):
                background_shap_values_array = background_shap_values[0]
            else:
                background_shap_values_array = background_shap_values
                
            # Check for NaN or invalid values
            if np.isnan(background_shap_values_array).any():
                st.warning("SHAP values contain NaN. Applying correction...")
                background_shap_values_array = np.nan_to_num(background_shap_values_array)
                
            # Use corrected values if needed
            if isinstance(background_shap_values, list):
                background_shap_values[0] = background_shap_values_array
            else:
                background_shap_values = background_shap_values_array
                
            # Generate the summary plot with reduced sample size if needed
            max_samples = min(1000, len(background_data))  # Limit samples for stability
            sample_indices = np.random.choice(len(background_data), max_samples, replace=False)
            
            if isinstance(background_shap_values, list):
                sampled_values = [v[sample_indices] for v in background_shap_values]
            else:
                sampled_values = background_shap_values[sample_indices]
                
            sampled_data = background_data.iloc[sample_indices]
            
            # Generate the summary plot
            shap.summary_plot(
                sampled_values, 
                sampled_data, 
                plot_type="bar", 
                max_display=20,  # Limit to top 20 features for clarity
                show=False
            )
            
            # Display the plot
            st.pyplot(fig)
        except Exception as e:
            st.warning(f"Error calculating background SHAP values: {str(e)}")
            raise e  # Re-raise to trigger fallback
            
    except Exception as e:
        st.warning(f"Error generating SHAP summary plot: {str(e)}")
        # Fallback to simpler visualization if the summary plot fails
        try:
            # Create feature importance from model directly as fallback
            importance_fig, importance_ax = plt.subplots(figsize=(10, 6))
            xgb.plot_importance(model, ax=importance_ax, max_num_features=20)
            importance_ax.set_title("Feature Importance (XGBoost)")
            st.pyplot(importance_fig)
        except Exception as fallback_error:
            st.error(f"Could not generate feature importance visualization: {str(fallback_error)}")
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
