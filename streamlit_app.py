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
import time # Optional: For showing download progress details

st.set_page_config(layout="wide")
st.title("🚀 ImpactIQ - Customer Acquisition Attribution Analysis")
st.markdown("By Bikramjeet Singh Bedi, Made during Synapses'25 Hackathon by IIT Roorkee")

@st.cache_resource(show_spinner="Training attribution model...") # More specific spinner text
def train_model():
    """
    Downloads data, preprocesses, trains XGBoost model, and creates SHAP explainer.
    Returns: tuple (model, scaler, explainer, feature_names, feature_ranges)
             Returns (None, StandardScaler(), None, [], {}) on failure.
    """
    try:
        st.info("Downloading dataset from Criteo (may take a moment)...")
        # URL of the dataset
        url = "http://go.criteo.net/criteo-research-uplift-v2.1.csv.gz"

        # Download with progress bar (more manual for better feedback)
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Ensure we got a valid response
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024 # 1 Kbyte
        start_time = time.time()
        downloaded = 0
        data_bytes = BytesIO()
        progress_bar = st.progress(0)
        status_text = st.empty()

        for data_chunk in response.iter_content(block_size):
            downloaded += len(data_chunk)
            data_bytes.write(data_chunk)
            progress = min(int(100 * downloaded / total_size), 100) if total_size > 0 else 50 # Avoid division by zero, show indeterminate if size unknown
            elapsed_time = time.time() - start_time
            speed = downloaded / elapsed_time / 1024 if elapsed_time > 0 else 0 # KB/s
            status_text.text(f"Downloading... {downloaded/1024**2:.1f}/{total_size/1024**2:.1f} MB ({speed:.1f} KB/s) - {progress}%")
            progress_bar.progress(progress)

        status_text.text("Download complete. Loading data...")
        progress_bar.empty() # Remove progress bar after completion

        data_bytes.seek(0) # Rewind the BytesIO object
        data = pd.read_csv(data_bytes, compression="gzip")
        st.success("✅ Dataset loaded successfully!")

        # --- Preprocessing ---
        st.info("Preprocessing data...")
        # Drop columns (adjust if 'visit' is not present or other columns needed)
        data = data.drop(columns=["visit"], errors="ignore")

        # Handle missing values (consider imputation if dropna removes too much)
        initial_rows = len(data)
        data = data.dropna()
        rows_after_na = len(data)
        st.write(f"Removed {initial_rows - rows_after_na} rows with missing values.")
        if rows_after_na == 0:
            st.error("No data remaining after removing missing values. Please check the dataset.")
            return None, StandardScaler(), None, [], {}

        # Optional: Limit size for speed during development/testing
        # data = data.sample(n=100000, random_state=42) # Smaller sample for faster iteration
        # st.write(f"Using a sample of {len(data)} rows.")

        # Split features and label
        if "conversion" not in data.columns:
            st.error("The required 'conversion' column is missing from the dataset.")
            return None, StandardScaler(), None, [], {}
        y = data["conversion"]
        X = data.drop(columns=["conversion"])

        # Check if features remain
        if X.shape[1] == 0:
             st.error("No feature columns remaining after preprocessing.")
             return None, StandardScaler(), None, [], {}

        # Scale features
        scaler = StandardScaler()
        # Fit scaler only on training data, transform both train and test
        # (Fitting here on all X before split for simplicity, but fit_transform on X_train is best practice)
        X_scaled_all = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

        # Split Data (using scaled data)
        st.info("Splitting data into training and testing sets...")
        # Ensure stratification is possible
        if len(y.unique()) < 2:
             st.warning("Target variable 'conversion' has less than 2 unique values. Stratification may not be effective.")
             stratify_param = None
        else:
             stratify_param = y

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled_all, y, test_size=0.2, stratify=stratify_param, random_state=42
        )
        st.write(f"Training set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")

        # Initialize return values
        model = None
        explainer = None
        feature_names = []
        feature_ranges = {}

        # --- Model Training ---
        st.info("Training XGBoost model...")
        # Calculate scale_pos_weight for imbalanced data
        neg_count = (y_train == 0).sum()
        pos_count = (y_train == 1).sum()

        # Handle potential division by zero if one class is missing in the training set (unlikely with stratify but safe)
        if pos_count > 0 and neg_count > 0:
             calculated_scale_pos_weight = neg_count / pos_count
        else:
             calculated_scale_pos_weight = 1 # Default if one class is missing
        st.write(f"Class distribution in train set: Negatives={neg_count}, Positives={pos_count}")
        st.write(f"Using calculated `scale_pos_weight`: {calculated_scale_pos_weight:.2f}")


        model = xgb.XGBClassifier(
            objective='binary:logistic', # Explicitly set objective
            use_label_encoder=False,     # Good practice for newer XGBoost versions
            eval_metric="logloss",       # Common metric for binary classification
            scale_pos_weight=calculated_scale_pos_weight, # Use calculated weight
            random_state=42              # For reproducibility
        )
        model.fit(X_train, y_train)

        # --- SHAP Explainer ---
        st.info("Creating SHAP explainer...")
        try:
            # Use a sample of the training data for the background dataset
            background_sample_size = min(1000, X_train.shape[0])
            background_data = X_train.sample(background_sample_size, random_state=1)
            explainer = shap.TreeExplainer(model, data=background_data) # Pass background data here
            st.write(f"SHAP explainer created using a background sample of {background_sample_size} instances.")
        except Exception as e:
            st.warning(f"SHAP explainer creation failed: {str(e)}")
            explainer = None

        # --- Feature Info ---
        feature_names = X.columns.tolist() # Use original X columns for names

        # Calculate feature ranges from the original (unscaled) data for sliders
        for col in X.columns:
            try:
                min_val = float(X[col].min())
                max_val = float(X[col].max())
                # Handle case where min and max are the same (e.g., constant feature)
                if min_val == max_val:
                    min_val -= 0.5 # Add some range for the slider
                    max_val += 0.5
                feature_ranges[col] = (min_val, max_val)
            except Exception as e:
                st.warning(f"Could not calculate range for feature '{col}': {e}. Using default (-1, 1).")
                feature_ranges[col] = (-1.0, 1.0)  # fallback range

        # --- Evaluation ---
        st.info("Evaluating model on test set...")
        try:
             y_pred_proba = model.predict_proba(X_test)[:, 1]
             auc = roc_auc_score(y_test, y_pred_proba)
             st.success(f"✅ Model trained. Test AUC = {auc:.4f}")
        except Exception as e:
             st.error(f"Failed to evaluate model on test set: {str(e)}")


        return model, scaler, explainer, feature_names, feature_ranges

    except requests.RequestException as e:
        st.error(f"Fatal Error: Failed to download dataset. Check URL or network connection. Details: {str(e)}")
        return None, StandardScaler(), None, [], {}
    except pd.errors.EmptyDataError:
        st.error("Fatal Error: Downloaded file is empty or not a valid CSV/Gzip.")
        return None, StandardScaler(), None, [], {}
    except Exception as e:
        st.error(f"Fatal Error during data processing or model training: {str(e)}")
        import traceback
        st.error(traceback.format_exc()) # Print full traceback for debugging
        return None, StandardScaler(), None, [], {}

# --- Main App Logic ---

# Train model (or load from cache)
model, scaler, explainer, feature_names, feature_ranges = train_model()

# Stop execution if model training failed
if model is None or scaler is None or not feature_names:
    st.error("Model training failed or returned invalid components. Cannot proceed.")
    st.stop() # Stop the script execution

# -------------------------------
# User Input for Attribution
# -------------------------------
st.sidebar.header("🔧 Input Features")

# Add reset button
if st.sidebar.button("Reset All Features to Defaults"):
    # Need to know defaults or recalculate them
    # Simple approach: just clear state, sliders will revert to initial value on next run
    for k in st.session_state.keys():
        if k.startswith("slider_"): # Target only slider keys
            del st.session_state[k]
    st.rerun() # Force rerun to apply default values immediately

user_input = {}
for feature in feature_names:
    if feature in feature_ranges:
        try:
            min_val, max_val = feature_ranges[feature]
            # Calculate a reasonable default (e.g., median or mean of original data)
            # Using midpoint of range as a fallback default if original data isn't easily accessible here
            default_val = float((min_val + max_val) / 2)

            # Define slider step (adjust logic as needed)
            range_diff = max_val - min_val
            step = 0.01 if range_diff < 1 else 0.1 # Example: smaller step for small ranges
            if range_diff > 100: step = 1.0 # Larger step for large ranges
            if range_diff == 0: step = 0.1 # Handle zero range case

            # Use unique key for each slider
            slider_key = f"slider_{feature}"

            user_input[feature] = st.sidebar.slider(
                label=feature,
                min_value=float(min_val),
                max_value=float(max_val),
                value=float(default_val), # Default value for the slider
                step=float(step) if step > 0 else None, # Ensure step is positive or None
                key=slider_key, # Assign unique key
                help=f"Range: ({min_val:.2f} to {max_val:.2f})" # Add tooltip
            )
        except Exception as e:
            st.sidebar.error(f"Error creating slider for feature '{feature}': {str(e)}")
            user_input[feature] = 0.0 # Fallback input value
    else:
         st.sidebar.warning(f"Range not found for feature '{feature}'. Using default input 0.0.")
         user_input[feature] = 0.0

# Removed the unused "Enable realtime updates" checkbox
# st.sidebar.markdown("---")
# auto_update = st.sidebar.checkbox("Enable realtime updates", value=True)

# Create DataFrame from user input
input_df = pd.DataFrame([user_input])

# Scale the user input using the *fitted* scaler
try:
    input_scaled = pd.DataFrame(scaler.transform(input_df), columns=feature_names)
except Exception as e:
    st.error(f"Error scaling user input: {str(e)}")
    st.stop()

# -------------------------------
# Prediction
# -------------------------------
st.subheader("🎯 Predicted Conversion Probability")
try:
    pred_prob = model.predict_proba(input_scaled)[0, 1]
    # Display prediction using a metric card
    col1, col2 = st.columns([1, 3]) # Adjust column widths if needed
    with col1:
        st.metric(
            label="Conversion Probability",
            value=f"{pred_prob:.4f}",
            # Delta is optional, could show change from previous run or baseline
            # delta=f"{pred_prob - 0.0025:.4f}", # Keeping your original delta logic
            # delta_color="inverse"
        )
    with col2:
        # Add a simple progress bar/indicator based on probability
        st.progress(pred_prob)
        st.caption("Likelihood of conversion based on current inputs.")

except Exception as e:
    st.error(f"Error during prediction: {str(e)}")
    pred_prob = None # Set to None if prediction fails

# -------------------------------
# SHAP Attribution (only if prediction succeeded and explainer exists)
# -------------------------------
if pred_prob is not None and explainer is not None:
    st.subheader("📊 Feature Attribution (Local SHAP Values)")
    st.markdown("How much each feature value contributed to *this specific prediction* (pushing it higher or lower).")
    try:
        # Calculate SHAP values for the single input instance
        # Ensure input_scaled has the correct format (e.g., DataFrame)
        shap_values_instance = explainer.shap_values(input_scaled)

        # For binary classification with TreeExplainer, shap_values returns a list [shap_values_class_0, shap_values_class_1]
        # Or sometimes just the shap_values for the positive class. Check the structure.
        # Usually, we are interested in the explanation for the positive class (class 1).
        if isinstance(shap_values_instance, list):
             shap_values_positive_class = shap_values_instance[1][0] # Explain class 1, first instance
        else:
             shap_values_positive_class = shap_values_instance[0] # Assume it directly returned class 1 shap values for first instance

        # Create DataFrame for display
        shap_df = pd.DataFrame({
            "Feature": feature_names,
            "SHAP Value": shap_values_positive_class
        })

        # Sort by absolute SHAP value to see strongest contributors first
        shap_df["Abs SHAP Value"] = shap_df["SHAP Value"].abs()
        shap_df = shap_df.sort_values("Abs SHAP Value", ascending=False).drop(columns=["Abs SHAP Value"])

        # Display sorted SHAP values with a background gradient
        st.dataframe(
             shap_df.style.format({"SHAP Value": "{:.4f}"}) # Format numbers
                        .background_gradient(cmap="coolwarm", subset=["SHAP Value"]),
             use_container_width=True
        )

        # Optional: Add a SHAP force plot for the single prediction
        st.markdown("---")
        st.markdown("**Force Plot (Local Explanation)**")
        st.caption("Visualizes the push and pull of features on the prediction.")
        # shap.force_plot(explainer.expected_value[1], shap_values_positive_class, input_scaled.iloc[0], matplotlib=False) # Needs adaptation for Streamlit display
        # Workaround for displaying force plot in Streamlit
        force_plot = shap.force_plot(explainer.expected_value[1], shap_values_positive_class, input_scaled.iloc[0], show=False)
        st.components.v1.html(force_plot.html(), height=100) # Use st.components.v1.html


    except Exception as e:
        st.error(f"Error calculating or displaying local SHAP values: {str(e)}")
        import traceback
        st.error(traceback.format_exc()) # Print full traceback


    # -------------------------------
    # Global SHAP Summary Plot (only if explainer and its background data exist)
    # -------------------------------
    st.subheader("🌐 Global Feature Importance (Overall Model Behavior)")
    st.markdown("Which features are most important for the model's predictions *on average* across many data points.")
    # Ensure the explainer has the background data it was created with
    if hasattr(explainer, 'data') and explainer.data is not None:
        try:
            st.info(f"Generating SHAP summary plot based on {len(explainer.data)} background samples...")
            fig, ax = plt.subplots() # Create matplotlib figure explicitly

            # Generate SHAP values for the background data
            shap_values_background = explainer.shap_values(explainer.data)

            # Determine which SHAP values to plot (for class 1)
            if isinstance(shap_values_background, list):
                shap_values_plot = shap_values_background[1] # Class 1
            else:
                shap_values_plot = shap_values_background # Assume only class 1 returned

            shap.summary_plot(
                 shap_values_plot,
                 explainer.data, # Pass the background data used by explainer
                 plot_type="bar",  # Bar plot shows mean absolute SHAP value
                 feature_names=feature_names, # Ensure feature names are passed
                 show=False # Prevent matplotlib from showing the plot directly
            )
            st.pyplot(fig) # Display the plot in Streamlit
            plt.close(fig) # Close the plot to free memory

            # Add dot summary plot for more detail
            st.markdown("---")
            st.markdown("**Detailed SHAP Summary Plot (Dot Plot)**")
            st.caption("Shows distribution of SHAP values for each feature.")
            fig2, ax2 = plt.subplots()
            shap.summary_plot(
                 shap_values_plot,
                 explainer.data,
                 feature_names=feature_names,
                 show=False
            )
            st.pyplot(fig2)
            plt.close(fig2)

        except Exception as e:
            st.error(f"Error generating global SHAP summary plot: {str(e)}")
            import traceback
            st.error(traceback.format_exc()) # Print full traceback

    else:
        st.warning("Could not generate global SHAP summary plot: Background data not found in explainer.")

elif explainer is None:
     st.warning("SHAP Explainer was not created successfully. Attribution plots are unavailable.")
else:
     st.warning("Prediction failed. SHAP attribution cannot be calculated.")

st.markdown("---")
st.info("App execution complete.")