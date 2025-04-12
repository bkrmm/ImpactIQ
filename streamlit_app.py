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
import time # Added for demonstrating download progress

# --- Page Configuration ---
st.set_page_config(layout="wide")
st.title("ImpactIQ - Customer Acquisition Attribution Analysis")
st.markdown("By Bikramjeet Singh Bedi, In Synapses'25 Hackathon by IIT Roorkee")

# --- Cache Initialization ---
if 'model_cache' not in st.session_state:
    st.session_state.model_cache = None

# --- Model Training and Data Processing ---
def train_model():
    """
    Downloads data, preprocesses it, trains an XGBoost model,
    and creates a SHAP explainer.
    Returns:
        tuple: (model, scaler, explainer, feature_names, feature_ranges, test_auc)
               Returns (None, None, None, [], {}, None) on failure.
    """
    # Check if model is already in session state
    if st.session_state.model_cache is not None:
        return st.session_state.model_cache

    try:
        st.info("Attempting to download dataset from Criteo...")
        # URL of the dataset
        url = "http://go.criteo.net/criteo-research-uplift-v2.1.csv.gz"

        # Download with progress (simulated here for clarity, actual download time varies)
        progress_bar = st.progress(0)
        status_text = st.empty()
        start_time = time.time()

        response = requests.get(url, stream=True)
        response.raise_for_status()  # Ensure we got a valid response (2xx status code)

        total_size = int(response.headers.get('content-length', 0))
        bytes_downloaded = 0
        chunk_size = 8192
        data_buffer = BytesIO()

        for chunk in response.iter_content(chunk_size=chunk_size):
            bytes_downloaded += len(chunk)
            data_buffer.write(chunk)
            if total_size > 0:
                progress = min(bytes_downloaded / total_size, 1.0)
                progress_bar.progress(progress)
            elapsed_time = time.time() - start_time
            speed = (bytes_downloaded / 1024 / 1024) / elapsed_time if elapsed_time > 0 else 0
            status_text.text(f"Downloading... {bytes_downloaded / 1024 / 1024:.2f} MB downloaded at {speed:.2f} MB/s")

        progress_bar.progress(1.0)
        status_text.text("Download complete. Loading data...")
        st.success("✅ Dataset downloaded successfully!")

        # Load the data directly from the response content in memory
        data_buffer.seek(0) # Rewind the buffer to the beginning
        data = pd.read_csv(data_buffer, compression="gzip")
        status_text.text(f"Dataset loaded. Shape: {data.shape}")

        # --- Data Preprocessing ---
        # Use a smaller sample for faster demo/deployment
        # Note: Sampling *before* dropna might be less efficient if many NaNs exist.
        # Consider data.dropna().sample(...) if memory allows loading full data first.
        data = data.sample(n=100000, random_state=42)
        st.write(f"Sampled data shape: {data.shape}")

        # Drop unnecessary columns (if they exist)
        data = data.drop(columns=["visit"], errors="ignore")

        # Handle missing values - Simple approach: drop rows with any NaN
        initial_rows = len(data)
        data = data.dropna()
        rows_after_na = len(data)
        st.write(f"Shape after dropping NaNs: {data.shape}. ({initial_rows - rows_after_na} rows removed)")

        if data.empty:
            st.error("No data remaining after removing missing values. Cannot train model.")
            return None, None, None, [], {}, None

        # Define features (X) and target (y)
        # Assuming 'conversion' is the target variable
        target_column = "conversion"
        if target_column not in data.columns:
            st.error(f"Target column '{target_column}' not found in the dataset.")
            # Attempt to find a potential target or list columns
            st.write("Available columns:", data.columns.tolist())
            return None, None, None, [], {}, None

        y = data[target_column]
        X = data.drop(columns=[target_column])
        feature_names = X.columns.tolist() # Store original feature names

        # Calculate feature ranges based on the *original* (unscaled) data for sliders
        feature_ranges = {}
        for col in feature_names:
            try:
                min_val = float(X[col].min())
                max_val = float(X[col].max())
                # Handle cases where min == max (e.g., constant column in sample)
                if min_val == max_val:
                    min_val -= 0.5
                    max_val += 0.5
                feature_ranges[col] = (min_val, max_val)
            except Exception as e:
                st.warning(f"Could not calculate range for feature '{col}': {e}. Using default range (-1, 1).")
                feature_ranges[col] = (-1.0, 1.0) # Fallback range

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        # --- Feature Scaling ---
        # IMPORTANT: Fit the scaler ONLY on the training data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test) # Use the SAME scaler fitted on train data

        # Convert scaled arrays back to DataFrames with original column names
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_names, index=X_train.index)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_names, index=X_test.index)

        # --- Model Training ---
        status_text.text("Training XGBoost model...")
        model = xgb.XGBClassifier(
            objective='binary:logistic', # Explicitly set objective for clarity
            use_label_encoder=False,     # Recommended for newer XGBoost versions
            eval_metric="logloss",       # Metric for evaluation during training (if early stopping used)
            scale_pos_weight=np.sum(y_train == 0) / np.sum(y_train == 1), # Adjust for class imbalance
            random_state=42
        )
        model.fit(X_train_scaled, y_train)
        status_text.text("Model training complete.")

        # --- Model Evaluation (on test set) ---
        test_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        test_auc = roc_auc_score(y_test, test_pred_proba)
        st.success(f"✅ Model trained successfully. Test AUC = {test_auc:.4f}")

        # --- SHAP Explainer ---
        status_text.text("Creating SHAP explainer...")
        # Use a subset of the training data as background for TreeExplainer for performance
        # Adjust sample size as needed based on performance/memory
        background_data_sample = shap.sample(X_train_scaled, min(100, len(X_train_scaled)), random_state=1)
        explainer = shap.TreeExplainer(model, data=background_data_sample)
        status_text.text("SHAP explainer created.")

        # Clear status text
        status_text.empty()
        progress_bar.empty()

        # Store results in session state before returning
        st.session_state.model_cache = (model, scaler, explainer, feature_names, feature_ranges, test_auc)
        return st.session_state.model_cache

    except requests.RequestException as e:
        st.error(f"Failed to download dataset: {str(e)}")
        return None, None, None, [], {}, None
    except pd.errors.EmptyDataError:
         st.error("Downloaded file is empty or corrupted.")
         return None, None, None, [], {}, None
    except Exception as e:
        st.error(f"An error occurred during model training or data processing: {str(e)}")
        import traceback
        st.error(f"Traceback: {traceback.format_exc()}")
        return None, None, None, [], {}, None

# --- Load Model and Related Objects ---
model, scaler, explainer, feature_names, feature_ranges, test_auc = train_model()

# Stop execution if model training failed
if model is None or scaler is None or explainer is None:
    st.error("Model components could not be loaded. Cannot proceed.")
    st.stop()

# --- Sidebar: User Input Features ---
st.sidebar.header("🔧 Input Features for Prediction")

# Reset button for sliders
if st.sidebar.button("Reset All Features to Defaults"):
    # Iterate through keys in session state that we added for sliders
    for key in list(st.session_state.keys()):
        if key.startswith("slider_"):
            del st.session_state[key]
    st.experimental_rerun() # Rerun to apply the reset

# Realtime update control
auto_update = st.sidebar.checkbox("Enable Realtime Updates", value=True)

user_input = {}
for feature in feature_names:
    if feature in feature_ranges:
        min_val, max_val = feature_ranges[feature]
        # Calculate a reasonable default (e.g., median or mean if available, otherwise midpoint)
        # Using midpoint as a simple default
        default_val = float((min_val + max_val) / 2)

        # Ensure values are valid for the slider
        min_val_f = float(min_val)
        max_val_f = float(max_val)
        default_val_f = np.clip(float(default_val), min_val_f, max_val_f)

        # Calculate a sensible step (e.g., 1/100th of the range)
        step = max(abs(max_val_f - min_val_f) / 100, 0.01) # Avoid step=0

        try:
            user_input[feature] = st.sidebar.slider(
                label=feature,
                min_value=min_val_f,
                max_value=max_val_f,
                value=default_val_f,
                step=step,
                key=f"slider_{feature}" # Unique key for session state
            )
        except Exception as e:
            st.sidebar.error(f"Error creating slider for {feature}: {e}")
            user_input[feature] = default_val_f # Use default if slider fails
    else:
        st.sidebar.warning(f"Range not found for feature '{feature}'. Using default input 0.")
        user_input[feature] = 0.0 # Default value if range calculation failed

# Button to trigger analysis if auto_update is off
run_analysis_button = False
if not auto_update:
    if st.sidebar.button("Run Analysis"):
        run_analysis_button = True

# --- Main Panel: Prediction and Attribution ---

# Only run prediction and SHAP if auto-updating or button is pressed
if auto_update or run_analysis_button:
    if not user_input:
        st.warning("Please configure input features in the sidebar.")
    else:
        # Create DataFrame from user input
        input_df = pd.DataFrame([user_input])

        # Ensure column order matches the training data
        input_df = input_df[feature_names]

        # Scale the user input using the *same* scaler fitted on training data
        try:
            input_scaled = scaler.transform(input_df)
            # Convert back to DataFrame for SHAP clarity (optional but good practice)
            input_scaled_df = pd.DataFrame(input_scaled, columns=feature_names)

            # --- Prediction ---
            pred_prob = float(model.predict_proba(input_scaled_df)[0, 1])  # Convert to Python float

            st.subheader("🎯 Predicted Conversion Probability")
            st.metric(
                label="Probability of Conversion",
                value=f"{pred_prob:.4f}"
            )
            st.progress(pred_prob)  # Now using Python float

            # --- Local SHAP Attribution ---
            st.subheader(f"📊 Feature Contribution for this Prediction (SHAP Values)")
            try:
                # Calculate SHAP values for the specific input instance
                shap_values_instance = explainer.shap_values(input_scaled_df)

                # For binary classification, shap_values usually returns a list [shap_values_class_0, shap_values_class_1]
                # or just shap_values_class_1 depending on the model/explainer version.
                # We are interested in the explanation for the positive class (conversion = 1).
                # Check the structure of shap_values_instance
                if isinstance(shap_values_instance, list) and len(shap_values_instance) == 2:
                    shap_values_to_display = shap_values_instance[1][0] # SHAP values for class 1, first instance
                else:
                     # Assuming it directly returned values for the positive class
                    shap_values_to_display = shap_values_instance[0] # SHAP values for the first instance


                # Create a DataFrame for display
                shap_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Input Value': input_df.iloc[0].values, # Show the original unscaled input value
                    'SHAP Value': shap_values_to_display
                })

                # Sort by absolute SHAP value to see the most impactful features
                shap_df['abs_SHAP'] = shap_df['SHAP Value'].abs()
                shap_df = shap_df.sort_values(by='abs_SHAP', ascending=False).drop(columns=['abs_SHAP'])

                # Display the DataFrame with conditional formatting
                st.dataframe(shap_df.style.format({'Input Value': '{:.4f}', 'SHAP Value': '{:+.4f}'})
                             .background_gradient(cmap='coolwarm', subset=['SHAP Value']),
                             use_container_width=True)

                # --- SHAP Force Plot (Local Explanation) ---
                st.write("Force plot showing feature impacts on this specific prediction:")
                # Need to make sure plot generation doesn't block or cause issues
                # shap.force_plot requires JS, use st.components.v1 for robust rendering
                force_plot = shap.force_plot(explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value,
                                             shap_values_to_display, # Use the correct shap values array
                                             input_scaled_df.iloc[0], # Use the scaled features for the plot labels
                                             matplotlib=False) # Generate HTML/JS plot

                # Correctly render the SHAP plot HTML
                shap_html = f"<head>{shap.getjs()}</head><body>{force_plot.html()}</body>"
                st.components.v1.html(shap_html, height=100, scrolling=True) # Adjust height as needed


            except Exception as e:
                st.error(f"Could not calculate or display local SHAP values: {e}")
                import traceback
                st.error(f"Traceback: {traceback.format_exc()}")


        except Exception as e:
            st.error(f"An error occurred during prediction or scaling: {e}")
            import traceback
            st.error(f"Traceback: {traceback.format_exc()")
else:
    if not auto_update:
         st.info("Realtime updates are disabled. Configure features and click 'Run Analysis' in the sidebar.")


# --- Global SHAP Summary Plot (shows overall feature importance) ---
# This part depends only on the trained model/explainer, not the specific user input,
# so it can be shown regardless of the 'Run Analysis' button state.
st.subheader("🌐 Global Feature Importance (Mean Absolute SHAP Values)")
try:
    # Use the background data sample stored in the explainer for the summary plot
    background_data = explainer.data

    if background_data is not None and not background_data.empty:
        st.write("This plot shows the average impact of each feature across many predictions.")
        # Calculate SHAP values for the background data
        # Ensure background_data is a DataFrame if needed by shap.summary_plot
        if not isinstance(background_data, pd.DataFrame):
             background_data_df = pd.DataFrame(background_data, columns=feature_names)
        else:
             background_data_df = background_data

        summary_shap_values = explainer.shap_values(background_data_df)

        # Handle potential list output for binary classification
        if isinstance(summary_shap_values, list) and len(summary_shap_values) == 2:
            summary_shap_values_for_plot = summary_shap_values[1] # Class 1
        else:
             summary_shap_values_for_plot = summary_shap_values


        # Create and display the summary plot
        fig, ax = plt.subplots()
        shap.summary_plot(summary_shap_values_for_plot, background_data_df, plot_type="bar", show=False)
        st.pyplot(fig)
        plt.close(fig) # Close the figure to free memory
    else:
        st.warning("Background data for SHAP summary plot is not available.")

except Exception as e:
    st.error(f"Could not generate global SHAP summary plot: {e}")
    import traceback
    st.error(f"Traceback: {traceback.format_exc()}")

st.markdown("---")
st.markdown(f"Model Test AUC: `{test_auc:.4f}` (Area Under the ROC Curve)")
st.caption("Higher AUC indicates better model performance at distinguishing between classes.")
