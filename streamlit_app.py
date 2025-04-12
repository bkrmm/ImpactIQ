import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

st.set_page_config(layout="wide")
st.title("ImapactIQ - Customer Acquisition Attribution Analysis")
st.markdown("By Bikramjeet Singh Bedi, In Synapses'25 Hackathon by IIT Roorkee")

@st.cache_resource
def train_model():
    st.info("Loading & training model on sample...")
    data = pd.read_csv(r"-------", compression="gzip")
    data = data.drop(columns=["visit"], errors="ignore")
    data = data.dropna()
    y = data["conversion"]
    X = data.drop(columns=["conversion"])

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=42)

    class_weights = {0: 1, 1: 0.1}  
    model = xgb.XGBClassifier(
        use_label_encoder=False, 
        eval_metric="logloss",
        scale_pos_weight=0.1
    )
    model.fit(X_train, y_train)

    explainer = shap.TreeExplainer(model, data=X_train.sample(1000, random_state=1))

    feature_ranges = {}
    for col in X.columns:
        min_val = float(X[col].min())
        max_val = float(X[col].max())

        if min_val == max_val:
            min_val -= 1
            max_val += 1
        feature_ranges[col] = (min_val, max_val)
        
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    st.success(f"✅ Model trained. Test AUC = {auc:.4f}")

    return model, scaler, explainer, X.columns.tolist(), feature_ranges

model, scaler, explainer, feature_names, feature_ranges = train_model()

st.sidebar.header("🔧 Input Features")

if st.sidebar.button("Reset All Features"):
    for k in st.session_state.keys():
        if k.startswith("slider_"):
            del st.session_state[k]

user_input = {}
for feature in feature_names:
    try:
        min_val, max_val = feature_ranges[feature]
        default = float((min_val + max_val) / 2)
        
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
        st.sidebar.error(f"Error with feature {feature}: {str(e)}")
        user_input[feature] = 0.0

st.sidebar.markdown("---")
auto_update = st.sidebar.checkbox("Enable realtime updates", value=True)

input_df = pd.DataFrame([user_input])
input_scaled = pd.DataFrame(scaler.transform(input_df), columns=feature_names)

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

st.subheader("📊 Feature Attribution (SHAP Values)")
shap_values = explainer.shap_values(input_scaled)

shap_df = pd.DataFrame({
    "Feature": feature_names,
    "SHAP Value": shap_values[0]
}).sort_values("SHAP Value", key=abs, ascending=False)

st.dataframe(shap_df.style.background_gradient(cmap="coolwarm", subset=["SHAP Value"]), use_container_width=True)

st.subheader("🌐 Global Feature Importance (SHAP Summary)")
background_data = explainer.data if hasattr(explainer, "data") else None

if background_data is not None:
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(explainer.shap_values(background_data), background_data, plot_type="bar", show=False)
    st.pyplot(fig)
else:
    st.warning("Global SHAP summary not available.")
