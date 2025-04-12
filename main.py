# main.py
import logging
import os
import pandas as pd
import xgboost as xgb
import shap
from fastapi import FastAPI, HTTPException
import numpy as np
from pydantic import BaseModel
from typing import Dict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ======================
# Configuration (Edit these values as needed)
# ======================
DATA_PATH = r"B:\Updated-VenV\hackathon-synapses2025\criteo-uplift-v2.1.csv"
MODEL_PARAMS = {
    "objective": "binary:logistic",
    "tree_method": "hist",
    "n_estimators": 1000,  # Increased for early stopping
    "max_depth": 4,
    "learning_rate": 0.05,  # Reduced for better convergence
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 1,
    "scale_pos_weight": 1,
    "enable_categorical": False,
    "eval_metric": ["auc", "logloss"]
}
TRAINING_SAMPLE_SIZE = 10000000
SHAP_BACKGROUND_SIZE = 100000
# ======================

class AttributionRequest(BaseModel):
    features: Dict[str, float]

def load_data():
    """Load and preprocess data"""
    logger.info(f"Loading data from {DATA_PATH}")
    try:
        data = pd.read_csv(DATA_PATH)
        sample = data.sample(n=TRAINING_SAMPLE_SIZE, random_state=42)
        y = sample["conversion"]
        X = sample.drop(["conversion", "visit"], axis=1)
        
        # Calculate class weight
        neg_pos_ratio = len(y[y==0]) / len(y[y==1])
        MODEL_PARAMS["scale_pos_weight"] = neg_pos_ratio
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale the features
        scaler = StandardScaler()
        X_train = pd.DataFrame(
            scaler.fit_transform(X_train),
            columns=X_train.columns
        )
        X_test = pd.DataFrame(
            scaler.transform(X_test),
            columns=X_test.columns
        )
        
        return X_train, X_test, y_train, y_test, scaler
    except Exception as e:
        logger.error(f"Data loading failed: {str(e)}")
        raise

def initialize_model(X_train, y_train, X_test, y_test):
    """Train and evaluate XGBoost model with progress tracking"""
    logger.info("Initializing model")
    try:
        # Initialize model without duplicate parameters
        model = xgb.XGBClassifier(**MODEL_PARAMS)
        
        # Train model with evaluation sets
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            verbose=True
        )
        
        # Get evaluation results
        results = model.evals_result()
        
        # Plot training progress
        plt.figure(figsize=(12, 6))
        
        # AUC plot
        plt.subplot(1, 2, 1)
        plt.plot(results['validation_0']['auc'], label='Train')
        plt.plot(results['validation_1']['auc'], label='Test')
        plt.title('AUC Progress')
        plt.ylabel('AUC Score')
        plt.xlabel('Iteration')
        plt.legend()
        
        # Log Loss plot
        plt.subplot(1, 2, 2)
        plt.plot(results['validation_0']['logloss'], label='Train')
        plt.plot(results['validation_1']['logloss'], label='Test')
        plt.title('Log Loss Progress')
        plt.ylabel('Log Loss')
        plt.xlabel('Iteration')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('training_progress.png')
        
        # Final evaluation
        y_pred = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_pred)
        logger.info(f"Final Model AUC-ROC on test set: {auc:.4f}")
        
        return model
    except Exception as e:
        logger.error(f"Model initialization failed: {str(e)}")
        raise

def initialize_shap(model, X):
    """Create SHAP explainer with visualization"""
    logger.info("Initializing SHAP explainer")
    try:
        background = X.sample(n=SHAP_BACKGROUND_SIZE, random_state=42)
        explainer = shap.TreeExplainer(
            model,
            data=background,
            feature_perturbation="interventional"
        )
        
        # Generate and save SHAP summary plot
        shap_values = explainer.shap_values(X.sample(10000))
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X.sample(10000), plot_type="bar")
        plt.savefig('shap_summary.png')
        logger.info("Saved SHAP summary plot to shap_summary.png")
        
        return explainer
    except Exception as e:
        logger.error(f"SHAP initialization failed: {str(e)}")
        raise

# Initialize application components
try:
    # Load data and initialize model
    X_train, X_test, y_train, y_test, scaler = load_data()
    model = initialize_model(X_train, y_train, X_test, y_test)
    explainer = initialize_shap(model, X_train)
    
    # Store scaler for preprocessing new data
    app = FastAPI(title="Conversion Attribution Service")
    app.state.scaler = scaler
    logger.info("Service initialization complete")

except Exception as e:
    logger.critical(f"Failed to initialize service: {str(e)}")
    raise

@app.post("/attribution", response_model=Dict[str, float])
async def get_attribution(request: AttributionRequest):
    """Calculate feature attributions for conversion prediction"""
    try:
        # Convert request to DataFrame and scale features
        input_features = pd.DataFrame([request.features])
        input_features = pd.DataFrame(
            app.state.scaler.transform(input_features),
            columns=input_features.columns
        )
        
        # Validate features
        missing = set(input_features.columns) - set(model.feature_names_in_)
        if missing:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid features: {', '.join(missing)}"
            )

        # Calculate SHAP values
        shap_values = explainer.shap_values(input_features)
        
        return {
            feature: float(value)
            for feature, value in zip(input_features.columns, shap_values[0])
        }
        
    except Exception as e:
        logger.error(f"Attribution failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)