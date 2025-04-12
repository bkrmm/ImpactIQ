# ImpactIQ - Uplift Attribution Engine
                                    ~*Measuring True Market Impact: Quantifying the Causal Uplift of Interventions.*
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Problem Statement
Businesses invest heavily in marketing campaigns, website features, and user experience improvements, but struggle to isolate the *true causal impact* of these interventions on conversion rates or other key metrics. Standard correlation-based attribution models often misattribute conversions, leading to inefficient resource allocation. It's challenging to determine if a user converted *because* of a specific touchpoint (true uplift) or if they would have converted anyway. ImpactIQ tackles this by employing causal inference techniques to measure the incremental impact (uplift) of specific treatments (e.g., ad exposure, feature usage) on individual user behavior, specifically focusing on purchase decisions within complex user journeys.

## Solution Approach
ImpactIQ leverages uplift modeling, a branch of causal inference, to estimate the difference in the probability of an outcome (e.g., purchase) when a user is exposed to a treatment versus when they are not.

1.  **Data Preparation:** Utilizes datasets with explicit treatment/control group assignments (like the Criteo Uplift Prediction dataset) or observational data where pseudo-experiments can be constructed. Requires careful preprocessing to handle features, treatment flags (`treatment`), and outcome flags (`visit`, `conversion`).
2.  **Uplift Modeling:** Implements advanced uplift models using libraries like EconML. Techniques may include:
    * **Meta-Learners:** Such as the Two-Model approach (T-Learner - separate models for treatment/control groups), S-Learner (single model with treatment as a feature), or X-Learner (more robust for heterogeneous treatment effects).
    * **Tree-based Methods:** Causal Forests or Uplift Trees that directly optimize for uplift.
    * **Instrumental Variable / Double Machine Learning (DML):** EconML's DML estimators are particularly powerful for observational data, using nuisance models (for outcome and treatment propensity) to isolate the causal effect. XGBoost is often used for these high-capacity nuisance models.
3.  **Feature Engineering:** Creates interaction features and relevant user characteristics to capture potential heterogeneous treatment effects (i.e., how uplift varies across different user segments).
4.  **Evaluation:** Assesses model performance using uplift-specific metrics like the Qini curve, Area Under the Uplift Curve (AUUC), and cumulative uplift charts, rather than traditional classification metrics (Accuracy, AUC).
5.  **Simulation & Deployment:** Provides a Streamlit-based simulator for exploring "what-if" scenarios and demonstrating model predictions. The architecture is designed conceptually for near real-time application using a Kafka stream for data ingestion and a model serving component.

## Technical Details
* **Dataset:** Primarily developed using the [Criteo Uplift Prediction Dataset](https://ailab.criteo.com/criteo-uplift-prediction-dataset/). This dataset contains user features, a binary treatment indicator, and binary outcome indicators (visit and conversion), making it ideal for uplift modeling benchmarking.
* **Modeling Core:** EconML's `CausalForestDML` or other DML estimators (`LinearDML`, `NonParamDML`) are often employed, leveraging `XGBoost` or `LightGBM` as the underlying machine learning models for estimating the conditional outcome E\[Y|X, W] and treatment propensity E\[T|X, W]. This approach helps control for confounding variables (W) when estimating the effect of treatment (T) on outcome (Y) given features (X).
* **Evaluation Metrics:**
    * **Qini Curve / AUUC:** Measures the cumulative uplift gained by targeting users in descending order of predicted uplift score. A higher curve indicates better model performance.
    * **Uplift@k:** The average uplift observed in the top k% of users ranked by predicted uplift.

## Conceptual Architecture
[User Interaction Data Stream (e.g., Website Clicks, Ad Views)] ->[Kafka Topic: Raw Events] ->[Python Processing Service (Consumer)] ->[Feature Engineering & Preprocessing] ->[EconML/XGBoost Uplift Model (Inference)] ->[Output: Uplift Scores per User/Interaction] ->[Streamlit Dashboard / API Endpoint (Results & Simulation)][Database/Data Store (Logging Scores)]* **Ingestion:** Kafka handles high-volume event streams.
* **Processing:** Python service consumes events, performs feature lookups/generation, and applies the trained uplift model.
* **Serving/Visualization:** Model predictions are served via an API or visualized directly in a Streamlit application for analysis and simulation.

## Challenges Encountered
* **Data Scale & Specificity:** Processing the large Criteo dataset efficiently; correctly handling the `treatment` and `conversion`/`visit` flags crucial for uplift modeling.
* **Uplift Modeling Complexity:** Tuning meta-learners or DML models requires careful cross-validation strategies specific to causal inference (e.g., ensuring treatment/control splits are respected). Interpreting uplift metrics (Qini/AUUC) differs significantly from standard classification evaluation.
* **Feature Engineering for Causality:** Identifying features that are pre-treatment confounders versus post-treatment mediators is critical. Ensuring features don't introduce bias and are robust for causal claims.
* **Near Real-Time Simulation:** Building a responsive Streamlit app that simulates the pipeline (data input -> feature processing -> model prediction -> result display) required efficient data handling and model loading within the app's lifecycle. The focus remained on a proof-of-concept simulation rather than a fully operational real-time system.
* **(Bonus) Simulator UI:** Designing an intuitive interface in Streamlit for users to input hypothetical user features/treatments and observe the predicted uplift score.

## Tech Stack
* **Core:** Python 3.8+
* **Causal ML / ML:** `EconML`, `XGBoost`, `scikit-learn`
* **Data Handling:** `pandas`, `numpy`
* **Streaming (Conceptual):** `kafka-python` (or similar)
* **Web UI / Simulation:** `Streamlit`
* **Visualization:** `Matplotlib`, `Seaborn`

## Getting Started
**Prerequisites:**
* Python (>= 3.8)
* Miniconda/Anaconda (Recommended for managing environments)
* Git

**Installation:**
1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/ImpactIQ.git](https://github.com/your-username/ImpactIQ.git) # Replace with your repo URL
    cd ImpactIQ
    ```
2.  **Create and activate a conda environment (Recommended):**
    ```bash
    conda create -n impactiq python=3.9 -y
    conda activate impactiq
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: Ensure `requirements.txt` includes all necessary packages: pandas, numpy, scikit-learn, econml, xgboost, matplotlib, seaborn, streamlit, kafka-python, etc.)*

## Usage
1.  **Prepare Data:** Ensure the Criteo dataset (or your own data formatted similarly) is available and accessible by the scripts (e.g., in a `data/` directory). Update configuration files if necessary to point to the data location.
2.  **Train Model (Example):**
    ```bash
    python src/train_uplift_model.py --config configs/config.yaml
    ```
    *(This is a hypothetical script; adapt based on your project structure)*
3.  **Run the Streamlit Simulator:**
    ```bash
    streamlit run src/app.py
    ```
    Navigate to the URL provided by Streamlit (usually `http://localhost:8501`) in your web browser. Interact with the UI components to simulate different user profiles and treatments and observe the predicted uplift.

## Contributing
Contributions are welcome! Please refer to the `CONTRIBUTING.md` file (if available) for guidelines on how to contribute, report issues, or suggest enhancements.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
* Criteo AI Lab for providing the Uplift Prediction Dataset.
* The developers of EconML, XGBoost, Streamlit, and other open-source libraries used in this project.
