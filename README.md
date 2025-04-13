# ImpactIQ - Customer Acquisition Attribution Analysis

A Streamlit application for analyzing customer acquisition attribution using XGBoost and SHAP values.

## Overview

This application analyzes the Criteo Uplift dataset to predict customer conversion probabilities and explain feature attributions using SHAP (SHapley Additive exPlanations) values.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Features

- Interactive feature input through sliders
- Real-time conversion probability prediction
- Feature attribution analysis with SHAP values
- Global feature importance visualization

## Deployment Instructions

### Local Development

1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the app: `streamlit run streamlit_app.py`

### Deploying to Streamlit Cloud

1. Push this repository to GitHub
2. Log in to [Streamlit Cloud](https://streamlit.io/cloud)
3. Create a new app and connect it to your GitHub repository
4. Set the main file path to `streamlit_app.py`
5. Deploy!

#### Important Notes for Deployment

- The app will first try to load the dataset from the local `data` directory
- If not found, it will try to load from a local compressed file
- As a last resort, it will download the dataset from Criteo's URL
- If all data loading methods fail, a small synthetic dataset will be used as a fallback

## Data

This app uses the Criteo Uplift dataset. The dataset will be automatically downloaded if not present locally.

## Requirements

See `requirements.txt` for a list of dependencies.

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
Got it! Below is the content rewritten in a Notion-friendly **Markdown format**, using clean formatting, indentation, and spacing that aligns with how Notion renders markdown (headings, bold, inline math, code blocks, and bullet points). It's ready to copy and paste into a Notion page or markdown file.

---

**ImpactIQ** is an uplift modeling engine designed to quantify the *incremental* impact of a treatment (e.g. marketing campaign, promotion) by comparing outcomes between a **treatment group** and a **control group**.

Unlike traditional machine learning models that predict likelihood of an outcome, **uplift models estimate the causal effect** of an intervention on an individual or segment level.

---

## Uplift Tree-based Algorithms

In uplift tree-based models, **each decision tree** uses both:

- `treatment_column`: indicates whether an observation received the treatment
- `response_column`: indicates whether an observation responded (e.g. converted, clicked)

This allows a **single tree** to model both treatment and control groups **jointly**, instead of building separate trees per group.

The algorithm searches for splits that **maximize the divergence** in outcomes between treatment and control subgroups, rather than optimizing metrics like Gini impurity or squared error.

---

## Uplift Metrics

The **splitting criteria** in uplift trees rely on a configurable `uplift_metric`, which measures how different the treatment and control distributions are after a potential split.

### Supported Metrics

---

#### **1. Kullback-Leibler Divergence** (`uplift_metric = "KL"`)

An asymmetric divergence measure that tends toward infinity when probabilities include zeros.

```math
KL(P, Q) = \sum_{i=0}^{N} p_i \log \frac{p_i}{q_i}
```

- Used in information theory
- Sensitive to zero probabilities

---

#### **2. Squared Euclidean Distance** (`uplift_metric = "euclidean"`)

A symmetric and stable distance metric that does not diverge to infinity.

```math
E(P, Q) = \sum_{i=0}^{N} (p_i - q_i)^2
```

- Numerically stable
- Common in clustering and distance-based models

---

#### **3. Chi-Squared Divergence** (`uplift_metric = "chi_squared"`)

An asymmetric measure normalized by the control group distribution.

```math
\chi^2(P, Q) = \sum_{i=0}^{N} \frac{(p_i - q_i)^2}{q_i}
```

- Useful for evaluating statistical independence
- May also diverge when control group has zeros

---

**Where:**

- `P`: treatment group distribution  
- `Q`: control group distribution

Each split's value in a node is computed as:

```math
\text{split\_score} = \text{metric}(P, Q) + \text{metric}(1 - P, 1 - Q)
```

The final **split gain** is normalized using:

- Gini impurity for `euclidean` or `chi_squared`
- Entropy for `KL`

---

## Uplift Prediction: Tree Leaves

Each **leaf** in an uplift tree contains two conditional probabilities:

- `TP_l`: treatment prediction (likelihood of response if treated)
- `CP_l`: control prediction (likelihood of response if not treated)

Computed as:

```math
TP_l = \frac{TY1_l + 1}{T_l + 2}
```

```math
CP_l = \frac{CY1_l + 1}{C_l + 2}
```

**Where:**

- `l`: current leaf
- `T_l`: number of treatment group samples in leaf  
- `C_l`: number of control group samples in leaf  
- `TY1_l`: treatment group responses (treatment = 1, response = 1)  
- `CY1_l`: control group responses (treatment = 0, response = 1)

The **uplift score** for a leaf is:

```math
uplift\_score_l = TP_l - CP_l
```

A **positive uplift score** indicates the treatment likely caused a higher response rate.  
A **negative score** means the control group had a higher response rate, implying the treatment may have had a negative impact.

---

## Final Prediction

As in standard decision forest models, the final prediction is the **average uplift score** across all trees in the ensemble.

---

## Output Columns (on Prediction)

When calling the `predict()` method on test data, the output includes:

- **`uplift_predict`**: uplift score (i.e. difference between predicted probabilities)
- **`p_y1_with_treatment`**: probability of response if treated
- **`p_y1_without_treatment`**: probability of response if not treated

![training_progress](https://github.com/user-attachments/assets/71d33c17-beb7-4c29-b7cf-9ccc50589187)
![image (1)](https://github.com/user-attachments/assets/7f3bd719-8d33-4675-8ecb-8c3affa33622)


```math
uplift\_predict = p\_y1\_with\_treatment - p\_y1\_without\_treatment
