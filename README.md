# Medical Insurance Charges Prediction

A machine learning project that predicts individual medical insurance charges based on policyholder demographic and lifestyle attributes, with a focus on identifying key cost drivers to support risk-based pricing and underwriting decisions.

## Business Problem

Insurers price health and medical cover based on risk factors tied to the policyholder. Underpricing high-risk policyholders erodes underwriting margins, while overpricing low-risk policyholders hurts competitiveness and retention. Accurately estimating expected claim cost at the point of underwriting is central to sustainable pricing.

This project addresses the following question: given policyholder demographic and lifestyle attributes (age, sex, BMI, number of dependents, smoking status, and region), can we predict expected medical insurance charges and identify which factors drive cost the most?

### Objectives

1. Build a regression model that estimates expected charges from policyholder attributes, to support risk-based pricing decisions.
2. Identify and quantify key cost drivers to inform underwriting guidelines and wellness-program targeting.
3. Visualize cost patterns by segment and region to help pricing and product teams identify high-risk pools and pricing gaps.

### Note on Data

This project uses a public medical insurance dataset (US-based) as a proxy for the type of policyholder data an insurer would hold. The modeling and dashboarding approach demonstrated here applies directly to proprietary policyholder and claims data, adjusted for local regulatory and demographic context.

## Dataset

Source: [Medical Cost Personal Datasets](https://www.kaggle.com/datasets/mirichoi0218/insurance) (Kaggle)

1,338 records with the following features:

| Column | Description |
|---|---|
| age | Age of policyholder |
| sex | Gender of policyholder |
| bmi | Body mass index |
| children | Number of dependents covered |
| smoker | Smoking status (yes/no) |
| region | Residential region |
| charges | Individual medical costs billed by insurance (target variable) |

## Methodology

### Exploratory Data Analysis

- Univariate analysis of all features, including distribution and skew checks
- Bivariate analysis of each feature against charges
- Multivariate analysis of key feature interactions
- Key finding: charges are highly right-skewed, and the interaction between smoking status and BMI is the strongest driver of high-cost cases

### Data Preprocessing

- Categorical encoding (one-hot encoding) for sex, smoker, and region
- Stratified train-test split on smoker status to preserve class representation given the underlying class imbalance
- Log-transformation was evaluated but not applied to features, as age, BMI, and children showed no meaningful skew; applying it did not improve model performance

### Modeling Approach

Several models were built and compared iteratively:

1. Linear Regression (baseline)
2. Linear Regression with engineered interaction terms (BMI x smoker, age x BMI, age x BMI x smoker)
3. Decision Tree Regressor (hyperparameter-tuned via grid search)
4. Random Forest Regressor (hyperparameter-tuned via grid search)

Models were evaluated using RMSE, R-squared, and MAE, with additional diagnostic checks including error breakdown by smoker status and inspection of the largest prediction errors.

### Model Selection

The tuned Random Forest Regressor was selected as the final model based on the following results:

| Model | RMSE | R-squared | MAE |
|---|---|---|---|
| Linear Regression (baseline) | 5,701.34 | 0.7885 | 4,087.84 |
| Linear Regression + BMI x smoker interaction | 4,624.93 | 0.8608 | 2,771.91 |
| Decision Tree (tuned) | 4,694.54 | - | - |
| Random Forest (tuned) | 4,408.70 | 0.8735 | 2,516.07 |

The final model shows no evidence of overfitting, with near-identical train and test R-squared (0.8735 vs 0.8735). Feature importance analysis confirms smoking status as the dominant cost driver (70.4 percent), followed by BMI (17.6 percent) and age (11.1 percent), with region and sex contributing negligibly.

### Key Finding: Unexplained Variance

Across every model tested, a consistent subgroup of older, high-BMI, non-smoking policyholders remained underpredicted. This pattern persisted regardless of model complexity, suggesting the presence of a latent risk factor not captured in this dataset, such as a pre-existing medical condition. In a production setting, this would point to the value of enriching the model with medical history or prior claims data.

## Project Structure

```
insurance_charges_prediction/
|---  README.md
|---  requirements.txt
|--- data/
│   |--- raw/                   Original dataset
│   |--- sim_data/                   simulated dataset
│--- underwriting.ipynb           Exploratory data analysis, Model training, tuning, and evaluation
|---  models/
│   |---  model.pkl              Trained Random Forest pipeline
|--- my_dag.py                   Airflow file for scheduled dataset generation
|--- simulate.py                 simulation for data

```

## Tools and Technologies

- Python (pandas, numpy, scikit-learn, seaborn, matplotlib)
- Power BI for dashboard visualization
- Apache Airflow for pipeline orchestration (see Future Work)

## Dashboard

A Power BI dashboard visualizes charges distribution, cost drivers, and regional patterns, and is updated with model predictions to compare actual versus expected charges by segment.

## Future Work

- Orchestrate the scoring pipeline using Apache Airflow, automating the extraction of new policyholder data, model scoring, and loading of results for dashboard consumption
- Evaluate the model against Microsoft Fabric as the underlying data platform for enterprise-scale deployment
- Incorporate additional risk factors, such as medical history, to address the unexplained variance identified in the high-BMI, non-smoking segment
- Extend the model to a claims-level dataset for a more direct fit with underwriting workflows

## Installation

```
git clone https://github.com/MarwaBrian/insurance_charges_prediction.git
cd insurance-charges-prediction
pip install -r requirements.txt
```

## Usage

1. Run `underwriting.ipynb` to reproduce the exploratory analysis plus model training and evaluation
