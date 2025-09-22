# 🚗 Used Car Price Modeling (CRISP-DM)

Predicting used-car prices for a mainstream dealership using linear regularization (Ridge, Lasso, Elastic Net) with a log(price) target, documented via the CRISP-DM process.

---

## 📌 Overview

**Business Goal**  
Identify what consumers value in regular used cars (excluding vintage, exotic, and sports models) to support pricing and acquisition decisions.

**Data Task**  
Supervised regression on `log(price)` using structured features such as age, odometer, brand, drivetrain, condition, fuel type, cylinders, body type, and paint bucket.

**Scope**  
Focused on mainstream retail inventory:
- Price range: $3,000–$60,000  
- Vehicle age: ≤15 years

---

## 📊 Data

**Source**  
Kaggle “Used Cars” subset (trimmed for speed and quality).

**Key Columns Used**
- `price`, `year`, `odometer`
- `manufacturer` (one-hot encoded, ~41 categories)
- `condition`, `cylinders` (bucketed), `fuel`, `drive`, `transmission`, `size`, `type`
- `paint_color` (bucketed), `VIN` (used for deduplication)

**Excluded**
- `state`: dropped for simplicity  
- `model`: excluded due to granularity and messiness

**Filters & Cleaning**
- Retain listings priced $3k–$60k and aged ≤15 years
- Drop “finance/installment” posts (not full prices)
- Impute `odometer` by year median (train-only) + missing flag
- Parse `cylinders` → impute by type median → bucket into: ≤4, 5, 6, 8, 10+, baseline = unknown
- Fill categorical NAs/blanks as `"unknown"`
- `paint_color` → collapsed into: neutrals / bright / other (baseline = other)
- One-hot encode remaining categoricals; collapse rare manufacturers to `"other"`

---

## 📈 CRISP-DM Summary

**Business Understanding**  
Focus on price drivers for mainstream retail; exclude placeholders and exotic listings.

**Data Understanding**  
Profiling, missingness analysis, and VIN-based duplicate detection (common in public listings).

**Data Preparation**  
Filtering, imputation, bucketing, and one-hot encoding of categorical variables.

**Modeling**  
Linear regularization methods (Ridge, Lasso, Elastic Net) applied to standardized numeric features and binary dummies. Target variable is `log(price)`.

**Evaluation**  
Metrics computed on log scale:
- MAE  
- RMSE  
- R²  

---

## 🔧 Modeling & Evaluation

**Why `log(price)`?**  
Stabilizes variance and interprets errors as percentage-style deviations.

**Tuning Strategy**  
- 5-fold cross-validation 
- Narrow alpha ranges to avoid convergence to OLS-like fits (minimal shrinkage)

===============================================================================
MODEL COMPARISON
===============================================================================
      Model             Best Alpha  MAE (log)  RMSE (log)  R² (log) Non-zero Features
      Ridge                 1.0000       0.33        0.52      0.65             83/83
      Lasso                 0.0100       0.34        0.53      0.64             62/83
Elastic Net 0.0100 (l1_ratio=0.10)       0.33        0.52      0.65             81/83

---
Best model: **Elastic Net** (α=0.01, l1_ratio=0.10) — ties Ridge on accuracy with mild sparsity.

Top drivers (log-scale coefficients — relative to baselines):

age                -0.516
fuel_gas           -0.212
fuel_other         -0.118
drive_fwd          -0.107
transmission_other +0.091
cyl_8              +0.091
condition_fair     -0.082
drive_unknown      -0.074
cyl_3_4            -0.072
fuel_hybrid        -0.069


---

## 🔍 Findings: What Moves Price

- **Age dominates**  
  Strong negative effect on log-price—newer cars command higher prices.

- **Powertrain matters**  
  8-cylinder configurations carry a premium versus small-cylinder baselines.  
  Gas and some “other” fuel types tend to discount relative to the baseline.

- **Drivetrain & condition**  
  FWD and fair condition are associated with discounts.  
  Transmission effects are present and consistent.

- **Geography**  
  `state` was excluded from the final model for simplicity.  
  Can be reintroduced (or grouped into regions) if cross-validation shows performance lift.

---

## 💼 Business Recommendations

- **Acquisition**  
  Favor younger, lower-mileage vehicles.  
  Selectively acquire trims with favorable powertrain signals (e.g., 8-cylinder premium segments).

- **Pricing**  
  Apply stronger markdowns for fair condition or FWD where discounts are observed.  
  Maintain firmer pricing for configurations with positive effects.

- **Listing Quality**  
  Avoid installment/placeholder prices.  
  Populate missing attributes—missingness correlates with lower realized prices.

- **Inventory Mix**  
  Maintain a balanced brand mix.  
  Emphasize manufacturers historically associated with stronger pricing power.

---

## 🔮 Next Steps

- **Model Families**  
  Try tree-based boosting (e.g., `HistGradientBoostingRegressor`, LightGBM, CatBoost) to capture nonlinearity and interactions.  
  Retain log target and compare dollar-scale MAE/RMSE (with Duan smearing).

- **Feature Set**  
  Reintroduce `state` (one-hot encoded).  
  Test light interactions (e.g., `manufacturer × type`, `state × drive`) via cross-validated A/B experiments.

- **Encodings**  
  For very wide categoricals (e.g., `region`, `model`), use K-fold target encoding with smoothing.

- **Transforms**  
  Explore `log1p(odometer)` and spline transformations for `age` within the ≤15-year scope.

- **Robustness**  
  Validate model performance on a hold-out market (different geography or time window) before deployment.
