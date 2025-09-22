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
(Dollar-scale metrics available via Duan smearing if needed)

---

## 🔧 Modeling & Evaluation

**Why `log(price)`?**  
Stabilizes variance and interprets errors as percentage-style deviations.

**Tuning Strategy**  
- 5-fold cross-validation  
- Narrow alpha ranges to avoid convergence to OLS-like fits (minimal shrinkage)

---

Feel free to explore the notebooks, inspect the feature engineering pipeline, or extend the modeling to tree-based methods or SHAP interpretation.
