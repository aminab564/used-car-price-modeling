Used Car Price Modeling (CRISP-DM)

Predicting used-car prices for a mainstream dealership using linear regularization (Ridge, Lasso, Elastic Net) with a log(price) target, documented through the CRISP-DM process.

Overview

Business goal: Identify what consumers value in a regular used car (not vintage/exotic/sport) and support pricing & acquisition decisions.

Data task: Supervised regression on log(price) using structured features (age, odometer, brand, drivetrain, condition, fuel, cylinders, body type, paint bucket).

Scope: Focused on mainstream retail inventory: $3,000–$60,000 and ≤15 years old.

Data

Source: Kaggle “Used Cars” subset (trimmed for performance and quality).

Columns used: price, year, odometer, manufacturer (one-hot ~41), condition, cylinders (bucketed), fuel, drive, transmission, size, type, paint_color (bucketed), VIN (for dedup checks), region/state (state ultimately not used).

CRISP-DM in brief

Business Understanding → dealership-oriented pricing, exclude installments & exotics.

Data Understanding → profiling, missingness, VIN duplicates (many VINs missing).

Data Preparation → filters, imputations, encodings (below).

Modeling → Ridge, Lasso, Elastic Net (log target).

Evaluation → MAE/RMSE/R² on log scale (dollar metrics available via smearing if needed).

Preparation (key rules)

Filters: keep $3k–$60k; age ≤ 15; drop listings with “finance” in model (installments).

Impute: odometer by year median (train-only) + flag; cylinders parsed → impute by type median → buckets (≤4, 5, 6, 8, 10+, baseline unknown).

Categoricals: fill blanks as "unknown" for size, fuel, condition, title_status, transmission, drive, type, paint_color.

Buckets: paint_color → neutrals | bright | other (baseline other).

One-hot: manufacturer (rare → “other”), state was NOT used in the final model; model dropped.

Modeling

Target: log_price = log(price).

Algorithms: Ridge, Lasso, Elastic Net with standardized numerics and one-hot cats.

Tuning note: Alphas were restricted because wide grids converged to OLS-like fits (minimal shrinkage).
