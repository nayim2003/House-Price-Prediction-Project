# 🏠 House Price Prediction — Ames Housing Dataset

An end-to-end machine learning regression project that predicts residential home sale prices using the Ames Housing Dataset (2,930 records, 82 features).

## 📊 Project Summary

| Metric | Value |
|---|---|
| Best Model | Gradient Boosting Regressor (GridSearchCV-tuned) |
| Test R² | **0.9429** |
| 5-Fold CV R² | 0.9039 ± 0.0139 |
| MAE | $14,563.90 |
| RMSE | $21,453.34 |
| Improvement over baseline | +94.38% R², −76.11% RMSE |

## 🔍 Workflow

1. **Dataset Overview** — 2,930 rows, 82 features (numerical + categorical), target = `SalePrice`
2. **Missing Value Analysis** — verified whether missing values represent absent features (e.g. no pool, no garage) vs. real gaps, then imputed accordingly (`None`/`0` for absent features, neighborhood-median for `Lot Frontage`, mode/median elsewhere)
3. **Exploratory Data Analysis (EDA)** — target distribution, correlation heatmap, categorical breakdowns, neighborhood-level pricing
4. **Outlier Handling** — conservative removal of only 3 records (partial-sale anomalies: `Gr Liv Area > 4000 & SalePrice < $300K`), while retaining legitimate luxury homes
5. **Feature Engineering** — created `Total SF`, `Total Bath`, `House Age`, `Remod Age`, `Is Remodeled`, `Total Porch SF`
6. **Encoding** — ordinal encoding for quality-scale columns, one-hot encoding for nominal categoricals
7. **Feature Selection** — top 30 features by correlation with `SalePrice`
8. **Scaling** — `StandardScaler` for linear models only; tree-based models trained on unscaled data
9. **Model Training** — Linear Regression, Ridge, Lasso, Random Forest, Gradient Boosting, XGBoost
10. **Hyperparameter Tuning** — GridSearchCV on Gradient Boosting (`learning_rate=0.05`, `max_depth=3`, `n_estimators=300`)
11. **Validation** — 5-fold cross-validation + baseline (mean-predictor) comparison
12. **Error Analysis** — error breakdown by price segment (highest error in luxury segment >$400K)

## 🏆 Top Predictive Features

1. Overall Qual (41.1% importance)
2. Total SF (36.1% importance)
3. Total Bath, House Age, Kitchen Qual, Bsmt Qual, Lot Area, ...

## 📁 Repository Structure

```
├── house_price_prediction.ipynb     # Full analysis notebook (EDA → modeling → evaluation)
├── reports/
│   ├── House_Price_Prediction_Report.docx
│   └── House_Price_Prediction_Report.pdf
├── README.md
└── .gitignore
```

## 🛠️ Tech Stack

- Python, Pandas, NumPy
- Matplotlib, Seaborn, Plotly
- Scikit-learn (Linear/Ridge/Lasso, Random Forest, Gradient Boosting, GridSearchCV)
- XGBoost

## 📄 Full Report

See [`reports/House_Price_Prediction_Report.pdf`](reports/House_Price_Prediction_Report.pdf) for the complete write-up with all charts and interpretations.

## 📌 Future Work

- Separate model / additional features for the luxury (>$400K) segment
- Try LightGBM / CatBoost
- Stacking/blending ensembles
- SHAP-based explainability analysis
