# Telco Customer Churn Prediction — Complete ML Workflow

## 1.Data Collection

  Purpose: Gather data for analysis and modeling
  
  Tool / Source: Kaggle — [Telco Customer Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
  
  Output: Raw dataset with customer features & target (Churn)

## 2.Exploratory Data Analysis (EDA)

  Purpose: Understand data distribution, missing values, relationships
  
  ### Techniques:
  
  `df.info()`, `df.describe()`, `df.isnull().sum()`
  
  Univariate analysis: histogram, boxplot, countplot
  
  Bivariate analysis: numeric vs target (box/violin), categorical vs target (countplot, barplot)
  
  Correlation heatmap for numeric + encoded categorical features
  
  Output: Insights about feature distributions, outliers, class imbalance

## 3.Feature Engineering

  Purpose: Improve predictive power / create business-relevant features
  
  ### Techniques:
  
  Encode categorical variables (OneHotEncoder, drop_first=True)
  
  Scale numeric features (StandardScaler)
  
  ### Create new features:
  
  tenure_group (0–12, 12–24, etc.)
  
  avg_monthly_charge = TotalCharges / (tenure + 1)
  
  high_monthly_charge flag
  
  Output: Preprocessed feature matrix X, target y

## 4.Train-Test Split

  Purpose: Separate data for unbiased evaluation
  
  ### Techniques:
  
  `train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)`
  
  stratify=y → preserves churn proportion
  
  random_state=42 → reproducible results
  
  Output: `X_train, X_test, y_train, y_test`

## 5.Handling Class Imbalance

  Purpose: Ensure model does not ignore churners
  
  ### Techniques:
  
  Cost-sensitive learning → class_weight='balanced' (for LR, RF, DT, SVM)
  
  Optional SMOTE → only if recall < threshold (not needed here)
  
  Output: Balanced model performance, high recall

## 6.Model Training

  ### Models Trained:
  
  Logistic Regression (class_weight='balanced')
  
  Decision Tree
  
  Random Forest (class_weight='balanced')
  
  XGBoost (scale_pos_weight)
  
  SVM (class_weight='balanced')
  
  KNN
  
  ### Evaluation Metrics:
  
  Recall (Churn = 1) → priority
  
  ROC-AUC → class separation
  
  Output: Trained pipelines for all models

## 7.Model Evaluation & Selection

  ### Comparison Table Example:
    
  | Model               | Recall | ROC-AUC |
  |--------------------|--------|---------|
  | Random Forest       | 0.7807 | 0.8452  |
  | Logistic Regression | 0.7834 | 0.8417  |
  | XGBoost             | 0.7513 | 0.8367  |
  | Decision Tree       | 0.8075 | 0.8302  |
  | SVM                 | 0.7807 | 0.8250  |
  | KNN                 | 0.5374 | 0.8026  |

    
  ### Final Best 3 Models:
  
  Random Forest
  
  Logistic Regression
  
  XGBoost
  
  Decision Basis: Recall, ROC-AUC, interpretability, stability

## 8.Feature Importance

  Purpose: Identify key churn drivers
  
  ### Techniques:
  
  Random Forest → feature_importances_
  
  XGBoost → plot_importance(importance_type='gain')
  
  SHAP → per-sample contributions
  
  ### Top Features:
  
  Contract_Month-to-Month, tenure, MonthlyCharges, InternetService_Fiber optic, PaymentMethod_Electronic check


## 9.SHAP Analysis

  Purpose: Explain individual predictions for business stakeholders
  
  ### Techniques:
  ```python
  import shap
  explainer = shap.TreeExplainer(rf_clf)
  shap_values = explainer.shap_values(X_train_processed)
  shap.summary_plot(shap_values[1], X_train_processed, feature_names=all_features)
  ```
  
  Outcome: Visualized per-customer and aggregate contributions to churn probability

## 9.Business Insights

  Short-term contracts → high churn
  
  High monthly charges → higher churn risk
  
  Fiber optic internet → higher churn
  
  Electronic check payments → slightly higher churn
  
  ### Actionable Recommendations:
  
  Target retention campaigns to Month-to-Month fiber customers
  
  Offer discounts or loyalty programs for high-billing churn-prone segments
