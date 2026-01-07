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


