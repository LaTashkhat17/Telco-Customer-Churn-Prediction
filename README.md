## Telco Customer Churn Prediction — Complete ML Workflow

# 1.Data Collection

Purpose: Gather data for analysis and modeling

Tool / Source: Kaggle — Telco Customer Churn Dataset

Output: Raw dataset with customer features & target (Churn)

# 2.Exploratory Data Analysis (EDA)

Purpose: Understand data distribution, missing values, relationships

Techniques:

df.info(), df.describe(), df.isnull().sum()

Univariate analysis: histogram, boxplot, countplot

Bivariate analysis: numeric vs target (box/violin), categorical vs target (countplot, barplot)

Correlation heatmap for numeric + encoded categorical features

Output: Insights about feature distributions, outliers, class imbalance

# 3.Feature Engineering

Purpose: Improve predictive power / create business-relevant features

Techniques:

Encode categorical variables (OneHotEncoder, drop_first=True)

Scale numeric features (StandardScaler)

Create new features:

tenure_group (0–12, 12–24, etc.)

avg_monthly_charge = TotalCharges / (tenure + 1)

high_monthly_charge flag

Output: Preprocessed feature matrix X, target y
