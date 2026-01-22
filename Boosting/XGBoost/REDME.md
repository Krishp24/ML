# XGBoost Machine Learning Projects

This folder contains two comprehensive machine learning projects demonstrating XGBoost implementations for both classification and regression tasks.

## Projects Overview

### 1. Holiday Package Prediction (Classification)
**File:** `XGBoostClassification.ipynb`

A classification model to predict whether customers will purchase a Wellness Tourism Package based on their characteristics and interaction history.

**Business Context:**
- "Trips & Travel.Com" wants to optimize marketing expenditure for a new Wellness Tourism Package
- Historical data shows only 18% of customers purchased packages with random marketing approach
- Goal: Use available customer data to make targeted marketing more efficient

**Dataset:**
- Source: [Kaggle - Holiday Package Purchase Prediction](https://www.kaggle.com/datasets/susant4learning/holiday-package-purchase-prediction)
- Size: 4,888 rows × 20 columns
- Target Variable: `ProdTaken` (Binary: 0/1)

**Features Include:**
- Demographics: Age, Gender, MaritalStatus, Occupation, Designation
- Engagement: TypeofContact, NumberOfFollowups, DurationOfPitch, PitchSatisfactionScore
- Travel History: NumberOfTrips, Passport, PreferredPropertyStar
- Product Info: ProductPitched, CityTier
- Financial: MonthlyIncome, OwnCar

**Data Preprocessing:**
- Missing value imputation (median for continuous, mode for categorical)
- Category consolidation (e.g., "Fe Male" → "Female", "Single" → "Unmarried")
- Feature engineering: Created `TotalVisiting` from person and children visiting counts
- One-Hot Encoding for categorical variables
- Standard Scaling for numerical features

**Models Compared:**
| Model | Test Accuracy | Test F1 | Test ROC-AUC |
|-------|---------------|---------|--------------|
| Logistic Regression | 0.8364 | 0.8087 | 0.6307 |
| Decision Tree | 0.9172 | 0.9161 | 0.8554 |
| Random Forest | 0.9254 | 0.9194 | 0.8188 |
| Gradient Boost | 0.8589 | 0.8398 | 0.6824 |
| AdaBoost | 0.8354 | 0.7987 | 0.6083 |
| **XGBoost** | **0.9427** | **0.9399** | **0.8673** |

**Best Model (After Hyperparameter Tuning):**
- XGBoost with parameters: `subsample=0.8, n_estimators=500, max_depth=10, learning_rate=0.1, colsample_bytree=0.8`
- Test Accuracy: **95.50%**
- Test ROC-AUC: **0.8927**

---

### 2. Used Car Price Prediction (Regression)
**File:** `XGBoostRegression.ipynb`

A regression model to predict selling prices of used cars based on vehicle specifications and seller information.

**Business Context:**
- Dataset from CardDekho.com (Indian used car marketplace)
- Goal: Provide price suggestions to sellers based on market conditions
- Help buyers understand fair market value

**Dataset:**
- Source: Web scraping from CardDekho website
- Size: 15,411 rows × 13 columns
- Target Variable: `selling_price` (Continuous)

**Features Include:**
- Vehicle Info: model, vehicle_age, km_driven, fuel_type, transmission_type
- Specifications: mileage, engine, max_power, seats
- Seller Info: seller_type

**Data Preprocessing:**
- Removed unnecessary columns (car_name, brand)
- Label Encoding for model names (120+ unique models)
- One-Hot Encoding for seller_type, fuel_type, transmission_type
- Standard Scaling for numerical features

**Models Compared:**
| Model | Test R² Score | Test RMSE | Test MAE |
|-------|---------------|-----------|----------|
| Linear Regression | 0.6645 | 502,544 | 279,619 |
| Lasso | 0.6645 | 502,543 | 279,615 |
| Ridge | 0.6645 | 502,534 | 279,557 |
| K-Neighbors | 0.9150 | 253,024 | 112,526 |
| Decision Tree | 0.8842 | 295,310 | 123,506 |
| **Random Forest** | **0.9269** | **234,526** | **102,606** |
| AdaBoost | 0.5905 | 555,228 | 444,058 |
| Gradient Boosting | 0.9130 | 255,923 | 126,453 |
| XGBoost | 0.9191 | 246,854 | 99,049 |

**Best Model (After Hyperparameter Tuning):**
- Random Forest with parameters: `n_estimators=200, min_samples_split=2, max_features=7, max_depth=None`
- Test R² Score: **0.9339**
- Test RMSE: **223,033**

---

## Key Techniques Demonstrated

### Data Preprocessing
- Handling missing values (median/mode imputation)
- Categorical encoding (One-Hot, Label Encoding)
- Feature scaling (StandardScaler)
- Feature engineering (combining related features)

### Model Training
- Train-test split (80-20)
- Column Transformer for preprocessing pipelines
- Multiple algorithm comparison

### Hyperparameter Tuning
- RandomizedSearchCV with cross-validation
- Grid search over key parameters:
  - `n_estimators`, `max_depth`, `learning_rate`
  - `subsample`, `colsample_bytree`
  - `min_samples_split`, `min_samples_leaf`

### Evaluation Metrics
- **Classification:** Accuracy, Precision, Recall, F1-Score, ROC-AUC
- **Regression:** R² Score, RMSE, MAE

---

## Requirements

```python
pandas
numpy
matplotlib
scikit-learn
xgboost
```

## Usage

1. Ensure all dependencies are installed
2. Update file paths in the notebooks to match your local setup
3. Run notebooks sequentially (cells are numbered)

## Key Findings

### Classification Project
- XGBoost outperformed all other models with 95.5% accuracy
- Feature importance suggests `ProductPitched`, `Passport`, and `Designation` are key predictors
- Hyperparameter tuning improved ROC-AUC from 0.8673 to 0.8927

### Regression Project
- Random Forest achieved the best test performance (R² = 0.9339)
- XGBoost showed strong performance but slight overfitting compared to Random Forest
- Key price predictors: `max_power`, `engine`, `vehicle_age`, `km_driven`

## Author Notes

These notebooks demonstrate end-to-end machine learning workflows including:
- Exploratory data analysis
- Data cleaning and preprocessing
- Model selection and comparison
- Hyperparameter optimization
- Model evaluation and interpretation