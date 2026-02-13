# 🏠 Gurgaon House Price Prediction using Machine Learning

## 📌 Project Overview
This project focuses on predicting house prices in Gurgaon using Machine Learning techniques. 
It uses various property features such as area, number of rooms, bathrooms, furnishing status, 
and floors to estimate the price of a house.

The main objective is to build an accurate and reliable regression model that can assist buyers, 
sellers, and real estate analysts in making better decisions.

---

## 📊 Dataset Description
The dataset contains approximately 25,000 records of housing properties in Gurgaon created using chatgpt (for practice purpose) with the following features:

- Area (in sq. ft.)
- Number of Rooms
- Number of Bathrooms
- Furnishing Status
- Number of Floors
- Location
- Price (Target Variable)

Data preprocessing includes feature scaling, categorical encoding, and stratified sampling 
based on price categories.

---

## ⚙️ Methodology

### 1. Data Preprocessing
- Removed unnecessary columns
- Created price categories for stratified splitting
- Separated numerical and categorical features
- Standardized numerical attributes

### 2. Feature Engineering
- StandardScaler for numerical features
- OneHotEncoder for categorical features
- Combined using ColumnTransformer

### 3. Model Training
The following regression models were implemented and evaluated:

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor

10-fold cross-validation was used for performance evaluation.

### 4. Pipeline Creation
A complete machine learning pipeline was created using Scikit-learn to automate:

- Data preprocessing
- Feature encoding
- Feature scaling
- Model training

The trained model and pipeline are saved using Joblib for future inference.

---

## 📈 Results and Evaluation
All models were evaluated using Root Mean Squared Error (RMSE).

| Model              | Performance |
|--------------------|-------------|
| Linear Regression  | Best Model  |
| Decision Tree      | Overfitting |
| Random Forest      | Moderate    |

Linear Regression achieved the lowest RMSE and showed the most stable performance.
Therefore, it was selected as the final model for deployment and prediction.

---

## 🚀 How to Run the Project

## 1. Install Dependencies
pip install -r requirements.txt

## How to Run

1. Install requirements:
   pip install -r requirements.txt

2. Train model:
   python gurgoan_final.py

3. Run inference:
   python gurgoan_final.py

## Author
Rajpal B Choudhary
