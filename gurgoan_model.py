import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit,cross_val_score
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline  

# 1. read the csv

data=pd.read_csv("gurgaon_housing_dataset_25k.csv")

# 2. split the data into training and test sets
split=StratifiedShuffleSplit(n_splits=2, test_size=0.2, random_state=42)
data["price_cat"]=pd.cut(data["price_inr"],bins=5, labels=[1,2,3,4,5])
for train_index, test_index in split.split(data, data["price_cat"]):
    train_set=data.iloc[train_index].drop("price_cat",axis=1)
    test_set=data.iloc[test_index].drop("price_cat",axis=1)

# Working data
housing=train_set.copy()

# 3. Separate labels and features
housing_labels=housing["price_inr"]
housing=housing.drop("price_inr",axis=1)

# 4. Defining numerical and categorical attributes

num_attrbs= housing.drop("furnishing",axis=1).columns.tolist()
cat_attrbs=["furnishing"]

# 5. Defining pipelines 
# For Numerical attrbs

num_pipeline=Pipeline([
    ("scaler",StandardScaler())
])

# For Categorical attrbs

cat_pipeline=Pipeline([
    ("onehot",OneHotEncoder())
])

# Full pipeline
full_pipeline=ColumnTransformer([
    ("num",num_pipeline,num_attrbs),
    ("category",cat_pipeline,cat_attrbs)
])

# 6. Transforming Final pipeline
housing_prepared=full_pipeline.fit_transform(housing)

# 7. Choosing the best Regressor method 

# Linear Regression
lin_reg=LinearRegression()
lin_reg.fit(housing_prepared,housing_labels)
lin_preds=lin_reg.predict(housing_prepared)
lin_rmses=-cross_val_score(lin_reg,
                          housing_prepared,
                          housing_labels,
                          scoring="neg_root_mean_squared_error",
                          cv=10)
print(f" Linear Regression \n{pd.Series(lin_rmses).describe()}")

# Decisionn tree regression
dec_tree=DecisionTreeRegressor(random_state=42)
dec_tree.fit(housing_prepared,housing_labels)
dec_preds=dec_tree.predict(housing_prepared)
dec_rmses=-cross_val_score(dec_tree,
                          housing_prepared,
                          housing_labels,
                          scoring="neg_root_mean_squared_error",
                          cv=10)
print(f" Decision Tree Regressor \n{pd.Series(dec_rmses).describe()}")

# Random Forest regression
Random_forest=RandomForestRegressor(random_state=42)
Random_forest.fit(housing_prepared,housing_labels)
random_forest_preds=Random_forest.predict(housing_prepared)
random_for_rmses=-cross_val_score(Random_forest,
                          housing_prepared,
                          housing_labels,
                          scoring="neg_root_mean_squared_error",
                          cv=10)
print(f" Random Forest Regressor \n{pd.Series(random_for_rmses).describe()}")