import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

MODEL_FILE="model.pkl"
PIPELINE_FILE="pipeline.pkl"

def build_pipeline(num_attrbs, cat_attrbs):
    
    num_pipeline=Pipeline([
        ("scaler",StandardScaler())
    ])

    cat_Pipeline=Pipeline([
        ("onehot",OneHotEncoder())
    ])

    full_pipeline=ColumnTransformer([
        ("num", num_pipeline,num_attrbs),
        ("category",cat_Pipeline,cat_attrbs)
    ])

    return full_pipeline

if not os.path.exists(MODEL_FILE):

    # REading CSV file
    data=pd.read_csv("gurgaon_housing_dataset_25k.csv")

    # Spliting Train and Test Set
    split=StratifiedShuffleSplit(n_splits=2, test_size=0.2, random_state=42)
    data["price_cat"]=pd.cut(data["price_inr"], bins=5, labels=[1,2,3,4,5])

    for train_index, test_index in split.split(data, data["price_cat"]):
        training_set=data.iloc[train_index].drop("price_cat",axis=1)
        test_set=data.iloc[test_index].drop("price_cat",axis=1).to_csv("input.csv",index=False)
    
    housing=training_set.copy()

    # Separating Labels and Features

    housing_labels=housing["price_inr"].copy()
    housing_features=housing.drop("price_inr", axis=1)

    # Separating Numerical and Categorical Attributes
    housing_num=housing_features.drop("furnishing",axis=1).columns.tolist()
    housing_cat=["furnishing"]

    # Building pipelines
    mypipeline=build_pipeline(housing_num,housing_cat)
    housing_prepared=mypipeline.fit_transform(housing_features)

    #Fitting the model
    model=LinearRegression()
    model.fit(housing_prepared,housing_labels)

    #Dumping the final model via joblib

    joblib.dump(model,MODEL_FILE)
    joblib.dump(mypipeline,PIPELINE_FILE)
    print("Model is Trained Sucessfully!!")

else:
    # Inference phase
    model=joblib.load(MODEL_FILE)
    mypipeline=joblib.load(PIPELINE_FILE)

    input_data=pd.read_csv("input.csv")
    transformed_input=mypipeline.transform(input_data)
    Predictions=model.predict(transformed_input)
    input_data["price_inr"]=Predictions

    input_data.to_csv("output.csv",index=False)
    print("Inference phase is completed. Results are saved to Output.csv!!")
