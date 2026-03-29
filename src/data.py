from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import pandas as pd

def load_data():

    data = fetch_california_housing()
    df = pd.DataFrame(data.data, columns=feature_names) 
    df["target"] = data.target
    return df

def preprocess_data(df):

    X = df.drop("target", axis=1).values
    y = df["target"].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y

def get_splits(test_size=0.2):
    
    df = load_data()
    X, y = preprocess_data(df)

    return train_test_split(X, y, test_size)