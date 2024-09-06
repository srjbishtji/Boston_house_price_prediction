import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

def load_data("data/housing.csv"):
    column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
    df = pd.read_csv(file_path, header=None, names=column_names, delim_whitespace=True)
    return df

def train_model(df):
    X = df.drop('MEDV', axis=1)
    y = df['MEDV']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Save the model
    joblib.dump(model, 'models/linear_regression_model.pkl')

    return model, X_test, y_test

if __name__ == "__main__":
    data = load_data('data/housing.csv')
    model, X_test, y_test = train_model(data)
    print("Model trained and saved!")

