import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

def load_data(file_path='data/housing.csv'):
    column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
    # Use raw string r'\s+' to avoid the invalid escape sequence warning
    df = pd.read_csv(file_path, header=None, names=column_names, sep=r'\s+')
    return df

def train_model(df):
    X = df.drop('MEDV', axis=1)
    y = df['MEDV']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Save the model to the models directory
    joblib.dump(model, 'models/linear_regression_model.pkl')

    return model, X_test, y_test

if __name__ == "__main__":
    # Use a relative path instead of an absolute path
    file_path = 'data/housing.csv'
    data = load_data(file_path)
    model, X_test, y_test = train_model(data)
    print("Model trained and saved!")
