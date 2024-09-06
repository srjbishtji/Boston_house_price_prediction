import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

def test_prediction():
    column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
    df = pd.read_csv('data/housing.csv', header=None, names=column_names, delim_whitespace=True)
    X = df.drop('MEDV', axis=1)
    y = df['MEDV']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Load the model
    model = joblib.load('models/linear_regression_model.pkl')
    
    # Test predictions
    y_pred = model.predict(X_test)
    
    assert len(y_pred) == len(X_test), "Prediction length mismatch!"
    print("Model test passed!")

if __name__ == "__main__":
    test_prediction()

