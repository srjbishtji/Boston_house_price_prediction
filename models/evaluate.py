import joblib
from sklearn.metrics import mean_squared_error, r2_score

def evaluate_model(X_test, y_test):
    model = joblib.load('models/linear_regression_model.pkl')
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f'Mean Squared Error: {mse}')
    print(f'R-squared: {r2}')

if __name__ == "__main__":
    from train import load_data
    data = load_data('data/housing.csv')
    X_test = data.drop('MEDV', axis=1)
    y_test = data['MEDV']
    
    evaluate_model(X_test, y_test)

