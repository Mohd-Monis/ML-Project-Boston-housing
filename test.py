import csv
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score

def train_ridge_regression(X, Y, alpha):
    model = Ridge(alpha=alpha)
    model.fit(X, Y)
    w = model.coef_
    b = model.intercept_
    
    return w, b, model

def compute_cost(X, Y, w, b):
    J = 0
    m, n = X.shape
    for i in range(m):
        J += np.square(np.dot(X[i], w) + b - Y[i])
    J /= m
    return J

def create_normalization(X):
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X, scaler

def normalize_testing(X_test, scaler):
    X_test = scaler.transform(X_test)
    return X_test

# Reading and training with the CSV data
w_model = []
b_model = 0
scaler_model = None

# Open the CSV file
Y_predict_train = []
Y_train = []
with open('train.csv', newline='') as csvfile:
    csv_reader = csv.reader(csvfile)
    l = []

    for row in csv_reader:
        l.append(row)
    
    dataset = np.array(l[1:])
    m, n = dataset.shape
    dataset = np.array([[np.float32(dataset[i][j]) for j in range(n)] for i in range(m)])
    X = dataset[:, 1:n - 1]
    Y = dataset[:, n-1]

    # Feature Scaling and Normalization
    X, scaler = create_normalization(X)
    scaler_model = scaler

    # Adding Polynomial Features
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X)

    # Ridge Regression with Cross-Validation to choose best alpha
    param_grid = {'alpha': [0.001, 0.01, 0.1, 1, 10, 100]}
    grid = GridSearchCV(Ridge(), param_grid, scoring='neg_mean_squared_error', cv=5)
    grid.fit(X_poly, Y)
    
    best_alpha = grid.best_params_['alpha']
    w_model, b_model, model = train_ridge_regression(X_poly, Y, best_alpha)

    Y_train = Y
    Y_predict_train = [np.dot(w_model, X_poly[i]) + b_model for i in range(m)]

# Testing Phase
X_test = []
Y_test = []
Y_predict = []

with open('test.csv', newline='') as csvfile:
    csv_reader = csv.reader(csvfile)
    l = []
    for row in csv_reader:
        l.append(row)
    dataset = np.array(l[1:])
    m, n = dataset.shape
    dataset = np.array([[np.float32(dataset[i][j]) for j in range(n)] for i in range(m)])
    X_test = dataset[:, 1:n]

    # Normalize test data using training scaler
    X_test_norm = normalize_testing(X_test, scaler_model)
    
    # Add polynomial features to test set
    X_test_poly = poly.transform(X_test_norm)
    
    Y_predict = [np.dot(w_model, X_test_poly[i]) + b_model for i in range(m)]

# Compare Predictions with True Values
with open('submission_example.csv', newline='') as csvfile:
    csv_reader = csv.reader(csvfile)
    l = []
    for row in csv_reader:
        l.append(row)
    print(l[0])
    Y = np.array(l[1:])
    Y_test = np.array([np.float32(Y[i][1]) for i in range(Y.shape[0])])

# Model Evaluation Metrics
mse_train = mean_squared_error(Y_train, Y_predict_train)
r2_train = r2_score(Y_train, Y_predict_train)

mse_test = mean_squared_error(Y_test, Y_predict)
r2_test = r2_score(Y_test, Y_predict)

print(f"Train MSE: {mse_train}, Train R2: {r2_train}")
print(f"Test MSE: {mse_test}, Test R2: {r2_test}")

# Output the average difference for inspection (optional)
average_diff = 0
for i in range(len(Y_predict_train)):
    average_diff += np.abs(Y_predict_train[i] - Y_train[i]) / np.float32(Y_train[i])
average_diff /= m
print("Average percentage difference on train set:", average_diff)
