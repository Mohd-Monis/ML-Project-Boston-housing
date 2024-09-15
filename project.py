import csv
import numpy as np
from sklearn.linear_model import LinearRegression

def train_linear_regression(X, Y):
    model = LinearRegression()
    model.fit(X, Y)
    w = model.coef_
    b = model.intercept_
    
    return w, b

def train(X, Y, alpha):
    m, n = X.shape
    w = np.zeros((n,))
    b = 0
    prev_cost = compute_cost(X, Y, w, b);
    delta_cost = 10
    while delta_cost > 0.00001:
        delta = np.zeros((n, ),dtype='float32')
        for j in range(n):
            delta_j = 0
            for k in range(m):
                delta_j += (np.dot(w, X[k]) + b - Y[k]) * X[k][j]
            delta_j /= m
            delta[j] = delta_j
        
        delta_b = 0.0
        for k in range(m):
            delta_b += (np.dot(w, X[k]) + b - Y[k])
        delta_b /= m
        w -= alpha * delta
        b -= alpha * delta_b
        curr_cost = compute_cost(X, Y, w, b)
        delta_cost = (prev_cost - curr_cost)/prev_cost
        prev_cost = curr_cost
    
    return w, b

def compute_cost(X, Y, w, b):
    J = 0;
    m, n = X.shape
    for i in range(m):
        J += np.square(np.dot(X[i], w) + b - Y[i])
    J /= m
    return J

def create_normalization(X):
    m, n = X.shape;
    means = np.array([np.mean(X[:, j]) for j in range(n)])
    stds = np.array([np.std(X[:, j]) for j in range(n)])
    for i in range(m):
        X[i] = (X[i] - means) / stds
    return X,means,stds

def normalize_testing(X_test, means, stds):
    m, n = X_test.shape
    for i in range(m):
        X_test[i] = (X_test[i] - means) / stds
    return X_test


w_model = []
b_model = 0
mean_model = []
stds_model = []

# Open the CSV file
Y_predict_train = []
Y_train = []
with open('train.csv', newline='') as csvfile:
    csv_reader = csv.reader(csvfile)
    l = []

    for row in csv_reader:
        l.append(row)
    
    dataset = np.array(l[1:])
    m,n = dataset.shape
    dataset = np.array([[np.float32(dataset[i][j]) for j in range(n)] for i in range(m)])
    X = dataset[:,1:n - 1]
    Y = dataset[:,n-1]
    print(X.shape)
   

    w = np.zeros(n-1)
    b = 0
    
    X, means, stds = create_normalization(X)
    mean_model = means;
    stds_model = stds
    w, b = train_linear_regression(X,Y);
    Y_train = Y
    Y_predict_train = [np.dot(w,X[i]) + b  for i in range(m)]
    w_model = w
    b_model = b

X_test = []
Y_test = []
Y_predict = []



with open('test.csv', newline='') as csvfile:
    csv_reader = csv.reader(csvfile)
    l = []
    for row in csv_reader:
        l.append(row)
    dataset = np.array(l[1:])
    m,n = dataset.shape
    dataset = np.array([[np.float32(dataset[i][j]) for j in range(n)] for i in range(m)])
    X_test = dataset[:, 1:n]
    X_test_norm = normalize_testing(X_test, mean_model, stds_model)
    Y_predict = [np.dot(w_model, X_test_norm[i]) + b_model for i in range(m)]
    
with open('submission_example.csv') as csvfile:
    csv_reader = csv.reader(csvfile)
    l = []
    for row in csv_reader:
        l.append(row)
    print(l[0])
    Y = np.array(l[1:])
    Y_test = np.array([np.float32(Y[i][1]) for i in range(Y.shape[0])])


# print(Y_test)

average_diff = 0;

for i in range(len(Y_predict_train)):
    print(Y_predict_train[i], Y_train[i])
    average_diff += np.abs(Y_predict_train[i] - Y_train[i]) / np.float32(Y_train[i])
    
average_diff /= m
print(average_diff)




    

