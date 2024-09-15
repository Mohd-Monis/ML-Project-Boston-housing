import csv
import numpy as np
from sklearn.model_selection import train_test_split


def get_poly(X):
    m,n = X.shape
    X_ply = np.zeros((m, 2 * n))
    for i in range(m):
        for j in range(n):
            X_ply[i][j] = X[i][j]
            X_ply[i][n + j] = X[i][j]** 2
            
    return X_ply

# from sklearn.linear_model import LinearRegression

# def train_linear_regression(X, Y):
#     model = LinearRegression()
#     model.fit(X, Y)
#     w = model.coef_
#     b = model.intercept_
    
#     return w, b

def train(X, Y, alpha):
    m, n = X.shape
    w = np.zeros((n,))
    b = 0
    prev_cost = compute_cost(X, Y, w, b);
    delta_cost = 10
    while delta_cost > 0.0001:
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
Y_predict_test = []
Y_predict_train = []
Y_train_ = []
Y_test_ = []

with open('train.csv', newline='') as csvfile:
    csv_reader = csv.reader(csvfile)
    l = []

    for row in csv_reader:
        l.append(row)
    
    dataset = np.array(l[1:])
    m,n = dataset.shape
    dataset = np.array([[np.float32(dataset[i][j]) for j in range(n)] for i in range(m)])
    X = dataset[:, 1:n - 1]
    X = get_poly(X)
    Y = dataset[:,n-1]

    w = np.zeros(2*(n-2))
    b = 0
    
    X, means, stds = create_normalization(X)
    mean_model = means;
    stds_model = stds
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=43)
    w, b = train(X_train,Y_train,alpha=0.02);
    Y_test_ = Y_test
    Y_train_ = Y_train
    Y_predict_test = [np.dot(w, X_test[i]) + b for i in range(X_test.shape[0])]
    Y_predict_train = [np.dot(w,X_train[i]) + b for i in range(X_train.shape[0])]
    w_model = w
    b_model = b

X_test_final = []
Y_test_final = []
Y_predict_final = []



with open('test.csv', newline='') as csvfile:
    csv_reader = csv.reader(csvfile)
    l = []
    for row in csv_reader:
        l.append(row)
    dataset = np.array(l[1:])
    m,n = dataset.shape
    dataset = np.array([[np.float32(dataset[i][j]) for j in range(n)] for i in range(m)])
    X_test = dataset[:, 1:n]
    X_test = get_poly(X_test)
    X_test_norm = normalize_testing(X_test, mean_model, stds_model)
    Y_predict_final = [np.dot(w_model, X_test_norm[i]) + b_model for i in range(m)]

ID = []    

with open('submission_example.csv') as csvfile:
    csv_reader = csv.reader(csvfile)
    l = []
    for row in csv_reader:
        l.append(row)
  
    Y = np.array(l[1:])

    ID = np.array([Y[i][0] for i in range(Y.shape[0])])


average_diff = 0;

for i in range(len(Y_predict_test)):
    average_diff += np.abs(Y_predict_test[i] - Y_test_[i]) / np.float32(Y_test_[i])
    
average_diff /= len(Y_predict_test)
print("The accuracy on test data of the model is :", 100 - average_diff*100,"%")

with open('predictions.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['ID','medv'])  
    for i in range(len(Y_predict_final)):
        writer.writerow([ID[i],Y_predict_final[i]])






    

