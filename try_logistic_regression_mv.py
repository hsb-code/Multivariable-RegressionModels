import numpy as np
import matplotlib.pyplot as plt
import copy
import math

# dataset

x_train = np.array([[0.5, 1.5], [1, 1], [1.5, 0.5],
                   [3, 0.5], [2, 2], [1, 2.5]])
print(x_train)
y_train = np.array([0,  0, 0, 1, 1, 1])

# plot dataset

plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train, cmap=plt.cm.Spectral)
plt.show()

# calculate sigmoid


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# calculate cost function

def compute_cost_logistic(X, y, w, b):
    m = X.shape[0]
    cost = 0.0
    for i in range(m):
        z_i = np.dot(X[i], w) + b
        f_wb_i = sigmoid(z_i)
        cost += -y[i]*np.log(f_wb_i) - (1-y[i])*np.log(1-f_wb_i)

    cost = cost / m
    return cost


# testing cost function

w_tmp = np.array([1, 1])
b_tmp = -3
print(compute_cost_logistic(x_train, y_train, w_tmp, b_tmp))

# compute gradient


def compute_gradient_logistic(X, y, w, b):
    m, n = X.shape  # rows and column
    dj_dw = np.zeros((n,))  # (n,)
    dj_db = 0.

    for i in range(m):
        f_wb_i = sigmoid(np.dot(X[i], w) + b)  # (n,)(n,)=scalar
        err_i = f_wb_i - y[i]  # scalar
        for j in range(n):
            dj_dw[j] = dj_dw[j] + err_i * X[i, j]  # scalar
        dj_db = dj_db + err_i
    dj_dw = dj_dw/m  # (n,)
    dj_db = dj_db/m  # scalar

    return dj_db, dj_dw

# this is for single feature

# def compute_gradient_logistic(X, y, w, b):
#     m = X.shape[0]
#     dj_dw = 0
#     dj_db = 0
#     for i in range(m):
#         z_i = np.dot(X[i], w) + b
#         f_wb_i = sigmoid(z_i)
#         dj_dw_i = (f_wb_i - y[i]) * X[i]
#         dj_db_i = f_wb_i - y[i]
#         dj_dw += dj_dw_i
#         dj_db += dj_db_i
#     dj_dw = dj_dw / m
#     dj_db = dj_db / m
#     return dj_dw, dj_db


# testing gradient

X_tmp = np.array([[0.5, 1.5], [1, 1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y_tmp = np.array([0, 0, 0, 1, 1, 1])
w_tmp = np.array([2., 3.])
b_tmp = 1.
dj_db_tmp, dj_dw_tmp = compute_gradient_logistic(X_tmp, y_tmp, w_tmp, b_tmp)
print(f"dj_db: {dj_db_tmp}")
print(f"dj_dw: {dj_dw_tmp.tolist()}")


# gradient descent

def gradient_descent(X, y, w_in, b_in, alpha, num_iters):

    # An array to store cost J and w's at each iteration primarily for graphing later
    J_history = []
    w = copy.deepcopy(w_in)  # avoid modifying global w within function
    b = b_in

    for i in range(num_iters):
        # Calculate the gradient and update the parameters
        dj_db, dj_dw = compute_gradient_logistic(X, y, w, b)

        # Update Parameters using w, b, alpha and gradient
        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        # Save cost J at each iteration
        if i < 100000:      # prevent resource exhaustion
            J_history.append(compute_cost_logistic(X, y, w, b))

        # Print cost every at intervals 10 times or as many iterations if < 10
        if i % math.ceil(num_iters / 10) == 0:
            print(f"Iteration {i:4d}: Cost {J_history[-1]}   ")

    return w, b, J_history  # return final w,b and J history for graphing


# testing gradient descent
w_tmp = np.zeros_like(x_train[0])
b_tmp = 0.
alph = 0.1
iters = 10000

w_out, b_out, _ = gradient_descent(x_train, y_train, w_tmp, b_tmp, alph, iters)
print(f"\nupdated parameters: w:{w_out}, b:{b_out}")


# prediction

x_test = [[0.5, 0.5], [1.5, 1.5], [3, 1]]

for i in range(len(x_test)):
    z = np.dot(x_test[i], w_out) + b_out
    y = sigmoid(z)
    print(f"input: {x_test[i]}, output: {y}")


# plot decision boundary

def plot_decision_boundary(X, y, w, b):
    x1 = np.linspace(np.min(X[:, 0]), np.max(X[:, 0]), 100)
    x2 = -(w[0] * x1 + b) / w[1]
    plt.plot(x1, x2, color='r', label='Decision Boundary')
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
    plt.legend()
    plt.show()


# plot decision boundary
plot_decision_boundary(x_train, y_train, w_out, b_out)
