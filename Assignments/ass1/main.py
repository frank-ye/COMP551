import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ASK ABOUT MSE VS LAMBDA PLOT

# Read sets from .csv files
train_set1 = pd.read_csv('hwk1_datasets/Datasets/Dataset_1_train.csv', header=None)
val_set1 = pd.read_csv('hwk1_datasets/Datasets/Dataset_1_valid.csv', header=None)
test_set1 = pd.read_csv('hwk1_datasets/Datasets/Dataset_1_test.csv', header=None)

# Drop "3rd column", axis = 1 --> column
train_set1.drop([2], axis=1, inplace=True)
val_set1.drop([2], axis=1, inplace=True)
test_set1.drop([2], axis=1, inplace=True)

# Create y vectors to validate sets later
y_train = train_set1[1]
y_val = val_set1[1]
y_test = test_set1[1]


# Separate features and o/p in validation set, training set, test set
val_set1.drop([1], axis=1, inplace=True)
val_set1[1] = 1
val_set1[0], val_set1[1] = val_set1[1], val_set1[0]
train_set1.drop([1], axis=1, inplace=True)
train_set1[1] = 1
train_set1[0], train_set1[1] = train_set1[1], train_set1[0]


test_set1.drop([1], axis=1, inplace=True)
test_set1[1] = 1
test_set1[0], test_set1[1] = test_set1[1], test_set1[0]


# Note to self: inplace = False returns a new dataFrame

# Method for generating an n-th order fit


def nthorderfit(inputset, order):
    for i in range(1, order):
        inputset[i + 1] = np.power(inputset[1], i + 1)
    return inputset


# create 20th-order polynomial fit
order = 20
train_set1 = nthorderfit(train_set1, order)
val_set1 = nthorderfit(val_set1, order)
test_set1 = nthorderfit(test_set1, order)

# print(val_set1)
# print(train_set1)



# w = XTX_inv * XTy
XT = train_set1.T
XTX = np.matmul(XT, train_set1)
XTX_inv = np.linalg.pinv(XTX)

# print(XTX_inv)

# Method to find weights given an n-order fit matrix, a desired training output (column of y) and desired lambda


def findweights(polymatrix, trainingoutput, regularization):
    XT = polymatrix.T
    XTX = np.matmul(XT, polymatrix)
    # XTX_inv = np.linalg.pinv(XTX)
    w_left = XTX + regularization * np.identity(XTX.shape[1])
    w_left_inv = np.linalg.pinv(w_left)
    fit_weights = np.matmul(np.matmul(w_left_inv, XT), trainingoutput)
    return fit_weights


weights = findweights(train_set1, y_train, 0)
# print(weights)


# Method to find mean-square error


def mse(polymatrix, weights, output):
    test_matrix = np.matmul(polymatrix, weights)
    square_error = np.power(np.subtract(test_matrix, output),2)
    error = square_error.mean()
    return error


# Training error uses nth-order fit on training X, training y
# Validation error uses nth-order fit on validation X, validation y

MSE_train = mse(train_set1, weights, y_train)
MSE_val = mse(val_set1, weights, y_val)

print(MSE_train)
print(MSE_val)




# Plot the fit
x_axis = pd.DataFrame(np.ones(100))
x_axis[1] = np.arange(-1, 1, 2 / 100)
for i in range(2, order + 1):
    x_axis[i] = pow(x_axis[1], i)
y_plot = np.matmul(x_axis, weights)

plt.scatter(train_set1[1], y_train)
plt.plot(x_axis[1], y_plot, '-r')
plt.show()


# 1.2 APPLY L2 REGRESSION:

# Method to find lambda by looping through possible values, best lambda is one corresponding to lowest validation MSE
# Returns optimal lambda, lowest validation MSE, and corresponding training MSE for that lambda



def lambda_optimal( trainingset, validationset, trainoutput, validationoutput, precision):
    lambda_opt = 0
    mse_opt_val = 100000000 # large arbitrary value
    mse_opt_train = 100000000
    for i in range(1, precision):

        test_lambda = i/precision
        test_weights = findweights(trainingset, trainoutput, test_lambda)
        mse_test = mse(validationset, test_weights, validationoutput)
        if mse_test < mse_opt_val:
            mse_opt_val = mse_test
            lambda_opt = test_lambda
            mse_opt_train = mse(trainingset, test_weights, trainoutput)
    return lambda_opt, mse_opt_train, mse_opt_val


precision = 10000
optimal_lambda, mse_opt_train, mse_opt_val = lambda_optimal(train_set1, val_set1, y_train, y_val, precision)

print(optimal_lambda, mse_opt_train, mse_opt_val)

# Plot the fit for the optimal_lambda

optimal_weights = findweights(train_set1, y_train, optimal_lambda)

x_axis = pd.DataFrame(np.ones(100))

x_axis[1] = np.arange(-1, 1, 2 / 100)
for i in range(2, order + 1):
    x_axis[i] = pow(x_axis[1], i)
y_plot = np.matmul(x_axis, optimal_weights)

plt.scatter(test_set1[1], y_test)
plt.plot(x_axis[1], y_plot, '-r')
plt.show()


# Visualize the MSE vs lambda, as lambda goes from 0 to 1.

plot_precision = 1000
# x_axis = pd.DataFrame(np.ones(plot_precision))

x_axis = np.arange(0, 1, 1/plot_precision)
mse_train = []
mse_val = []
mse_test = []

for i in range(0, plot_precision):

    currentweights = findweights(train_set1, y_train, i/plot_precision)
    mse_train.append(mse(train_set1, currentweights, y_train))
    mse_val.append(mse(val_set1, currentweights, y_val))
    mse_test.append(mse(test_set1, currentweights, y_test))

plt.plot(x_axis, mse_train, '-b')
plt.show()

plt.plot(x_axis, mse_val, '-o')
plt.show()

print(mse(test_set1, optimal_weights, y_test))




