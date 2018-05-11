import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# All methods are defined first, then all questions are answered below

# Read sets from .csv files
train_set1 = pd.read_csv('Datasets/Dataset_2_train.csv', header=None)
val_set1 = pd.read_csv('Datasets/Dataset_2_valid.csv', header=None)
test_set1 = pd.read_csv('Datasets/Dataset_2_test.csv', header=None)

# Drop "3rd column", axis = 1 --> column

train_set1.drop([2], axis=1, inplace=True)
val_set1.drop([2], axis=1, inplace=True)
test_set1.drop([2], axis=1, inplace=True)

# Create y vectors to validate sets later

y_train = train_set1[1]
y_val = val_set1[1]
y_test = test_set1[1]

# Separate features and o/p in validation set
val_set1.drop([1], axis=1, inplace=True)
train_set1.drop([1], axis=1, inplace=True)
test_set1.drop([1], axis=1, inplace=True)
# Add column of 1s to test and validation sets for w0, creating "polymatrix"

val_set1[1] = 1
val_set1[0], val_set1[1] = val_set1[1], val_set1[0]

train_set1[1] = 1
train_set1[0], train_set1[1] = train_set1[1], train_set1[0]

test_set1[1] = 1
test_set1[0], test_set1[1] = test_set1[1], test_set1[0]


### Basic plotting functions

def basic_plot(x_axis, y_axis):
    plt.plot(x_axis, y_axis, '-r')
    plt.show()
    return


def basic_plot_names(x_axis, y_axis):
    plt.plot(x_axis, y_axis, '-r')
    plt.title('MSE vs Number of Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('MSE')
    plt.show()
    return


def plot_both(x_axis, y1, y2):
    plt.plot(x_axis, y1, 'bo', x_axis, y2, 'ro')
    plt.show()

# basic_plot(train_set1[1], y_train)

# Plot data to approximate best fit model, use polyfit to validate weights provided by SGD.
# From the plot, the model seems to be linear with Gaussian noise


# Method to plot

def plot_fitted_weights(input, output, weights, order, lowerbound, upperbound, epoch):

    x_axis = pd.DataFrame(np.ones(100))
    x_axis[1] = np.arange(lowerbound, upperbound, ((upperbound - lowerbound) / 100))
    for i in range(2, order + 1):
        x_axis[i] = pow(x_axis[1], i)
    y_plot = np.matmul(x_axis, weights)
    plt.scatter(input, output)
    plt.title('SGD fit vs Validation Data for ' + str(epoch) + ' Epochs')
    plt.plot(x_axis[1], y_plot, '-r', label = str(weights[1]) + 'x + ' + str(weights[0]))
    plt.show()


weights_test = np.polyfit(train_set1[1], y_train, 1)


def rand_vector(size):
    return np.random.rand(size)


# Create arbitrary weight vector to start online-SGD
initial_weights = rand_vector(2)

# Method to find mean-square error


def mse(polymatrix, weights, output):
    test_matrix = np.matmul(polymatrix, weights)
    square_error = np.power(np.subtract(test_matrix, output),2)
    error = square_error.mean()
    return error

# Define helper method for gradient descent error calculation


def grad_error( x, y, w):
    x_t = x.T
    error_left = np.matmul(np.matmul(x_t, x), w)
    error_right = np.matmul(x_t, y)
    return 2*(error_left - error_right)


# Define method to perform online stochastic gradient descent to find optimize weights for linear regression
# The hyper-parameters for this function are the step-size, and the number of epochs (could also be substituted by
# an epsilon to stop computing)


def online_sgd( trainin, trainout, valin, valout, weights, step_size, num_epoch):
    mse_train_unit =[]
    mse_val_unit =[]
    mse_train_epoch = []
    mse_val_epoch = []
    updated_weights = weights
    for i in range(1, num_epoch):
        for i in range(1, trainin.shape[0]):
            mse_train_unit.append(mse(trainin,updated_weights, trainout))
            mse_val_unit.append(mse(valin, updated_weights, valout))
            updated_weights -= step_size * grad_error(trainin, trainout, updated_weights)
        mse_train_epoch.append(np.asarray(mse_train_unit).mean())
        mse_val_epoch.append(np.asarray(mse_val_unit).mean())
    return updated_weights, mse_train_epoch, mse_val_epoch


# Given hyper-param
alpha = 1e-6

# Arbitrary number of epochs to find appropriate weights for linear regression model for the given step-size.
num_epoch = 10000


# 2.1. FIND LINEAR REGRESSION MODEL USING STEP-SIZE 1E-6.

# w_alpha, mse_train_a, mse_val_a = online_sgd(train_set1, y_train, val_set1, y_val, initial_weights, alpha,num_epoch)
# print(" SGD weights: ")
# print(w_alpha)
# print("Ideal weights: ")
# print(weights_test)

# PLOT LEARNING CURVE FOR TRAINING MSE  AND VALIDATION MSE FOR EVERY EPOCH

x_axis_epoch = []
for i in range(1, num_epoch):
    x_axis_epoch.append(i)


# PLOT THE TRAINING MSE VS EPOCHS

# basic_plot(x_axis_epoch, mse_train_a)

# PLOT THE VALIDATION MSE VS EPOCHS

# basic_plot(x_axis_epoch, mse_val_a)

# PLOT THE LINEAR FIT USING THE WEIGHTS FROM ONLINE SGD USING ALPHA = 1E-6, NUM_EPOCH = 100

# plot_fitted_weights(val_set1[1], y_val, w_alpha, 1, 0, 2, num_epoch)



# 2.2 TRY DIFFERENT STEP SIZES, CHOOSE BEST STEP SIZE USING VALIDATION DATA

# w_alpha, mse_train, mse_test = online_sgd(train_set1, y_train, test_set1, y_test, initial_weights, alpha,num_epoch)
# print("For " + str(alpha) +" The minimum validation MSE is " )
# print(np.array(mse_test).min())
#
# alpha_2 = 5e-6
#
# w_alpha_2, mse_train_2, mse_test_2 = online_sgd(train_set1, y_train, test_set1, y_test, initial_weights, alpha_2,num_epoch)
# print("For " + str(alpha_2) +" The minimum validation MSE is ")
# print(np.array(mse_test_2).min())
#
# alpha_3 = 1e-7
#
# w_alpha_3, mse_train_3, mse_test_3 = online_sgd(train_set1, y_train, test_set1, y_test, initial_weights, alpha_3,num_epoch)
# print("For " + str(alpha_3) +" The minimum validation MSE is ")
# print(np.array(mse_test_3).min())
#
# alpha_4 = 1e-5
#
# w_alpha_4, mse_train_4, mse_test_4 = online_sgd(train_set1, y_train, test_set1, y_test, initial_weights, alpha_3,num_epoch)
# print("For " + str(alpha_4) +" The minimum validation MSE is ")
# print(np.array(mse_test_4).min())




# The best alpha seems to be 5e-6, more on this in the report

# REPORT TEST MSE OF THE CHOSEN MODEL.

# mse_test = mse(test_set1, [3.63628729, 4.2504195], y_test)
# print(mse_test)

# 2.3 VISUALIZE THE FIT FOR EVERY EPOCH, REPORT 5 VISUALIZATIONS SHOWING THE EVOLUTION OF THE REGRESSION FIT

num_epoch = 1
w, mse_train, mse_val = online_sgd(train_set1, y_train, val_set1, y_val, initial_weights, 5e-6,num_epoch)
# print(w)
# print(np.array(mse_val).min())
plot_fitted_weights(val_set1[1], y_val, w, 1, 0, 2, num_epoch)





# plot_fitted_weights(val_set1[1], y_val, sgd_weights_1, 1, 0, 2, )
# plot_fitted_weights(val_set1[1], y_val, sgd_weights_2, 1, 0, 2, 1000)
# plot_fitted_weights(val_set1[1], y_val, sgd_weights_3, 1, 0, 2, 2000)
# plot_fitted_weights(val_set1[1], y_val, sgd_weights_4, 1, 0, 2, 5000)
# plot_fitted_weights(val_set1[1], y_val, sgd_weights_5, 1, 0, 2, 10000)




