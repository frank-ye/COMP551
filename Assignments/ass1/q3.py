import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

na_values = ['?']
path = 'Datasets/'
raw_data = pd.read_csv(path + 'communities.data', na_values=na_values, header=None)


# Remove non-predictive data columns in dataset

# 3.1 CLEAN THE DATASET


# Define method that removes all non-predictive features


def shift_columns(dataset, pivot):
    for index in range(0, dataset.shape[1]-pivot):
        dataset[index] = dataset[index+pivot]
    for index in range(dataset.shape[1]-pivot, dataset.shape[1]):
        dataset.drop([index], axis=1, inplace=True)
    return dataset



predictors_set = shift_columns(raw_data, 5)

# following line computes the mean of every column in the dataset, replaces NaN with computed mean

predictors_set.fillna(predictors_set.mean(axis=1), inplace=True)

# Write the cleaned up data set to a .csv file
# predictors_set.to_csv('clean_set_q3.csv')



# Method to create 5 splits


def n_fold_split(dataset, n):

    for index in range(1, n+1):
        mask = np.random.rand(len(dataset)) < 0.8
        train = dataset[mask]
        test = dataset[~mask]
        train.to_csv('CandC-train'+ str(index)+'.csv')
        test.to_csv('CandC-test' + str(index) + '.csv')

    return

k = 5

n_fold_split(predictors_set, k)


### Method that splits features and outputs in all sets

def drop_columns(k):
    x_train = []
    x_test = []
    y_train = []
    y_test = []

    for i in range(0, k):
        temp_train = pd.read_csv('CandC-train'+ str(i+1)+'.csv', header=None)
        temp_test = pd.read_csv('CandC-test' + str(i+1) + '.csv', header=None)
        shift_columns(temp_train, 1)
        shift_columns(temp_test, 1)
        temp_train_y = temp_train[122]
        temp_test_y = temp_test[122]

        temp_train.drop([122], axis=1, inplace=True)
        temp_test.drop([122], axis=1, inplace=True)

        x_train.append(temp_train)
        x_test.append(temp_test)
        y_train.append(temp_train_y)
        y_test.append(temp_test_y)
    return x_train, y_train, x_test, y_test


x_train, y_train, x_test, y_test = drop_columns(5)



# Set up linear regression using all features



def findweights(polymatrix, trainingoutput, regularization):
    XT = polymatrix.T
    XTX = np.matmul(XT, polymatrix)
    # XTX_inv = np.linalg.pinv(XTX)
    w_left = XTX + regularization * np.identity(XTX.shape[1])
    w_left_inv = np.linalg.pinv(w_left)
    fit_weights = np.matmul(np.matmul(w_left_inv, XT), trainingoutput)
    return fit_weights



def mse(polymatrix, weights, output):
    test_matrix = np.matmul(polymatrix, weights)
    square_error = np.power(np.subtract(test_matrix, output),2)
    error = square_error.mean()
    return error



# Method to perform k-fold cross-validation, returning the average test MSE, and the parameters learned for each model


def k_fold_cross_validation(train_x, train_y, test_x, test_y, penalty):
    params = []
    mse_test = []
    for i in range(0, len(train_x)):
        temp_weights = findweights(train_x[i], train_y[i], penalty)
        params.append(temp_weights)
        mse_test.append(mse(test_x[i], temp_weights, test_y[i]))
    mse_mean = np.asarray(mse_test).mean()
    return mse_mean, params


k_fold_mean, k_fold_weights = k_fold_cross_validation(x_train, y_train, x_test, y_test, 0)
print(k_fold_mean)

# print(k_fold_weights)


### Use ridge regression on above data


def k_fold_valid_ridge(train_x, train_y, test_x, test_y, precision):
    lambda_param = []
    lambda_mse_mean = 10000
    best_lambda = 0
    for i in range(0, 10*precision):
        test_lambda = i/precision
        some_mean, some_params = k_fold_cross_validation(train_x, train_y, test_x, test_y, test_lambda)
        if some_mean < lambda_mse_mean:
            lambda_mse_mean = some_mean
            lambda_param = some_params
            best_lambda = test_lambda
    return lambda_mse_mean, best_lambda, lambda_param



def plot_fitted_weights(input, output, weights, order, lowerbound, upperbound):

    x_axis = pd.DataFrame(np.ones(100))
    x_axis[1] = np.arange(lowerbound, upperbound, ((upperbound - lowerbound) / 100))
    # for i in range(2, order + 1):
    #     x_axis[i] = pow(x_axis[1], i)
    # y_plot = np.matmul(x_axis, weights)
    plt.scatter(input, output)
    # plt.plot(x_axis[1], y_plot, '-r')
    plt.show()


# print(x_test[0].shape[1])
# print(y_test[0])
# y_plot = np.matmul(np.asarray(x_test[0]).T, y_test[0])
# print(y_plot)

# print(np.asarray(k_fold_weights[0]).T.shape[1])
fitted_weights = findweights(x)
plot_fitted_weights(x_test[0], y_test[0], np.asarray(k_fold_weights[0]), 121, 0, 10)


def basic_plot(x_axis, y_axis):
    plt.plot(x_axis, y_axis, '-r')
    plt.show()
    return


basic_plot(x_test[0], y_test[0])

# l2_mean, l2_lambda, lambda_params = k_fold_valid_ridge(x_train, y_train, x_test, y_test, 1000)

# print(l2_mean, l2_lambda)
# print(lambda_params)
