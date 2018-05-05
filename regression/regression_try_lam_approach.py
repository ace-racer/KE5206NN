import pandas as pd
from sklearn import preprocessing
from neupy import environment
from sklearn.model_selection import train_test_split
from neupy import environment

# training_df = pd.read_csv(
#     "/home/pier/Machine_Learning/KE5206NN/regression/data_with_fields_removed/train_70.0_fields_removed.csv")
training_df = pd.read_csv(
    "/home/pier/Machine_Learning/KE5206NN/regression/original_data/train_70.0.csv")
training_X = training_df.loc[:, training_df.columns != " shares"]
training_Y = training_df.loc[:, " shares"]
training_X = training_X.drop(training_X.columns[[0, 1]], axis=1)
# test_df = pd.read_csv(
#     "/home/pier/Machine_Learning/KE5206NN/regression/data_with_fields_removed/test_30.0_fields_removed.csv")
test_df = pd.read_csv(
    "/home/pier/Machine_Learning/KE5206NN/regression/original_data/test_30.0.csv")
testing_X = test_df.loc[:, test_df.columns != " shares"]
testing_X = testing_X.drop(testing_X.columns[[0, 1]], axis=1)
testing_Y = test_df.loc[:, " shares"]

# remove url

print("Training shapes")
print(training_X.shape)
print(training_Y.shape)

print("Testing shapes")
print(testing_X.shape)
print(testing_Y.shape)

# x_data_scaler = preprocessing.StandardScaler()
# y_data_scaler = preprocessing.StandardScaler()
#
# x_data_scaler.fit(training_X)
# y_data_scaler.fit(training_Y.reshape(-1, 1))
#
# training_X = x_data_scaler.transform(training_X)
# training_Y = y_data_scaler.transform(training_Y.reshape(-1, 1))
#
# testing_X = x_data_scaler.transform(testing_X)
# testing_Y = y_data_scaler.transform(testing_Y.reshape(-1, 1))
scaler = preprocessing.StandardScaler()

for n in range(training_X.shape[1]):
    training_X[n] = scaler.fit_transform(training_X[n].reshape(-1, 1))
    testing_X[n] = scaler.fit_transform(testing_X[n].reshape(-1, 1))

training_Y = scaler.fit_transform(training_X[' shares'].reshape(-1, 1))
testing_Y = scaler.fit_transform(testing_X[' shares'].reshape(-1, 1))

# MLPP Scikit-Learn - RMSE 0.01567453951930472
from sklearn.neural_network import MLPRegressor
from scipy import stats
from sklearn.grid_search import RandomizedSearchCV
from neupy import estimators

mlp = MLPRegressor(hidden_layer_sizes=(10, 10, 10), max_iter=2000, activation='tanh', solver='sgd',
                   learning_rate='constant', early_stopping=True, learning_rate_init=0.04, alpha=100, beta_1=0.616,
                   beta_2=0.194)
mlp.fit(training_X, training_Y)
y_predicted = mlp.predict(testing_X)
print("RMSE = " + str(estimators.rmse(y_predicted, testing_Y.ravel())))
print("MAE = " + str(estimators.mae(y_predicted, testing_Y.ravel())))
actual_mae = y_data_scaler.inverse_transform(estimators.mae(y_predicted, testing_Y))
print("MAE (no. of shares) = " + str(actual_mae.squeeze()))
