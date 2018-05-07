import pandas as pd
from sklearn import preprocessing
from neupy import environment
from sklearn.model_selection import train_test_split
from neupy import environment
from operator import itemgetter
import numpy as np
from neupy import algorithms, layers, estimators
from sklearn import datasets, grid_search
from sklearn.model_selection import GridSearchCV


def scorer(actual, predicted):
    return estimators.rmse(predicted, actual)

# training_df = pd.read_csv(
#     "/home/pier/Machine_Learning/KE5206NN/regression/data_with_fields_removed/train_70.0_fields_removed.csv")
training_df = pd.read_csv(
    "/Users/pierlim/PycharmProjects/KE5206NN/regression/data_with_fields_removed/train_70.0_updated.csv")
training_X = training_df.loc[:, training_df.columns != " shares"]
training_Y = training_df.loc[:, " shares"]

# test_df = pd.read_csv(
#     "/home/pier/Machine_Learning/KE5206NN/regression/data_with_fields_removed/test_30.0_fields_removed.csv")
test_df = pd.read_csv(
    "/Users/pierlim/PycharmProjects/KE5206NN/regression/data_with_fields_removed/test_30.0_updated.csv")
testing_X = test_df.loc[:, test_df.columns != " shares"]
testing_Y = test_df.loc[:, " shares"]

print("Training shapes")
print(training_X.shape)
print(training_Y.shape)

print("Testing shapes")
print(testing_X.shape)
print(testing_Y.shape)

x_data_scaler = preprocessing.MinMaxScaler()
y_data_scaler = preprocessing.MinMaxScaler()

columns_to_scale = [" n_tokens_title", " n_tokens_content", " num_hrefs", " num_self_hrefs", " num_imgs", " num_videos", " average_token_length", " num_keywords", " kw_avg_min"," kw_avg_max", " kw_avg_avg"," self_reference_avg_sharess"]

x_data_scaler.fit(training_X.loc[:, columns_to_scale])
y_data_scaler.fit(training_Y.reshape(-1, 1))

training_X.loc[:, columns_to_scale] = x_data_scaler.transform(training_X.loc[:, columns_to_scale])
training_Y = y_data_scaler.transform(training_Y.reshape(-1, 1))

testing_X.loc[:, columns_to_scale] = x_data_scaler.transform(testing_X.loc[:, columns_to_scale])
testing_Y = y_data_scaler.transform(testing_Y.reshape(-1, 1))


from sklearn.neural_network import MLPRegressor
from scipy import stats
from sklearn.grid_search import RandomizedSearchCV
from neupy import estimators
import _pickle

with open('/Users/pierlim/PycharmProjects/KE5206NN/regression/regression_models/multi_layer_perceptron.pkl', 'rb') as fid:
    mlp = _pickle.load(fid)

with open('/Users/pierlim/PycharmProjects/KE5206NN/regression/regression_models/grnn.pkl', 'rb') as fid:
    grnn = _pickle.load(fid)

y_mlp_predicted = mlp.predict(testing_X)
print("MLP RMSE = " + str(estimators.rmse(y_mlp_predicted, testing_Y.ravel())))
print("MLP MAE = " + str(estimators.mae(y_mlp_predicted, testing_Y.ravel())))
actual_mae = y_data_scaler.inverse_transform(estimators.mae(y_mlp_predicted, testing_Y))
print("MLP MAE (no. of shares) = " + str(actual_mae.squeeze()))

y_grnn_predicted = grnn.predict(testing_X)
print("GRNN RMSE = " + str(estimators.rmse(y_grnn_predicted, testing_Y.ravel())))
print("GRNN MAE = " + str(estimators.mae(y_grnn_predicted, testing_Y.ravel())))
actual_mae = y_data_scaler.inverse_transform(estimators.mae(y_grnn_predicted, testing_Y))
print("GRNN MAE (no. of shares) = " + str(actual_mae.squeeze()))

models = [mlp, grnn]
y_ensemble_predicted = np.zeros(shape=(testing_X.shape[0], 1))
print(y_ensemble_predicted.shape)
denom = 0

for model in models:
    y_predicted = model.predict(testing_X)
    mae = estimators.mae(y_predicted, testing_Y.ravel())
    denom = denom + (1.0 / (1+mae))
    y_ensemble_predicted = np.add(y_ensemble_predicted, ((1.0 / (1+mae)) * y_predicted))

print(y_ensemble_predicted.shape)
print(y_ensemble_predicted[1:10])


