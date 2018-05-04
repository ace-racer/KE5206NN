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
    "/home/pier/Machine_Learning/KE5206NN/regression/data_with_fields_removed/train_70.0_updated.csv")
training_X = training_df.loc[:, training_df.columns != " shares"]
training_Y = training_df.loc[:, " shares"]

# test_df = pd.read_csv(
#     "/home/pier/Machine_Learning/KE5206NN/regression/data_with_fields_removed/test_30.0_fields_removed.csv")
test_df = pd.read_csv(
    "/home/pier/Machine_Learning/KE5206NN/regression/data_with_fields_removed/test_30.0_updated.csv")
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


environment.reproducible()

# best_model = None
# best_error = 99
# for std_in in np.array([0.01, 0.1, 0.5, 0.8, 1.0, 1.2, 1.4, 2.0, 5.0, 10.0]):
#     grnnet = algorithms.GRNN(std=std_in, verbose=True)
#     grnnet.train(training_X, training_Y)
#     predicted = grnnet.predict(testing_X)
#     error = scorer(testing_Y, predicted)
#     print("GRNN RMSE = {:.5f}\n".format(error))
#     if error < best_error:
#         print("New Best Error Found: " + str(error))
#         best_error = error
#         best_model = grnnet

grnnet2 = algorithms.GRNN(std=0.1, verbose=True)
grnnet2.train(training_X, training_Y)
y_predicted = grnnet2.predict(testing_X)
print("RMSE = " + str(estimators.rmse(y_predicted, testing_Y.ravel())))
print("MAE = " + str(estimators.mae(y_predicted, testing_Y.ravel())))
actual_mae = y_data_scaler.inverse_transform(estimators.mae(y_predicted, testing_Y))
print("MAE (no. of shares) = " + str(actual_mae.squeeze()))

# GRNN Best values
#
# RMSE = 0.019625235702837703
# MAE = 0.004645349837893423
# MAE (no. of shares) = 3208.1448827317818