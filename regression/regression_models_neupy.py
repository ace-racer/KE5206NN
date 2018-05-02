import pandas as pd
from sklearn import preprocessing
from neupy import environment
from sklearn.model_selection import train_test_split
from neupy import environment

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

data_scaler = preprocessing.MinMaxScaler()

columns_to_scale = [" n_tokens_title", " n_tokens_content", " num_hrefs", " num_self_hrefs", " num_imgs", " num_videos",
                    " average_token_length", " num_keywords", " kw_avg_min", " kw_avg_max", " kw_avg_avg",
                    " self_reference_avg_sharess"]
training_X.loc[:, columns_to_scale] = data_scaler.fit_transform(training_X.loc[:, columns_to_scale])
training_Y = data_scaler.fit_transform(training_Y.reshape(-1, 1))
#
testing_X.loc[:, columns_to_scale] = data_scaler.fit_transform(testing_X.loc[:, columns_to_scale])
# testing_Y = data_scaler.fit_transform(testing_Y.reshape(-1, 1))
# training_X = data_scaler.fit_transform(training_X)
# testing_X = data_scaler.fit_transform(testing_X)
# training_Y = data_scaler.fit_transform(training_Y.reshape(-1, 1))
testing_Y = data_scaler.fit_transform(testing_Y.reshape(-1, 1))

from neupy import algorithms, layers, estimators

# environment.reproducible()
#
# nw = algorithms.GRNN(std=0.1, verbose=True)
# nw.train(training_X, training_Y)
# y_predicted = nw.predict(testing_X)
# print(estimators.rmse(y_predicted, testing_Y))

# MLPP Scikit-Learn - RMSE 0.01567453951930472
from sklearn.neural_network import MLPRegressor
mlp = MLPRegressor(hidden_layer_sizes=(100, 50, 50), max_iter=2000, activation='relu', solver='adam',
                   learning_rate='adaptive', early_stopping=True)
mlp.fit(training_X, training_Y)
y_predicted = mlp.predict(testing_X)
print(estimators.rmse(y_predicted, testing_Y))

# Neupy - RMSE 0.015445209155545404
environment.reproducible()
from neupy import algorithms, layers

cgnet = algorithms.ConjugateGradient(
    connection=[
        layers.Input(training_X.shape[1]),
        layers.Sigmoid(50),
        layers.Sigmoid(1),
    ],
    search_method='golden',
    show_epoch=1,
    verbose=True,
    addons=[algorithms.LinearSearch],
    step=0.01,
)

cgnet.train(training_X, training_Y, testing_X, testing_Y, epochs=200)
from neupy import plots

plots.error_plot(cgnet)
y_predicted = cgnet.predict(testing_X)
print(estimators.rmse(y_predicted, testing_Y))
