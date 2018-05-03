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

x_data_scaler = preprocessing.Normalizer()
y_data_scaler = preprocessing.Normalizer()

columns_to_scale = [" n_tokens_title", " n_tokens_content", " num_hrefs", " num_self_hrefs", " num_imgs", " num_videos", " average_token_length", " num_keywords", " kw_avg_min"," kw_avg_max", " kw_avg_avg"," self_reference_avg_sharess"]

x_data_scaler.fit(training_X.loc[:, columns_to_scale])
y_data_scaler.fit(training_Y.reshape(-1, 1))

training_X.loc[:, columns_to_scale] = x_data_scaler.transform(training_X.loc[:, columns_to_scale])
training_Y = y_data_scaler.transform(training_Y.reshape(-1, 1))

testing_X.loc[:, columns_to_scale] = x_data_scaler.transform(testing_X.loc[:, columns_to_scale])
testing_Y = y_data_scaler.transform(testing_Y.reshape(-1, 1))


from neupy import algorithms, layers, estimators

# environment.reproducible()
#
# nw = algorithms.GRNN(std=0.1, verbose=True)
# nw.train(training_X, training_Y)
# y_predicted = nw.predict(testing_X)
# print(estimators.rmse(y_predicted, testing_Y))

# MLPP Scikit-Learn - RMSE 0.01567453951930472
from sklearn.neural_network import MLPRegressor
from scipy import stats
from sklearn.grid_search import RandomizedSearchCV

mlp = MLPRegressor(hidden_layer_sizes=(100, 50, 50), max_iter=2000, activation='relu', solver='lbfgs',
                   learning_rate='constant', early_stopping=True, learning_rate_init=0.01)
mlp.fit(training_X, training_Y)
y_predicted = mlp.predict(testing_X)
print(estimators.rmse(y_predicted, testing_Y.ravel()))

rs = RandomizedSearchCV(mlp, param_distributions={
    'learning_rate': ["constant", "invscaling", "adaptive"],
    'hidden_layer_sizes': [(100, 50, 50), (1000, 100, 100)],
    'learning_rate_init': stats.uniform(0.001, 0.05),
    'activation': ["relu", "tanh"],
    'alpha': stats.uniform(0.0001, 1),
    'beta_1': stats.uniform(0, 1.0),
    'beta_2': stats.uniform(0, 1.0)}, verbose=2)

rs.fit(training_X, training_Y.ravel())
bs = rs.best_estimator_

y_predicted = bs.predict(testing_X)
print(estimators.rmse(y_predicted, testing_Y.ravel()))

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

# # SOM try out
# from neupy import algorithms
#
# num_epochs = 100
# num_clusters = 3
# num_features = training_X.shape[1]
#
# sofm = algorithms.SOFM(n_inputs=num_features, n_outputs=num_clusters, step=0.1, learning_radius=0, verbose=True,
#                        grid_type='rect')
# sofm.train(training_X, epochs=num_epochs)
# print(sofm.weight.shape)
