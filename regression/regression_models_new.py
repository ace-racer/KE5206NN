import pandas as pd
from sklearn import preprocessing
from neupy import environment
from sklearn.model_selection import train_test_split
from neupy import environment

# don't think we should drop timedelta as number of shares is a factor of time passed
# since article was published

training_df = pd.read_csv(
    "/home/pier/Machine_Learning/KE5206NN/regression/data_with_fields_removed/train_70.0_fields_removed.csv")
training_X = training_df.loc[:, training_df.columns != " shares"]
training_Y = training_df.loc[:, " shares"]

test_df = pd.read_csv(
    "/home/pier/Machine_Learning/KE5206NN/regression/data_with_fields_removed/test_30.0_fields_removed.csv")
testing_X = test_df.loc[:, test_df.columns != " shares"]
testing_Y = test_df.loc[:, " shares"]

print("Training shapes")
print(training_X.shape)
print(training_Y.shape)

print("Testing shapes")
print(testing_X.shape)
print(testing_Y.shape)

data_scaler = preprocessing.MinMaxScaler()

# columns_to_scale = [" n_tokens_title", " n_tokens_content", " num_hrefs", " num_self_hrefs", " num_imgs", " num_videos",
#                     " average_token_length", " num_keywords", " kw_avg_min", " kw_avg_max", " kw_avg_avg",
#                     " self_reference_avg_sharess"]
# training_X.loc[:, columns_to_scale] = data_scaler.fit_transform(training_X.loc[:, columns_to_scale])
# training_Y = data_scaler.fit_transform(training_Y.reshape(-1, 1))
#
# testing_X.loc[:, columns_to_scale] = data_scaler.fit_transform(testing_X.loc[:, columns_to_scale])
# testing_Y = data_scaler.fit_transform(testing_Y.reshape(-1, 1))
training_X = data_scaler.fit_transform(training_X)
testing_X = data_scaler.fit_transform(testing_X)
training_Y = data_scaler.fit_transform(training_Y.reshape(-1, 1))
testing_Y = data_scaler.fit_transform(testing_Y.reshape(-1, 1))


from neupy import algorithms, layers, estimators

environment.reproducible()

nw = algorithms.GRNN(std=0.1, verbose=True)
nw.train(training_X, training_Y)
y_predicted = nw.predict(testing_X)
print(estimators.rmse(y_predicted, testing_Y))
