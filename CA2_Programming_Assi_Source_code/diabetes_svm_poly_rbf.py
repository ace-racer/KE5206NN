import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, \
    classification_report
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA
import os

enable_one_hot = True
enable_scaling = True

# Load Data
# os.chdir("/home/pier/Machine_Learning/KE5206NN/diabetes_svm")
os.chdir("/Users/pierlim/PycharmProjects/KE5206NN/diabetes_svm")
dfs = pd.read_excel("data/diabetic_data.xlsx", sheet_name=None)
df = dfs['in']
df = df.iloc[:, 2:]
print(df.shape)

# Data Exploration
corr = df.corr()
sns.heatmap(corr,
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)
plt.show()
df.describe()

df_numeric = df.select_dtypes(include=[np.number])

# drop missing values
df = df.replace('?', np.nan)
df = df.replace('Unknown/Invalid', np.nan)
print(df.columns[df.isnull().any()])
df.isnull().sum()
# df = df.dropna() # dangerous! dropped until left 1k plus rows

print(df.shape)
df = df.drop(columns=df.columns[df.nunique() == 1])  # drop columns which only have 1 category

to_num = ['time_in_hospital', 'num_lab_procedures', 'num_procedures',
          'num_medications', 'number_outpatient', 'number_emergency',
          'number_inpatient', 'number_diagnoses']

to_cat_codes = list(set(df.columns) - set(to_num))
df_test = df

# X_features = list(to_num)
# for c in to_cat_codes:
#     df_test[c + '_cat'] = df_test[c].cat.codes
#     X_features += [c + '_cat']
#
# X_features.remove('readmitted_cat')

obj_df = df.select_dtypes(include=['object']).copy()
obj_df.head()
df = df.fillna({'weight': df['weight'].value_counts().index[0]})
df = df.fillna({'payer_code': 'NOT_SPECIFIED', 'medical_specialty': 'NOT_SPECIFIED'})

enable_categorization = True
if enable_categorization:
    df['age'] = df['age'].astype('category')
    df['age'] = df['age'].cat.codes
    df['weight'] = df['weight'].astype('category')
    df['weight'] = df['weight'].cat.codes
    df['readmitted'] = df['readmitted'].astype('category')
    df['readmitted'] = df['readmitted'].cat.codes

if enable_one_hot:
    # one hot the rest
    column_names = list(df.select_dtypes(include=['object']).columns.values)
    one_hot = pd.get_dummies(df.select_dtypes(include=['object']))
    df = df.drop(column_names, axis=1)
    df = df.join(one_hot)

df_x = df.loc[:, df.columns != 'readmitted']
df_y = df.loc[:, df.columns == 'readmitted']

X_train, X_test, y_train, y_test = train_test_split(
    df_x, df_y, test_size=0.3, random_state=42)

from sklearn.svm import SVC

if enable_scaling:
    scaler = MinMaxScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train))
    X_test = pd.DataFrame(scaler.transform(X_test))
# for duality in [True, False]:
#     for var_c in [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]:
#         print("var c = %d and duality is %d" % (var_c, duality))
#         clf = LinearSVC(max_iter=5000, dual=duality, C=var_c, random_state=42).fit(X_train,
#                                                                                    y_train.values.ravel())
#         # clf = SVC(kernel='linear', class_weight='balanced', C=1.0, random_state=42, verbose=1).fit(X_train,
#         #                                                                                           y_train.values.ravel())
#         print('training accuracy: {:.2f}'.format(clf.score(X_train, y_train)))
#         print('test accuracy: {:.2f}'.format(clf.score(X_test, y_test)))

clf = LinearSVC(max_iter=5000, dual=False, C=10, random_state=42).fit(X_train, y_train.values.ravel())
print('training accuracy: {:.2f}'.format(clf.score(X_train, y_train)))
print('test accuracy: {:.2f}'.format(clf.score(X_test, y_test)))

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVC

# Memory Error!
# polynomial_svm = Pipeline([
#     ("poly_features", PolynomialFeatures(degree=3)),
#     ("scaler", MinMaxScaler()),
#     ("svm_clf", LinearSVC(C=10, loss="hinge", random_state=42))
# ])
#
# polynomial_svm.fit(X_train, y_train)
# print('training accuracy: {:.2f}'.format(clf.score(X_train, y_train)))
# print('test accuracy: {:.2f}'.format(clf.score(X_test, y_test)))

poly_kernel_svm_clf = Pipeline([
    ("scaler", MinMaxScaler(feature_range=(-1, 1))),
    ("svm_clf", SVC(kernel="poly", degree=3, coef0=1, C=5, verbose=1))
])
poly_kernel_svm_clf.fit(X_train, y_train.values.ravel())
print('training accuracy: {:.3f}'.format(poly_kernel_svm_clf.score(X_train, y_train.values.ravel())))
print('test accuracy: {:.3f}'.format(poly_kernel_svm_clf.score(X_test, y_test.values.ravel())))


def plot_confusion(clf):
    svm_predicted_mc = clf.predict(X_test)
    confusion_mc = confusion_matrix(y_test, svm_predicted_mc)
    df_cm = pd.DataFrame(confusion_mc,
                         index=[i for i in range(0, 3)], columns=[i for i in range(0, 3)])

    plt.figure(figsize=(6, 4))
    ax_ticks = ['<30', '>30', 'NO']
    sns.heatmap(df_cm, annot=True, xticklabels=ax_ticks, yticklabels=ax_ticks, fmt='g')
    plt.title('RBF Kernel \nAccuracy:{0:.3f}'.format(accuracy_score(y_test,
                                                                           svm_predicted_mc)))
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    print('Micro-averaged precision = {:.2f} (treat instances equally)'
          .format(precision_score(y_test, svm_predicted_mc, average='micro')))
    print('Macro-averaged precision = {:.2f} (treat classes equally)'
          .format(precision_score(y_test, svm_predicted_mc, average='macro')))
    print('Micro-averaged f1 = {:.2f} (treat instances equally)'
          .format(f1_score(y_test, svm_predicted_mc, average='micro')))
    print('Macro-averaged f1 = {:.2f} (treat classes equally)'
          .format(f1_score(y_test, svm_predicted_mc, average='macro')))
    print(classification_report(y_test, svm_predicted_mc))


plot_confusion(poly_kernel_svm_clf)


import _pickle
with open('/Users/pierlim/PycharmProjects/KE5206NN/diabetes_svm/poly_kernel.pkl', 'wb') as fid:
    _pickle.dump(poly_kernel_svm_clf, fid)

# ...............................................................................................
# Warning: using -h 0 may be faster
# *.............................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................*..................*
# optimization finished, #iter = 894467
# obj = -79154.995305, rho = 0.916367
# nSV = 21294, nBSV = 11991
# ...................................................................................................................
# Warning: using -h 0 may be faster
# *......................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................*.................*
# optimization finished, #iter = 665365
# obj = -79224.925456, rho = 0.746238
# nSV = 20465, nBSV = 12927
# ..............................................................*.........................*..*
# optimization finished, #iter = 88652
# obj = -234873.372288, rho = -6.361463
# nSV = 48352, nBSV = 47034
# Total nSV = 59022
# [LibSVM]training accuracy: 0.58
# test accuracy: 0.57

for var_c in [0.1, 1, 10, 50, 100]:
    rbf_kernel_svm_clf = Pipeline([
        ("scaler", MinMaxScaler(feature_range=(-1, 1))),
        ("svm_clf", SVC(kernel="rbf", C=var_c))
    ])
    rbf_kernel_svm_clf.fit(X_train, y_train.values.ravel())
    print('training accuracy: {:.3f}'.format(rbf_kernel_svm_clf.score(X_train, y_train.values.ravel())))
    print('test accuracy: {:.3f}'.format(rbf_kernel_svm_clf.score(X_test, y_test.values.ravel())))

