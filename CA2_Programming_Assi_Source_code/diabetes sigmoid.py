
# coding: utf-8

# ## <font color=blue>Diabetes dataset age 70-100<font>
# 
# [Baseline Categorical](#SVM)
# 
# [One Hot](#hot)

# In[1]:


# get_ipython().magic(u'matplotlib notebook')
import argparse
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC, SVC
from tqdm import tqdm

parser = argparse.ArgumentParser(prog='Sigmoid all')
parser.add_argument('--save_dir', default='result', help='Directory for saved output')
parser.add_argument('--test_output', default=0 , help='Test saved output')
args = parser.parse_args()

if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

# In[2]:


def plot_confusion(clf, title, X_test, y_test):
    svm_predicted_mc = clf.predict(X_test)
    confusion_mc = confusion_matrix(y_test, svm_predicted_mc)
    df_cm = pd.DataFrame(confusion_mc, 
                         index = [i for i in range(0,3)], columns = [i for i in range(0,3)])

    plt.figure(figsize=(6,4))
    ax_ticks= ['<30', '>30', 'NO']
    sns.heatmap(df_cm, annot=True, xticklabels=ax_ticks, yticklabels=ax_ticks, fmt='g')
    plt.title(title + '\nAccuracy:{0:.3f}'.format(accuracy_score(y_test, 
                                                                           svm_predicted_mc)))
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(args.save_dir+'/' + title)
    plt.close()
    
    print('Micro-averaged precision = {:.2f} (treat instances equally)'
      .format(precision_score(y_test, svm_predicted_mc, average = 'micro')))
    print('Macro-averaged precision = {:.2f} (treat classes equally)'
      .format(precision_score(y_test, svm_predicted_mc, average = 'macro')))
    print('Micro-averaged f1 = {:.2f} (treat instances equally)'
          .format(f1_score(y_test, svm_predicted_mc, average = 'micro')))
    print('Macro-averaged f1 = {:.2f} (treat classes equally)'
          .format(f1_score(y_test, svm_predicted_mc, average = 'macro')))

    print(title + ' classification' + '\n' + classification_report(y_test, svm_predicted_mc, target_names=ax_ticks))


# In[3]:


df = pd.read_csv("diabetic_data.csv", dtype='category').iloc[:,2:]
df.shape


# In[4]:


df['age'].head(2)


# In[5]:

if args.test_output:
    df = df.loc[(df['age'] == '[10-20)') ]
# else:
    # df = df.loc[(df['age'] == '[70-80)') | (df['age'] == '[80-90)')
          # | (df['age'] == '[90-100)')]

# In[6]:


df.head(2)


# In[7]:


df.describe()


# In[8]:


def show_unique(dataF):
    for c in dataF.columns:
        print(c, dataF[c].unique())
        print('*'*50)

show_unique(df)


# ### <font color =blue>1. remove columns with missing data</font>

# In[9]:


df = df.drop(columns=['weight', 'payer_code', 'medical_specialty'])
df.head(2)


# ### <font color=blue> 2. remove incomplete columns and rows </font>

# In[10]:


# drop missing values
df = df.replace('?', np.nan)
df = df.replace('Unknown/Invalid', np.nan)
df = df.dropna()
df.shape


# In[11]:


df = df.drop(columns= df.columns[df.nunique() == 1])


# In[12]:


show_unique(df)


# ### <font color = blue>3. categorical variables</font>

# In[13]:


to_num = ['time_in_hospital', 'num_lab_procedures', 'num_procedures',
         'num_medications', 'number_outpatient', 'number_emergency',
         'number_inpatient', 'number_diagnoses']

to_cat_codes = list(set(df.columns) - set(to_num))


# In[14]:


X_features = list(to_num)
for c in to_cat_codes:
    df[c+'_cat'] = df[c].cat.codes        
    X_features += [c+'_cat']
    
X_features.remove('readmitted_cat')
X_features


# In[15]:


df['readmitted'].head(11)


# In[16]:


df['readmitted_cat'].head(11)


# #### <font color=red>Target Mapping: < 30 = 0, >30 = 1, NO = 2, </font>

# ### <font color=blue>4. split</font>

# In[17]:


for n in to_num:
    df[n] = df[n].astype('int')

df[to_num].dtypes


# ## <font color=green>to Categorical</font>

# In[18]:


X_train, X_test, y_train, y_test = train_test_split(
    df[X_features], df['readmitted_cat'] , random_state = 0)


# In[19]:


X_train.head(2)


# In[20]:


y_train.head(2)


# In[21]:


X_test.head(2)


# In[22]:


y_test.head(2)


# ### <font color=green>min max scaling</font>

# In[23]:


X_train.dtypes


# ### <font color=green>before scale</font>

# In[24]:


X_train.describe()


# In[25]:


X_test.describe()


# In[26]:


scaler = MinMaxScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_features)
X_test = pd.DataFrame(scaler.transform(X_test), columns = X_features)


# ### <font color=green>after scale</font>

# In[27]:


X_train.iloc[:,:20].describe()


# In[28]:


X_train.iloc[:,21:40].describe()


# In[29]:


y_train.describe()


# In[30]:


X_test.shape


# In[31]:


X_test.iloc[:,:20].describe()


# In[32]:


X_test.iloc[:,21:40].describe()


# In[33]:


y_test.describe()


# <a id='SVM'></a>

# ### <font color=green>Baseline Categorical</font>

# In[34]:


# get_ipython().run_cell_magic(u'time', u'', u"

from sklearn.dummy import DummyClassifier
d_major = DummyClassifier(strategy='most_frequent').fit(X_train, y_train)

print('train accuracy: {:.2f}'.format(d_major.score(X_train, y_train)))
print('test accuracy: {:.2f}'.format(d_major.score(X_test, y_test)))


# In[35]:


plot_confusion(d_major, 'Categorical Dummy Classifier', X_test, y_test)


# In[36]:


# get_ipython().run_cell_magic(u'time', u'', u"

clf = LinearSVC(verbose=True).fit(X_train, y_train)
print('training accuracy: {:.2f}'.format(clf.score(X_train, y_train)))
print('test accuracy: {:.2f}'.format(clf.score(X_test, y_test)))
plot_confusion(clf, 'Categorical Linear Kernel', X_test, y_test)


# ### <font color=green>SVM optimise over accuracy</font>

# In[37]:


# get_ipython().run_cell_magic(u'time', u'', u"
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
clf = SVC() 
grid_values = [ {'C': [0.1, 1, 10, 100, 1000], 'gamma':[0.001, 0.01, 0.05, 0.1, 1, 10, 100], 'kernel':['sigmoid']}] 
grid_clf_acc = GridSearchCV(clf, param_grid = grid_values, n_jobs= 4)
grid_clf_acc.fit(X_train, y_train)
y_decision_fn_scores_acc = grid_clf_acc.decision_function(X_test) 
print('Grid best parameter (max. accuracy): ', grid_clf_acc.best_params_)
print('Grid best score (accuracy): ', grid_clf_acc.best_score_)


# In[38]:


pd.DataFrame(grid_clf_acc.cv_results_).sort_values(by=['mean_test_score'], ascending=False).iloc[:,2:8]


# In[39]:


plot_confusion(grid_clf_acc, 'Categorical Grid Search Sigmoid', X_test,y_test)


# In[40]:


# precision recall curve only for binary class


# <a id='hot'></a>

# ## <font color=Orange>One Hot</font>

# In[41]:


X_train, X_test, y_train, y_test = train_test_split(
    df[X_features], df['readmitted_cat'] , random_state = 0)


# In[42]:


X_train.dtypes


# In[43]:


X_train.describe()


# In[44]:


X_test.describe()


# In[45]:


to_num


# ### <font color=orange>scale numerical</font>

# In[46]:


scaler = MinMaxScaler()
X_train_hot = pd.DataFrame(scaler.fit_transform(X_train[to_num]), columns=to_num)
X_test_hot = pd.DataFrame(scaler.transform(X_test[to_num]), columns = to_num)


# In[47]:


from sklearn.preprocessing import OneHotEncoder
hot_features = list(set(X_features) - set(to_num))
hot_features


# In[48]:


enc = OneHotEncoder()
enc.fit(df[hot_features])
enc.n_values_


# In[49]:


enc.feature_indices_


# ### <font color=orange> convert to one hot </font>

# In[50]:


X_train_hot = pd.concat([X_train_hot,                          pd.DataFrame(enc.transform(X_train[hot_features]).toarray())], axis=1)

X_test_hot = pd.concat([X_test_hot,                          pd.DataFrame(enc.transform(X_test[hot_features]).toarray())], axis=1)


# In[51]:


X_train_hot.head(2)


# In[52]:


X_test_hot.head(2)


# ### <font color=orange> SVM one hot</font>

# In[53]:


# get_ipython().run_cell_magic(u'time', u'', u"

from sklearn.dummy import DummyClassifier
d_major = DummyClassifier(strategy='most_frequent').fit(X_train_hot, y_train)
print('train accuracy: {:.2f}'.format(d_major.score(X_train_hot, y_train)))
print('test accuracy: {:.2f}'.format(d_major.score(X_test_hot, y_test)))
plot_confusion(d_major, 'One Hot Dummy Classifier', X_test_hot, y_test)


# In[ ]:


# get_ipython().run_cell_magic(u'time', u'', u"
clf = LinearSVC(verbose=True).fit(X_train_hot, y_train)
print('training accuracy: {:.2f}'.format(clf.score(X_train_hot, y_train)))
print('test accuracy: {:.2f}'.format(clf.score(X_test_hot, y_test)))
plot_confusion(clf, 'One Hot Linear Kernel', X_test_hot, y_test )


# ### <font color=orange>SVM optimise over accuracy</font>

# In[ ]:


# get_ipython().run_cell_magic(u'time', u'', u"
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
clf = SVC() 
grid_values = [ {'C': [0.1, 1, 10, 100, 1000], 'gamma':[0.001, 0.01, 0.05, 0.1, 1, 10, 100], 'kernel':['sigmoid']}] 
grid_clf_acc = GridSearchCV(clf, param_grid = grid_values, n_jobs= 4 )
grid_clf_acc.fit(X_train_hot, y_train)
y_decision_fn_scores_acc = grid_clf_acc.decision_function(X_test_hot) 
print('Grid best parameter (max. accuracy): ', grid_clf_acc.best_params_)
print('Grid best score (accuracy): ', grid_clf_acc.best_score_)


# In[ ]:


pd.DataFrame(grid_clf_acc.cv_results_).sort_values(by=['mean_test_score'], ascending=False).iloc[:,2:8]


# In[ ]:


plot_confusion(grid_clf_acc, 'One Hot Grid Search Sigmoid', X_test_hot, y_test)

