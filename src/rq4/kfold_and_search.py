from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn import metrics
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC
import seaborn as sn
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_validate 
from sklearn.datasets import make_classification
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# from light_labyrinth.dim2 import LightLabyrinthClassifier

import pandas as pd 
import numpy as np
import random

dropNaN = True
VERBOSE = True
VERBOSE_LVL = 10
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

df = pd.read_csv('/Users/awestfalewicz/Projects/2022-internship-andrzej-westfalewicz/mined_features_for_most_common_projects.csv')
if dropNaN:
    df = df.dropna()
fixes = df[df['is_vulnerability_fix']] 
fixes_p = fixes.sample(frac = 0.7)
fixes_n = fixes.drop(fixes_p.index)


not_fixes = df[df['is_vulnerability_fix'] == False] 
not_fixes_p = not_fixes.sample(n=fixes_p.shape[0]*10)
not_fixes_n = not_fixes.drop(not_fixes_p.index)

print('fixes     in train',fixes_p.shape[0])
print('fixes     in test ',fixes_n.shape[0])
print('non-fixes in train',not_fixes_p.shape[0])
print('non-fixes in test ',not_fixes_n.shape[0])
train = pd.concat([fixes_p, not_fixes_p])
test = pd.concat([fixes_n, not_fixes_n])

features = []
features.append('hours_diff') # drop by around 0.01++
features.append('changed_files') # drop by around 0.01
features.append('changed_lines') # drop below 0.01
features.append('is_merge') # slightly improved precission
features.append('avg_removed_count') # drop below 0.01
features.append('avg_mod_count') #sligthly improved recall
features.append('avg_added_count') # drop below 0.01
features.append('avg_file_size') # drop by around 0.01++
features.append('dmm_unit_complexity') # drop by around 0.01++
features.append('dmm_unit_interfacing') # drop by around 0.02++
features.append('dmm_unit_size') # drop by around 0.01
features.append('avg_changed_methods') # drop below 0.001, recall dropped by 0.02
features.append('avg_complexity') # drop below 0.01
features.append('avg_nloc') # drop by around 0.02++
features.append('avg_tokens') # drop by around 0.01
features.append('contains_cwe_title') # drop below 0.01
features.append('contains_security_info') # drop by around 0.03
features.append('contains_security_in_message') # drop below 0.01

x2_train = train[features]
x2_test = test[features]
y2_train = train['is_vulnerability_fix']
y2_test = test['is_vulnerability_fix']


model = RandomForestClassifier(random_state=SEED)
X_train, X_test = x2_train, x2_test
y_train, y_test = y2_train, y2_test


#Random Search
param_dist = {
    'n_estimators': list(range(50, 300, 10)),
    'min_samples_leaf': list(range(1, 50)),
    'max_depth': list(range(2, 20)),
    'max_features': ['sqrt', 'log2'],
    'criterion': ['gini', 'entropy', 'log_loss'],
    'min_impurity_decrease': np.arange(0, 1, 0.1),
    'bootstrap': [True, False]}




n_iter = 20
model_random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_dist,
    n_iter=n_iter,
    scoring = 'f1',
    refit=True,
    cv=5,
    verbose=VERBOSE_LVL)

result = model_random_search.fit(X_train, y_train)
print(result)
print(result.best_estimator_.get_params())





# cv_inner = KFold(n_splits=3, shuffle=True, random_state=SEED)
# space = dict()
# space['n_estimators'] = [50, 100, 200]
# space['max_features'] = [2, 4, 6]
# space['criterion'] = ['gini', 'entropy', 'log_loss']
# print('start learning')
# search = GridSearchCV(model, space, scoring='f1', cv=cv_inner, refit=True, verbose=VERBOSE_LVL)
# result = search.fit(X_train, y_train)
# best_model = result.best_estimator_
# print(best_model.get_params())


# yhat = best_model.predict(X_test)
# # evaluate the model
# acc = accuracy_score(y_test, yhat)
# # store the result
# outer_results.append(acc)
# # report progress








# {'bootstrap': True,
#  'ccp_alpha': 0.0,
#  'class_weight': None,
#  'criterion': 'gini',
#  'max_depth': None,
#  'max_features': 6,
#  'max_leaf_nodes': None,
#  'max_samples': None,
#  'min_impurity_decrease': 0.0,
#  'min_samples_leaf': 1,
#  'min_samples_split': 2,
#  'min_weight_fraction_leaf': 0.0,
#  'n_estimators': 200,
#  'n_jobs': None,
#  'oob_score': False,
#  'random_state': 1,
#  'verbose': 0,
#  'warm_start': False}