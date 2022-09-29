from sklearn.model_selection import RandomizedSearchCV, KFold, GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN

import glob
import pandas as pd 
import numpy as np
import random

dropNaN = True
VERBOSE = True
VERBOSE_LVL = 10
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

path = 'results\checkpoints4'

csv_files = glob.glob(path + "/*.csv")
dfs  = (pd.read_csv(file) for file in csv_files)
df   = pd.concat(dfs, ignore_index=True)
df = df.fillna(0)
all_data = df

features =  ['author_to_commiter_date_diff',
'same_author_as_commiter',
'committed_by_bot',
'authored_by_bot',
'changed_files',
'dmm_unit_complexity',
'dmm_unit_interfacing',
'dmm_unit_size',
'message_contains_cwe_title',
'title_contains_cwe_title',
'message_contains_security_keyword',
'title_contains_security_keyword',
'has_npm',
'has_mvn',
'has_pypi',
'is_code',
'is_add',
'is_rename',
'is_delete',
'is_modify',
'test_in_filename',
'test_in_path',
'has_methods_with_security_keywords',
'change_contains_cwe_title',
'file_contains_cwe_title',
'change_contains_security_keyword',
'file_contains_security_keyword',
'removed_lines_count_avg',
'removed_lines_count_max',
'added_lines_count_avg',
'added_lines_count_max',
'changed_lines_count_avg',
'changed_lines_count_max',
'removed_lines_ratio_avg',
'removed_lines_ratio_max',
'added_lines_ratio_avg',
'added_lines_ratio_max',
'modified_lines_count_avg',
'modified_lines_count_max',
'modified_lines_ratio_avg',
'modified_lines_ratio_max',
'file_size_avg',
'file_size_max',
'changed_methods_count_avg',
'changed_methods_count_max',
'total_methods_count_avg',
'total_methods_count_max',
'file_complexity_avg',
'file_complexity_max',
'file_nloc_avg',
'file_nloc_max',
'file_token_count_avg',
'file_token_count_max',
'file_changed_method_count_avg',
'file_changed_method_count_max',
'max_method_token_count_avg',
'max_method_token_count_max',
'max_method_complexity_avg',
'max_method_complexity_max',
'max_method_nloc_avg',
'max_method_nloc_max',
'max_method_parameter_count_avg',
'max_method_parameter_count_max',
'avg_method_token_count_avg',
'avg_method_token_count_max',
'avg_method_complexity_avg',
'avg_method_complexity_max',
'avg_method_nloc_avg',
'avg_method_nloc_max',
'avg_method_parameter_count_avg',
'avg_method_parameter_count_max']


sampler, X_temp, y_temp = None
train_data = all_data.sample(frac = 0.7)
test_data = all_data.drop(train_data.index)

# scenario 1: No resampleing
# x_train = train_data[features]
# y_train = train_data['is_fix']
# # TRAIN: 5205:50281
# # TEST : 2245:21534
# # Precision: 0.717741935483871
# # F1_score: 0.3106457242582897
# # Recall  : 0.19821826280623608


# scenario 2: Shrink to 50:50
# fixes = train_data[train_data['is_fix']] 
# not_fixes = train_data.drop(fixes.index)
# not_fixes = not_fixes.sample(min(fixes.shape[0], not_fixes.shape[0]))
# new_train = pd.concat([fixes, not_fixes])

# x_train = new_train[features]
# y_train = new_train['is_fix']
# # TRAIN: 5205:5205
# # TEST : 2245:21534
# # Precision: 0.2038785404439908
# # F1_score: 0.31696915600515724
# # Recall  : 0.711804008908686

# scenario 3: RandomOverSampler
# X_temp = train_data[features]
# y_temp = train_data['is_fix']
# sampler = RandomOverSampler(random_state=SEED)
# # TRAIN: 50281:50281
# # TEST : 2245:21534
# # Precision: 0.5809312638580931
# # F1_score: 0.33301557038449314
# # Recall  : 0.2334075723830735

# scenario 4: SMOTE
# X_temp = train_data[features]
# y_temp = train_data['is_fix']
# sampler = SMOTE(random_state=SEED)
# # TRAIN: 50281:50281
# # TEST : 2245:21534
# # Precision: 0.46004319654427644
# # F1_score: 0.2686849574266793
# # Recall  : 0.18975501113585747

if sampler != None:
    x_train, y_train = sampler.fit_resample(X_temp, y_temp)


def show_classes_balance(X_train, X_test, y_train, y_test):
    train_fixes = y_train[y_train==True]
    train_not_fixes = y_train[y_train==False]
    test_fixes = y_test[y_test==True]
    test_not_fixes = y_test[y_test==False]

    print(f'TRAIN: {train_fixes.shape[0]}:{train_not_fixes.shape[0]}')
    print(f'TEST : {test_fixes.shape[0]}:{test_not_fixes.shape[0]}')


X_train, X_test = x_train, test_data[features]
y_train, y_test = y_train, test_data['is_fix']
show_classes_balance(X_train, X_test, y_train, y_test)

def random_search(model):
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
    return result.best_estimator_



def grid_search(model):
    cvfold = KFold(n_splits=5, shuffle=True, random_state=SEED)
    space = dict()
    space['n_estimators'] = [135]
    # space['criterion'] = ['gini', 'entropy'] # gini
    # space['max_depth'] = None
    # space['min_samples_leaf'] = 1
    # space['min_weight_fraction_leaf'] = 0
    space['max_features'] = [15, 'sqrt']
    # space['max_leaf_nodes'] = None
    # space['min_impurity_decrease'] = 0
    # space['bootstrap'] = True

    search = GridSearchCV(model, space, scoring='f1', cv=cvfold, refit=True, verbose=VERBOSE_LVL)
    result = search.fit(X_train, y_train)
    print(result)
    print(result.best_estimator_.get_params())
    return result.best_estimator_

model = RandomForestClassifier(n_estimators=135, max_features=15, random_state=SEED)
# random_search(model)
# model = grid_search(model)
model.fit(x_train, y_train)

y_pred = model.predict(X_test)
print("Precision:", metrics.precision_score(y_test, y_pred))
print("F1_score:",metrics.f1_score(y_test, y_pred))
print("Recall  :",metrics.recall_score(y_test, y_pred))


# # From Random search:
# {'bootstrap': False,
#  'ccp_alpha': 0.0,
#  'class_weight': None,
#  'criterion': 'entropy',
#  'max_depth': 15,
#  'max_features': 'sqrt',
#  'max_leaf_nodes': None,
#  'max_samples': None,
#  'min_impurity_decrease': 0.0,
#  'min_samples_leaf': 36,
#  'min_samples_split': 2,
#  'min_weight_fraction_leaf': 0.0,
#  'n_estimators': 110,
#  'n_jobs': None,
#  'oob_score': False,
#  'random_state': 42,
#  'verbose': 0,
#  'warm_start': False}

# # From Grid search:
# {'bootstrap': True,
#  'ccp_alpha': 0.0,
#  'class_weight': None,
#  'criterion': 'gini',
#  'max_depth': None,
#  'max_features': 'auto',
#  'max_leaf_nodes': None,
#  'max_samples': None,
#  'min_impurity_decrease': 0.0,
#  'min_samples_leaf': 1,
#  'min_samples_split': 2,
#  'min_weight_fraction_leaf': 0.0,
#  'n_estimators': 140,
#  'n_jobs': None,
#  'oob_score': False,
#  'random_state': 42,
#  'verbose': 0,
#  'warm_start': False}

