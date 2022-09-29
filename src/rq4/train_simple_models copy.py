from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn import metrics
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC
import seaborn as sn
import matplotlib.pyplot as plt

# from light_labyrinth.dim2 import LightLabyrinthClassifier

import pandas as pd 
import numpy as np
import random
import glob

nan_strategy = 'drop' # None zero drop
VERBOSE = False
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
path = 'results\checkpoints4'
# df = pd.read_csv('/Users/awestfalewicz/Projects/2022-internship-andrzej-westfalewicz/mined_features_for_most_common_projects.csv')

csv_files = glob.glob(path + "/*.csv")
dfs  = (pd.read_csv(file) for file in csv_files)
df   = pd.concat(dfs, ignore_index=True)

df = df.dropna()
for x in df:
    for y in df[x]:
        print(y)
# if nan_strategy == 'drop':
#     df = df.dropna()
# elif nan_strategy == 'zero':
#     df = df.fillna(0)

fixes_sample = df[df['is_fix'] == True] 
not_fixes = df[df['is_fix'] == False] 
not_fixes_sample = not_fixes.sample(n=fixes_sample.shape[0], random_state=SEED)
data_balanced = pd.concat([fixes_sample, not_fixes_sample])

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



all_Precision= 0.731651376146789
all_Accuracy= 0.7242888402625821
all_Recall  = 0.7026431718061674

print('Balanced dataset', data_balanced.shape)
x_balanced=data_balanced[features]
y_balanced=data_balanced['is_fix']
x2_train, x2_test, y2_train, y2_test = train_test_split(x_balanced, y_balanced, test_size=0.3)

print('fixes     in train',x2_train.shape[0])
print('fixes     in test ',x2_test.shape[0])
print('non-fixes in train',y2_train.shape[0])
print('non-fixes in test ',y2_test.shape[0])


# corr = x_all.corrwith(y_all)
# corr = x_all.corr()
# corr = corr.round(decimals = 2)
# sn.heatmap(corr, annot=True)
# plt.show()

print('Trying RandomForestClassifier balanced')
criterion = 'gini' # 'gini' 'entropy' 'log_loss'
forest_classifier = RandomForestClassifier(n_estimators=350, criterion=criterion, verbose=VERBOSE)
forest_classifier.fit(x2_train,y2_train)
y_pred = forest_classifier.predict(x2_test)
print("Precision:", metrics.precision_score(y2_test, y_pred))
print("F1_score:", metrics.f1_score(y2_test, y_pred))
print("Recall  :", metrics.recall_score(y2_test, y_pred))
# Accuracy: 0.7429718875502008
# Recall  : 0.722007722007722

# print('Trying MLPClassifier')
# scaler = preprocessing.RobustScaler().fit(x2_train) # StandardScaler RobustScaler MaxAbsScaler
# x2_train_scaled = scaler.transform(x2_train)
# act_fnc = 'tanh' # identity, logistic, tanh, relu
# solve_alg = 'adam' # lbfgs, sgd, adam
# perceptron = MLPClassifier(hidden_layer_sizes=(40,40), activation=act_fnc, max_iter=250, n_iter_no_change=5, solver=solve_alg, verbose=VERBOSE)
# perceptron.fit(x2_train_scaled,y2_train)
# x2_test_scaled = scaler.transform(x2_test)
# y_pred=perceptron.predict(x2_test_scaled)
# print("Precision:", metrics.precision_score(y2_test, y_pred))
# print("F1_score:",metrics.f1_score(y2_test, y_pred))
# print("Recall  :",metrics.recall_score(y2_test, y_pred))
# # Accuracy: 0.7289156626506024
# # Recall  : 0.7374517374517374


# print('Trying LogisticRegression balanced')
# solve_alg = 'liblinear' # 'lbfgs' 'liblinear' 'sag' 'saga'
# logistic_regressor = LogisticRegression(solver=solve_alg, verbose=VERBOSE)
# logistic_regressor.fit(x2_train, y2_train)
# y_pred = logistic_regressor.predict(x2_test)
# print("Precision:", metrics.precision_score(y2_test, y_pred))
# print("F1_score:", metrics.f1_score(y2_test, y_pred))
# print("Recall  :", metrics.recall_score(y2_test, y_pred))
# # Accuracy: 0.6345381526104418
# # Recall  : 0.47876447876447875


# print('Trying SGDClassifier')
# scaler = preprocessing.RobustScaler().fit(x2_train) # StandardScaler RobustScaler MaxAbsScaler
# x2_train_scaled = scaler.transform(x2_train)
# loss = 'hinge'
# sgd = SGDClassifier(loss=loss, verbose=VERBOSE)
# sgd.fit(x2_train_scaled,y2_train)
# x2_test_scaled = scaler.transform(x2_test)
# y_pred = sgd.predict(x2_test_scaled)
# print("Precision:", metrics.precision_score(y2_test, y_pred))
# print("F1_score:", metrics.f1_score(y2_test, y_pred))
# print("Recall  :", metrics.recall_score(y2_test, y_pred))


# print('Trying SVC')
# kernel = SVC( verbose=VERBOSE)
# scaler = preprocessing.RobustScaler().fit(x2_train) # StandardScaler RobustScaler MaxAbsScaler
# x2_train_scaled = scaler.transform(x2_train)
# kernel.fit(x2_train_scaled,y2_train)
# x2_test_scaled = scaler.transform(x2_test)
# y_pred = kernel.predict(x2_test_scaled)
# print("Precision:", metrics.precision_score(y2_test, y_pred))
# print("F1_score:", metrics.f1_score(y2_test, y_pred))
# print("Recall  :", metrics.recall_score(y2_test, y_pred))


# # First need to resolve import error
# print('Trying Light Labirynth')
# light = LightLabyrinthClassifier(height=50, width=50)
# scaler = preprocessing.RobustScaler().fit(x2_train) # StandardScaler RobustScaler MaxAbsScaler
# x2_train_scaled = scaler.transform(x2_train)
# light.fit(x2_train_scaled, y2_train, 10)
# x2_test_scaled = scaler.transform(x2_test)
# y_pred = light.predict(x2_test_scaled)
# print("Precision:", metrics.precision_score(y2_test, y_pred))
# print("F1_score:", metrics.f1_score(y2_test, y_pred))
# print("Recall  :", metrics.recall_score(y2_test, y_pred))
