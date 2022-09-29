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

dropNaN = True
VERBOSE = False
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# df = pd.read_csv('/Users/awestfalewicz/Projects/2022-internship-andrzej-westfalewicz/mined_features_for_issues_most_starred.csv')
df = pd.read_csv('/Users/awestfalewicz/Projects/2022-internship-andrzej-westfalewicz/mined_features_for_most_common_projects.csv')
if dropNaN:
    df = df.dropna()
fixes_sample = df[df['is_vulnerability_fix']] 
not_fixes = df[df['is_vulnerability_fix'] == False] 
not_fixes_sample = not_fixes.sample(n=fixes_sample.shape[0], random_state=SEED)
data_balanced = pd.concat([fixes_sample, not_fixes_sample])

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
# # New features
# features.append('biggest_file_size')
# features.append('biggest_file_change')


all_Precision= 0.731651376146789
all_Accuracy= 0.7242888402625821
all_Recall  = 0.7026431718061674

print('Original dataset', df.shape)
x_all=df[features]  # Features
y_all=df['is_vulnerability_fix']
x1_train, x1_test, y1_train, y1_test = train_test_split(x_all, y_all, test_size=0.3)

print('Balanced dataset', data_balanced.shape)
x_balanced=data_balanced[features]
y_balanced=data_balanced['is_vulnerability_fix']
x2_train, x2_test, y2_train, y2_test = train_test_split(x_balanced, y_balanced, test_size=0.3)


# corr = x_all.corrwith(y_all)
corr = x_all.corr()
corr = corr.round(decimals = 2)
sn.heatmap(corr, annot=True)
plt.show()

# print('Trying RandomForestClassifier with weights')
# weights = {True: 10, False: 1}
# criterion = 'gini' # 'gini' 'entropy' 'log_loss'
# forest_classifier = RandomForestClassifier(n_estimators=50, criterion=criterion, class_weight=weights, verbose=VERBOSE)
# forest_classifier.fit(x1_train,y1_train)
# y_pred = forest_classifier.predict(x1_test)
# print("Precision:", metrics.precision_score(y1_test, y_pred))
# print("Accuracy :", metrics.accuracy_score(y1_test, y_pred))
# print("Recall   :", metrics.recall_score(y1_test, y_pred))
# # Accuracy: 0.9296751813308105
# # Recall  : 0.22304832713754646

print('Trying RandomForestClassifier balanced')
criterion = 'gini' # 'gini' 'entropy' 'log_loss'
forest_classifier = RandomForestClassifier(n_estimators=150, criterion=criterion, verbose=VERBOSE)
forest_classifier.fit(x2_train,y2_train)
y_pred = forest_classifier.predict(x2_test)
print("Precision:", metrics.precision_score(y2_test, y_pred))
print("Accuracy:", metrics.accuracy_score(y2_test, y_pred))
print("Recall  :", metrics.recall_score(y2_test, y_pred))
# Accuracy: 0.7429718875502008
# Recall  : 0.722007722007722

# print('Trying MLPClassifier')
# scaler = preprocessing.StandardScaler().fit(x2_train) # StandardScaler RobustScaler MaxAbsScaler
# x2_train_scaled = scaler.transform(x2_train)
# act_fnc = 'tanh' # identity, logistic, tanh, relu
# solve_alg = 'adam' # lbfgs, sgd, adam
# perceptron = MLPClassifier(hidden_layer_sizes=(40,40), activation=act_fnc, max_iter=250, n_iter_no_change=5, solver=solve_alg, verbose=VERBOSE)
# perceptron.fit(x2_train_scaled,y2_train)
# x2_test_scaled = scaler.transform(x2_test)
# y_pred=perceptron.predict(x2_test_scaled)
# print("Precision:", metrics.precision_score(y2_test, y_pred))
# print("Accuracy:",metrics.accuracy_score(y2_test, y_pred))
# print("Recall  :",metrics.recall_score(y2_test, y_pred))
# # Accuracy: 0.7289156626506024
# # Recall  : 0.7374517374517374

# print('Trying LogisticRegression weighted')
# weights = {True: 10, False: 1}
# solve_alg = 'liblinear' # 'lbfgs' 'liblinear' 'sag' 'saga'
# logistic_regressor = LogisticRegression(class_weight=weights, solver=solve_alg, verbose=VERBOSE)
# logistic_regressor.fit(x1_train, y1_train)
# y_pred = logistic_regressor.predict(x1_test)
# print("Precision:", metrics.precision_score(y1_test, y_pred))
# print("Accuracy:", metrics.accuracy_score(y1_test, y_pred))
# print("Recall  :", metrics.recall_score(y1_test, y_pred))
# # Accuracy: 0.7410911384421318
# # Recall  : 0.6431226765799256


# print('Trying LogisticRegression balanced')
# solve_alg = 'liblinear' # 'lbfgs' 'liblinear' 'sag' 'saga'
# logistic_regressor = LogisticRegression(solver=solve_alg, verbose=VERBOSE)
# logistic_regressor.fit(x2_train, y2_train)
# y_pred = logistic_regressor.predict(x2_test)
# print("Precision:", metrics.precision_score(y2_test, y_pred))
# print("Accuracy:", metrics.accuracy_score(y2_test, y_pred))
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
# print("Accuracy:", metrics.accuracy_score(y2_test, y_pred))
# print("Recall  :", metrics.recall_score(y2_test, y_pred))


# print('Trying SVC')
# kernel = SVC( verbose=VERBOSE)
# scaler = preprocessing.RobustScaler().fit(x2_train) # StandardScaler RobustScaler MaxAbsScaler
# x2_train_scaled = scaler.transform(x2_train)
# kernel.fit(x2_train_scaled,y2_train)
# x2_test_scaled = scaler.transform(x2_test)
# y_pred = kernel.predict(x2_test_scaled)
# print("Precision:", metrics.precision_score(y2_test, y_pred))
# print("Accuracy:", metrics.accuracy_score(y2_test, y_pred))
# print("Recall  :", metrics.recall_score(y2_test, y_pred))


# # # First need to resolve import error
# # print('Trying Light Labirynth')
# # light = LightLabyrinthClassifier(verbose=VERBOSE)
# # scaler = preprocessing.RobustScaler().fit(x2_train) # StandardScaler RobustScaler MaxAbsScaler
# # x2_train_scaled = scaler.transform(x2_train)
# # light.fit(x2_train_scaled,y2_train)
# # x2_test_scaled = scaler.transform(x2_test)
# # y_pred = light.predict(x2_test_scaled)
# # print("Precision:", metrics.precision_score(y2_test, y_pred))
# # print("Accuracy:", metrics.accuracy_score(y2_test, y_pred))
# # print("Recall  :", metrics.recall_score(y2_test, y_pred))
