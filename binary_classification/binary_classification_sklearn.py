import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

import bcolors

data = pd.read_csv('digit_en_learn.csv', sep=',')
dataWithHeader = pd.read_csv('digit_en_learn.csv', sep=',')

should_scale_data = True
use_prediction_file = False
should_plot = True

print(dataWithHeader.describe())

print('Data shape: ', data.shape)
y = data.iloc[:, data.shape[1] - 1]
X = data.iloc[:, :data.shape[1] - 1]

X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.2)

print('=======================')

if should_plot:
    plt.plot(X_train)
    plt.show()

# preprocessing data
if should_scale_data:
    scaled_data_X_train = preprocessing.StandardScaler().fit_transform(X_train)
    scaled_data_X_test = preprocessing.StandardScaler().fit_transform(X_test)
else:
    scaled_data_X_train = X_train
    scaled_data_X_test = X_test

if should_plot:
    plt.plot(scaled_data_X_train)
    plt.show()

# Random Forest method
RF = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
RF.fit(scaled_data_X_train, y_train)
RF.predict(scaled_data_X_test)
print('=======================')

scores_rf = cross_val_score(RF, X_test, y_test, cv=10)

print('Random Forest classifier score: ', scores_rf)
# print('Random Forest classifier score: ', round(RF.score(scaled_data_X_test, y_test), 4) * 100, '%')
print('=======================')

# preparing prediction data
if use_prediction_file:
    dataset = pd.read_csv('digit_en_pred2.csv', sep=',')
    pred_data = np.array(dataset)
else:
    pred_data = [[52, 0.0, 0.3333333333, -8.0, 47.82608696, 0, 0, 1, 0, 3, 0, 0.8888888889, 1],
                 [2, 0.8571428571, 0.1428571429, -2, 97.05882353, 0, 0, 0, 0, 7, 0, 0.4446290144, 0],
                 [65, 0.0243902439, 0.3170731707, -28.66666667, 52.50965251, 0, 0, 0, 0, 41, 0, 0.8849026281, 1],
                 [81, 0.02489353418, 0.5, -6, 90.90909091, 0, 0, 0, 0, 2, 0, 0.934149793, 1],
                 [80, 0.08208499862, 0.25, -10, 77.77777778, 0, 0, 0, 0, 4, 0, 0.9003546517, 1],
                 [28, 0, 0.5, -6, 83.33333333, 0, 0, 0, 0, 2, 0, 0.8410852713, 0],
                 [100, 0, 1, -2, 0, 0, 0, 0, 0, 1, 0, 1, 1],
                 [95, 0.1839397206, 0.5, -6, 37.5, 0, 0, 0, 0, 2, 0, 0.8996357672, 1],
                 [33, 0, 0.5, -6, 83.33333333, 0, 0, 0, 0, 2, 0, 0.8507751938, 1],
                 [74, 0, 0.1428571429, -16, 54.23728814, 0, 0, 0, 0, 7, 0, 0.9163898117, 0]
                 ]

# data scaling
if should_scale_data:
    pred_data = np.array(pred_data)
    temp = np.zeros(shape=(pred_data.shape[0], pred_data.shape[1] - 1))
    temp[:, :] = pred_data[:, :-1]

    temp = preprocessing.StandardScaler().fit_transform(temp)

    temp2 = np.zeros(shape=(pred_data.shape[0], pred_data.shape[1]))
    temp2[:, :-1] = temp[:, :]
    temp2 = np.array(temp2)
    temp2[:, -1] = pred_data[:, -1]

    pred_data = temp2
# end of data scaling

# predicting data
i = 1
for p in pred_data:
    data_arr = np.array(p)
    print('Prediction ', i, ' Expected ', data_arr[-1], ': ', RF.predict([data_arr[:-1]]))
    i += 1

print('=======================')
print('=======================')

# Neural Network
NN = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1, max_iter=1000)
NN.fit(scaled_data_X_train, y_train)
NN.predict(X_test)

scores_nn = cross_val_score(NN, X_test, y_test, cv=10)

print('=======================')
# print('NN classifier score: ', round(NN.score(X_test, y_test), 4) * 100, '%')
print('NN classifier score: ', scores_nn)
print('=======================')

i = 1
for p in pred_data:
    data_arr = np.array(p)
    print('Prediction ', i, ' Expected ', data_arr[-1], ': ', NN.predict([data_arr[:-1]]))
    i += 1

print('=======================')

kn = KNeighborsClassifier(n_neighbors=21)
kn.fit(scaled_data_X_train, y_train)
kn.predict(X_test)

scores_kn = cross_val_score(kn, X_test, y_test, cv=10)

print('=======================')
print('KN classifier score: ', scores_kn)
print('=======================')

i = 1
for p in pred_data:
    data_arr = np.array(p)
    print('Prediction ', i, ' Expected ', data_arr[-1], ': ', kn.predict([data_arr[:-1]]))
    i += 1


print('=======================')

unseen_data = pd.read_csv('ml_review_digit2.csv', sep=',')
unseenX = unseen_data.iloc[:, :]

# preprocessing data
if should_scale_data:
    scaled_data_unseen = preprocessing.StandardScaler().fit_transform(unseenX)
else:
    scaled_data_unseen = unseenX

# Unseen data
# Neural Network
unseen_predicted = NN.predict(scaled_data_unseen)

unseen_predicted_df = np.array(unseen_predicted)

i = 1
result_one = 0
result_zero = 0
for p in unseen_predicted_df:
    col = bcolors.bcolors.OKCYAN
    if p == 1:
        col = bcolors.bcolors.WARNING
        result_one += 1
    else:
        result_zero += 1
    print(col, i, '.Result:', p, bcolors.bcolors.ENDC)
    i += 1

unseen_predicted_df = pd.DataFrame(unseen_predicted)
print(unseen_predicted_df.describe())
print('====================')
print(f'number of zeros: {result_zero}, number of ones: {result_one}')
unseen_predicted_df.to_csv("ml_results_digit_en.csv", header=False, index=False)
