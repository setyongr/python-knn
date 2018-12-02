"""
KNN - K Nearest Neighbor
Author: Setyo Nugroho

Requirement:
    - Python 3.6
    - Pandas
    - Numpy
"""

from collections import Counter
from math import sqrt

import pandas as pd
import numpy as np


def eculiden_distance(data1, data2):
    """
    Get euclidean distance of 2 array
    :param data1: 1D ndarray
    :param data2: 1D ndarray
    :return: eauclidean distance
    """
    if len(data1) != len(data2):
        raise Exception("Data length should same")

    power_sum = 0
    for i in range(len(data1)):
        power_sum += (data1[i] - data2[i]) ** 2

    return sqrt(power_sum)


def knn(k, train, test):
    """
    Do KNN
    :param k: K used
    :param train: 2D array of X1, X2, X3, X4, X5, Y
    :param test: 2D array of X1, X2, X3, X4, X5, Y
    :return: list of prediction result
    """
    result = []
    for ts in test:
        distance_list = []
        for tr in train:
            # Calculate euclidean without Y column
            distance = eculiden_distance(ts[:5], tr[:5])
            # Add distance list with value (Y, distance)
            distance_list.append((tr[5], distance))

        # Sort by distance
        distance_list.sort(key=lambda x: x[1])
        distance_list = distance_list[:k]

        # Counter for finding most common Y in distance list
        counter = Counter()
        for d in distance_list:
            counter[d[0]] += 1
        result.append(int(counter.most_common(1)[0][0]))

    return result


def get_accuracy(k, train, test):
    """
    Run KNN and calculate the accuracy. Test data Y column should has a value
    :param k: The magic K number
    :param train: 2D array of X1, X2, X3, X4, X5, Y
    :param test: 2D array of X1, X2, X3, X4, X5, Y
    :return: Accuracy
    """
    result = knn(k, train, test)
    correct = 0
    for i in range(len(test)):
        if test[i][5] == result[i]:
            correct += 1

    return correct / len(result) * 100


def cross_validation(k_fold, k, data):
    """
    Do cross validation and return the average accuracy
    :param k_fold: The fold
    :param k: The magic K
    :param data: 2D array of X1, X2, X3, X4, X5, Y
    :return: Average accuracy
    """
    fold = np.split(data, k_fold)
    accuracies = np.array([])
    for i in range(k_fold):
        test = fold[i]
        train = fold[:]  # Copy fold list
        del train[i]  # Remove current fold from train
        train = np.concatenate(train)  # Concat train array
        accuracy = get_accuracy(k, train, test)
        accuracies = np.append(accuracies, accuracy)

    return accuracies.mean()


df_train = pd.read_csv('DataTrain_Tugas3_AI.csv')
df_test = pd.read_csv('DataTest_Tugas3_AI.csv')

print("Finding The Best K")
current_k, current_accuracy = 0, 0

# Find best K in range 1 to 50
for k in range(1, 51):
    # Cross validation using 10 fold
    accuracy = cross_validation(10, k, df_train.iloc[:, 1:].values)
    if accuracy > current_accuracy:
        current_k = k
        current_accuracy = accuracy
    print("K = ", k, "Accuracy = ", accuracy)

print("=====================")
print("Best K")
print("K = ", current_k, "Accuracy = ", current_accuracy)

print("=====================")
print("Do Prediction....")

prediction_result = knn(current_k, df_train.iloc[:, 1:].values, df_test.iloc[:, 1:].values)
df_pred = pd.DataFrame(prediction_result)
df_pred.to_csv('TebakanTugas3.csv', index=False, header=False)
print("Prediction Saved...")
