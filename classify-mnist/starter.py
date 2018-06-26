import numpy as np
import matplotlib.pyplot as plt

# Load the training dataset
train_features = np.load("train_features.npy")
train_labels = np.load("train_labels.npy").astype("int8")

n_train = train_labels.shape[0]


def visualize_digit(features, label):
    # Digits are stored as a vector of 400 pixel values. Here we
    # reshape it to a 20x20 image so we can display it.
    plt.imshow(features.reshape(20, 20), cmap="binary")
    plt.xlabel("Digit with label " + str(label))
    plt.show()


# Visualize a digit
# visualize_digit(train_features[0, :], train_labels[0])


# TODO: Plot three images with label 0 and three images with label 1
index_label_0 = np.where(train_labels == 0)[0][:3]
index_label_1 = np.where(train_labels == 1)[0][:3]

for i in range(3):
    visualize_digit(train_features[index_label_0[i], :], train_labels[index_label_0])
    visualize_digit(train_features[index_label_1[i], :], train_labels[index_label_1])


# Linear regression
def linear_regression(X, y):
    best_beta = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), y)
    return best_beta


# TODO: Solve the linear regression problem, regressing
train_labels[train_labels == 0] = -1
weight_vector = linear_regression(train_features, train_labels)
X = np.dot(train_features, weight_vector[:, None])

# TODO: Report the residual error and the weight vector
residual_error = np.mean((train_labels - X) ** 2)
print residual_error
print weight_vector

# Load the test dataset
# It is good practice to do this after the training has been
# completed to make sure that no training happens on the test
# set!
test_features = np.load("test_features.npy")
test_labels = np.load("test_labels.npy").astype("int8")

n_test = test_labels.shape[0]

# TODO: Implement the classification rule and evaluate it
# on the training and test set
X_ = np.round(np.dot(train_features, weight_vector[:, None]))
residual_error = np.mean((train_labels - X_) ** 2)
print residual_error

# TODO: Try regressing against a vector with 0 for class 0
# and 1 for class 1
train_labels[train_labels == -1] = 0
print np.dot(test_features[0, :], weight_vector)

# TODO: Form a new feature matrix with a column of ones added
# and do both regressions with that matrix

# Logistic Regression

# You can also compare against how well logistic regression is doing.
# We will learn more about logistic regression later in the course.

import sklearn.linear_model

lr = sklearn.linear_model.LogisticRegression()
lr.fit(X, train_labels)

test_error_lr = 1.0 * sum(lr.predict(test_features) != test_labels) / n_test
