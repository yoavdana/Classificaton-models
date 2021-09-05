from model import *
from time import time
import matplotlib.pyplot as plt


import tensorflow as tf
'''
classification of two Mnist data set numbers. compare between different models.
'''
#load mnist data
mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()

train_images = np.logical_or((y_train == 0), (y_train == 1))
test_images = np.logical_or((y_test == 0), (y_test == 1))
x_train, y_train = x_train[train_images], y_train[train_images]
x_test, y_test = x_test[test_images], y_test[test_images]


def  rearrange_data(X):
    return np.reshape(X,(-1,784))

def draw_mnist(m, X, y):
    '''
    draw random images from mnist data set
    :param m:
    :param X:
    :param y:
    :return:
    '''
    samples_idx = np.random.randint(y.size, size=m)
    y_chosen = y[samples_idx]
    while np.all(y_chosen == 0) or np.all(y_chosen == 1):
        samples_idx = np.random.randint(y.size, size=m)
        y_chosen = y[samples_idx]
    return rearrange_data(X[samples_idx, :, :]), y[samples_idx]

#comparison between different models
num_iters = 50

m_s = [50, 100, 300, 500]
s = len(m_s)
mean_accuracy_logistic = [0] * s
mean_accuracy_svm = [0] * s
mean_accuracy_tree = [0] * s
mean_accuracy_knn = [0] * s

time_train = np.zeros((4, s))
time_test = np.zeros((4, s))

test_X = rearrange_data(x_test)
for j in range(num_iters):
    for i, m in enumerate(m_s):
        svm=SVM()
        logistic=Logistic()
        k_nearest=KNearestNeighbors()
        decision=DecisionTree()

        X,y=draw_mnist(m,x_train, y_train)
        t = time()
        svm.fit(X,y)
        time_train[0, i] += time() - t
        t = time()
        logistic.fit(X, y)
        time_train[1, i] += time() - t
        t=time()
        k_nearest.fit(X,y)
        time_train[2, i] += time() - t
        t=time()
        decision.fit(X,y)
        time_train[3, i] += time() - t

        t = time()
        mean_accuracy_logistic[i] += logistic.score(test_X, y_test)['accuracy'] \
                                                               / num_iters
        time_test[0, i] += time() - t
        t = time()
        mean_accuracy_svm[i] += svm.score(test_X, y_test)['accuracy'] / num_iters
        time_test[1, i] += time() - t
        t = time()
        mean_accuracy_tree[i] += decision.score(test_X, y_test)['accuracy'] / \
                                 num_iters
        time_test[2, i] += time() - t
        t = time()
        mean_accuracy_knn[i] += k_nearest.score(test_X, y_test)['accuracy'] / num_iters
        time_test[3, i] += time() - t

plt.plot(m_s, mean_accuracy_svm, label='SVM')
plt.plot(m_s, mean_accuracy_logistic, label='Logistic Regression')
plt.plot(m_s, mean_accuracy_tree, label='Decision Tree')
plt.plot(m_s, mean_accuracy_knn, label='k-Nearest Neighbors')
plt.legend()
plt.xlabel('sample size')
plt.ylabel('accuracy')

avg_time_train = time_train / num_iters
avg_time_test = time_test / num_iters
print('Runtime of logistic:\ttrain=%f sec\ttest=%f sec' % (
avg_time_train[1, -1], avg_time_test[0, -1]))
print(
    'Runtime of SVM:\t\ttrain=%f sec\ttest=%f sec' % (avg_time_train[0, -1], avg_time_test[1, -1]))
print('Runtime of dec-tree:\ttrain=%f sec\ttest=%f sec' % (
avg_time_train[2, -1], avg_time_test[2, -1]))
print('Runtime of k-NN:\ttrain=%f sec\ttest=%f sec' % (avg_time_train[3, -1], avg_time_test[3, -1]))

plt.show()
