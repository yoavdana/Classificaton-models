import matplotlib.pyplot as plt
from model import *




def f(X):
    return np.sign(X @ np.array([0.3, -0.5]) + 0.1)


def draw_points(m):
    sample = np.random.randn(m, 2)
    while np.all(f(sample) > 0) or np.all(f(sample) < 0):
        sample = np.random.randn(m, 2)
    return sample, f(sample)

fig, axs = plt.subplots(2, 3, figsize=(25, 12))

for i, m in enumerate([5, 10, 15, 25, 70]):
    X, y = draw_points(m)
    test_X, test_y = draw_points(10000)

    perceptron = Perceptron()
    svm = SVM()

    svm.fit(X, y)
    perceptron.fit(X, y)
    class_1 = y > 0
    class_2 = y < 0
    axs.flatten()[i].scatter(X[class_1, 0], X[class_1, 1])
    axs.flatten()[i].scatter(X[class_2, 0], X[class_2, 1])
    axs.flatten()[i].plot([-3, 3], [-3 * 0.3 / 0.5 + 0.1 / 0.5, 3 * 0.3 / 0.5 + 0.1 / 0.5],
                          label='f')
    axs.flatten()[i].plot([-3, 3], [
        -3 * perceptron.w[0] / -perceptron.w[1] + perceptron.w[2] / -perceptron.w[1],
        3 * perceptron.w[0] / -perceptron.w[1] + perceptron.w[2] / -perceptron.w[1]],
                          label='perceptron')
    axs.flatten()[i].plot([-3, 3], [
        -3 * svm.coef_[0, 0] / -svm.coef_[0, 1] + svm.intercept_ / -svm.coef_[0, 1],
        3 * svm.coef_[0, 0] / -svm.coef_[0, 1] + svm.intercept_ / -svm.coef_[0, 1]], label='SVM')
    #     axs.flatten()[i].set_title('svm=' + str(1 - svm.score(test_X, test_y)) + ' perceptron=' + str(1 - perceptron.score(test_X, test_y)))
    axs.flatten()[i].set_title(f'm={m}')
    axs.flatten()[i].legend()

num_iters = 500

mean_accuracy_perceptron = [0] * 5
mean_accuracy_svm = [0] * 5
mean_accuracy_lda = [0] * 5
sample_sizes = [5, 10, 15, 25, 70]

for j in range(num_iters):
    for i, m in enumerate(sample_sizes):
        train_X, train_y = draw_points(m)
        test_X, test_y = draw_points(10000)

        perceptron = Perceptron()
        svm = SVM()
        lda = LDA()

        svm.fit(train_X, train_y)
        perceptron.fit(train_X, train_y)
        lda.fit(train_X, train_y)

        mean_accuracy_perceptron[i] += perceptron.score(test_X, test_y)['accuracy'] / num_iters
        mean_accuracy_svm[i] += svm.score(test_X, test_y)['accuracy'] / num_iters
        mean_accuracy_lda[i] += lda.score(test_X, test_y)['accuracy'] / num_iters

plt.plot(sample_sizes, mean_accuracy_svm, label='SVM')
plt.plot(sample_sizes, mean_accuracy_perceptron, label='Perceptron')
plt.plot(sample_sizes, mean_accuracy_lda, label='LDA')
plt.legend()
plt.xlabel('sample size')
plt.xlabel('accuracy')
plt.show()
