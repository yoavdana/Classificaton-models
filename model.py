import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


def general_score(model, X, y):
    '''
    determine the score of a classifier
    :param model: a given classifier
    :param X: 
    :param y: 
    :return: dict of different scores parameters
    '''

    TP = np.sum(np.logical_and(model.predict(X) == 1, y == 1))
    FN = np.sum(np.logical_and(model.predict(X) == -1, y == 1))
    FP = np.sum(np.logical_and(model.predict(X) == 1, y == -1))
    TN = np.sum(np.logical_and(model.predict(X) == -1, y == -1))
    P = y[y == 1].sum()
    N = np.abs(y[y == -1]).sum()
    acc = (TP + TN) / (P + N)
    FPTN = max(1,FP+TN)
    TPFN = max(1,TP+FN)
    TPFP = max(1,TP+FP)
    out_dict = {'num_samples' : len(y),
               'error' : 1-acc,
               'accuracy' : acc,
               'FPR' : FP / FPTN,
               'TPR' : TP / TPFN,
               'precision' : TP / TPFP,
               'recall' : TP / TPFN}
    return out_dict


class  Perceptron:

    def __init__(self):
        self.w=None


    def fit(self, X, y):
        X = np.hstack((X, np.ones((X.shape[0], 1))))
        w = np.zeros(X.shape[1])
        while np.any(np.sign(X @ w) - y):
            labeled_wrong_index = np.nonzero(np.sign(X @ w) - y)[0][0]
            w += X[labeled_wrong_index] * y[labeled_wrong_index]

        self.w = w

    def predict(self, X):
        X = np.hstack((X, np.ones((X.shape[0], 1))))
        return np.sign(X @ self.w)

    def score(self, X, y):
        return general_score(self, X, y)

class LDA:

    def __init__(self):
        self.pr_y = None
        self.mu_y = None
        self.sigma = None
        self.sigma_inv = None
        self.bias = None

    def fit(self,X,y):
        self.pr_y = np.array([(y == 1).mean(), (y == -1).mean()])
        mu_y0=X[y==1].mean(axis=0)
        mu_y1 = X[y==-1].mean(axis=0)
        self.mu_y=np.array([mu_y0,mu_y1])
        self.sigma = (X[y == 1] - self.mu_y[:, 0]).T @ (
                    X[y == 1] - self.mu_y[:, 0]) + (X[y == -1] - self.mu_y[:, 1]).T @ (X[y == -1] - self.mu_y[:, 1])

        self.sigma=self.sigma/y.size
        self.sigma_inv=np.linalg.inv(self.sigma)
        self.bias = - 0.5 * np.diag(self.mu_y.T @ self.sigma_inv @ self.mu_y) + np.log(self.pr_y)


    def predict(self,X):

        return -2*np.argmax(X @ self.sigma_inv @ self.mu_y + self.bias, axis=1)+1

    def score(self, X, y):
        return general_score(self, X, y)




class SVM(SVC):
    def __init__(self):
        SVC.__init__(self, C=1e10, kernel='linear')


    def score(self, X, y):
        return general_score(self, X, y)

class Logistic(LogisticRegression):
    def __init__(self):
        LogisticRegression.__init__(self, solver='liblinear')

    def score(self, X, y):
        return general_score(self, X, y)

class DecisionTree(DecisionTreeClassifier):
    def __init__(self):
        DecisionTreeClassifier.__init__(self, max_depth=5)

    def score(self, X, y):
        return general_score(self, X, y)


class KNearestNeighbors(KNeighborsClassifier):
    def __init__(self):
        KNeighborsClassifier.__init__(self, n_neighbors=40)

    def score(self, X, y):
        return general_score(self, X, y)

def draw_point(m):
    mu = np.array([0, 0])
    cov = np.eye(2)
    X = np.array(np.random.multivariate_normal(mu, cov, m))
    y = np.sign(np.inner(np.array([0.3, -0.5]), X) + 0.1).astype(int)
    y[y == 0] = 1
    return X, y
