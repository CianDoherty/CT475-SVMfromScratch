import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import cvxopt
from cvxopt import matrix, solvers
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
import itertools

class SVM(BaseEstimator, ClassifierMixin):
    @staticmethod
    def gaussian_kernel(x, y):
        # Formula for the kernel: 
        #   K(x, x') = exp( -\sqrt{\frac{|| x - x' ||^2}{2\sigma^2}})
        return np.exp(-np.sqrt(np.linalg.norm(x - y) ** 2 / (2 * 0.5 ** 2)))
    
    def __init__(self, kernel = None):
        
        if not kernel:
            kernel = SVM.gaussian_kernel
            
        self.kernel = kernel
        self.b = 0
        
    def fit(self, X, y):
        #labels must be converted to binary in this case 1 and -1
        y = np.array([1 if bool(n) else -1 for n in y])

        sample_count, feature_count = X.shape
        
        self._label_encoder = LabelEncoder()
        s = self._label_encoder.fit_transform(y)

        K = np.zeros((sample_count, sample_count))
        
        self.coef_ = K
        for i in range(sample_count**2):
            j = i % sample_count
            k = int(np.floor(i / sample_count))

            K[j, k] = self.kernel(X[j], X[k])

        # calculating CVXOPT
        P = cvxopt.matrix(np.outer(y, y) * K)
        q = cvxopt.matrix(np.ones(sample_count) * -1)

        G = cvxopt.matrix(np.diag(np.ones(sample_count) * -1))
        h = cvxopt.matrix(np.zeros(sample_count)) 

        A = cvxopt.matrix(y, (1, sample_count), "d")
        b = cvxopt.matrix(0.0)

        # solve QP problem
        solved = cvxopt.solvers.qp(P, q, G, h, A, b)

        # Grab the multipliers
        multipliers = np.ravel(solved["x"])

        # Now select the indices of the support vectors that are over the threshold
        selv = multipliers > 1e-5
        index = np.arange(len(multipliers))[selv]
      
        # Grab the support vectors
        self.s_v = X[selv]
        self.s_v_mult = multipliers[selv]
        self.s_v_labels = y[selv]


        # calculating the intercept b
        b = 0
        for i in range(len(self.s_v_mult)):
            b += self.s_v_labels[i]
            b -= np.sum(self.s_v_mult * \
                self.s_v_labels * \
                K[index[i], selv]) 

        # Normalize and save b
        self.b = b / len(self.s_v_mult)

        alphas = np.array(solved['x'])
        return alphas

    def predict(self, X):
        decision = np.asarray(np.dot(X, self.b))
        pred = decision.argmax(axis=1)
        return self._label_encoder.inverse_transform(pred)

    def plot(self, X, y, grid_size):
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, grid_size),
                         np.linspace(y_min, y_max, grid_size),
                         indexing='ij')
        flatten = lambda m: np.array(m).reshape(-1,)

        plt.contourf(xx, yy, 
                    cmap=cm.Paired,
                    levels=[-0.001, 0.001],
                    extend='both',
                    alpha=0.8)
        plt.scatter(flatten(X[:, 0]), flatten(X[:, 1]), 
                    c=flatten(y), cmap=cm.Paired)
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.show()


import pandas

data = pandas.read_csv("owls.csv")


selectData = data[data['species'].str.contains("BarnOwl") == False]

X = selectData[["body-length", "wing-length", "body-width", "wing-width"]].values
Y = selectData["species"].values

y = []
for i in Y:
    if i == "LongEaredOwl":
        y.append(0)
    elif i == "SnowyOwl":
        y.append(1)
    else: y.append(2)

svm = SVM()

predict = svm.fit(X,y)


svm.predict(X)


grid_size = 40
svm.plot(X, y, grid_size)






