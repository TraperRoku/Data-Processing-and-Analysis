import numpy as np
from sklearn import datasets
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from collections import Counter
from sklearn.neighbors import KNeighborsRegressor

def euclidean_distance(x1, x2):

    distance = np.sqrt(np.sum((x1 - x2) ** 2))
    return distance

class KNN:
    def __init__(self, n_neighbors=1, use_KDTree=False):
        self.n_neighbors = n_neighbors
        self.use_KDTree = use_KDTree

    def euclidean(X_train, X):
        return np.sqrt(np.sum((X_train - X) ** 2, axis=1))

    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        return self

    def predict(self, X):
        y = []
        if not self.use_KDTree:
            for x in X:
                distances = KNN.euclidean(self.X_train, x)
                neighbors_idx = np.argsort(distances)[:self.n_neighbors]
                neighbors_labels = self.y_train[neighbors_idx]
                yi = np.mean(neighbors_labels)
                y.append(yi)
        else:
            tree = KDTree(self.X_train)
            for x in X:
                _, neighbors_idx = tree.query([x], k=self.n_neighbors)
                neighbors_idx = neighbors_idx[0]
                neighbors_labels = self.y_train[neighbors_idx]
                yi = np.mean(neighbors_labels)
                y.append(yi)

        y = np.array(y)
        return y

    def score(self, X, y):
        y_pred = self.predict(X)
        MSE = np.mean(np.square(np.subtract(y, y_pred)))
        return MSE


cmap = ListedColormap(['#FF0000','#00FF00','#0000FF'])



#----------------------------------
#---------------3.1----------------
X, y = make_classification(
    n_samples=100,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    n_clusters_per_class=1,
    random_state=3 )

#----------------------------------
#---------------3.2----------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

plt.figure()
plt.title('3.2')
plt.scatter(X[:,0],X[:,1], c=y, cmap=cmap, edgecolor='k', s=20)
plt.show()


clf = KNN(n_neighbors= 5, use_KDTree=False)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)

print(predictions)

acc = np.sum(predictions == y_test) / len(y_test)
print(acc)
#----------------------------------
#---------------3.3----------------

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))


Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)


plt.figure()
plt.contourf(xx, yy, Z, cmap=cmap, alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap, edgecolor='k', s=20)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title('3.3')
plt.show()


#----------------------------------
#---------------3.4----------------
#----------------------------------
from sklearn.decomposition import PCA

# Load the Iris dataset
iris = datasets.load_iris()
X, y = iris.data, iris.target



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

#----------------------------------
#---------------3.5----------------
#----------------------------------

clf = KNN(n_neighbors=5, use_KDTree=False)
clf.fit(X_train, y_train)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plt.figure()
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap=cmap, edgecolor='k', s=20)
plt.title('PCA 3.4')
plt.show()


x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

xx_pca = pca.inverse_transform(np.c_[xx.ravel(), yy.ravel()])

Z = clf.predict(xx_pca)
Z = Z.reshape(xx.shape)

plt.figure()
plt.contourf(xx, yy, Z, cmap=cmap, alpha=0.8)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap=cmap, edgecolor='k', s=20)
plt.title('PCA meshgrid 3.5')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.show()

# #----------------------------------
# #---------------3.6----------------
# #----------------------------------
#

#----------------------------------
#---------------3.7----------------
#----------------------------------


#----------------------------------
#---------------4.1----------------
#----------------------------------

X, y = (datasets.make_regression
        (n_samples=100, n_features=1, noise=2, random_state=5))


#----------------------------------
#---------------4.2----------------
#----------------------------------
knnRegressor = KNN(n_neighbors=2, use_KDTree=False)
knnRegressor.fit(X, y)
y_pred = knnRegressor.predict(X)


#----------------------------------
#---------------4.3----------------
#----------------------------------
MSE = knnRegressor.score(X, y)
print(f"Mean Squared Error: {MSE:.2f}")

# Plotting
plt.scatter(X, y, label='dane uczace')
plt.scatter(X, y_pred, label='predykcja')
plt.legend()
plt.show()


