weimport pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys


def distp(point, centroids):

    return np.sqrt(np.sum((point - centroids) ** 2, axis=1))


def distm(point, data, cov=None):

    y_mu = point - np.mean(data, axis=0)
    if cov is None:
        cov = np.cov(data.T)
    inv_covmat = np.linalg.inv(cov)
    left = np.dot(y_mu, inv_covmat)
    mahal = np.dot(left, y_mu.T)
    return mahal


def ksrodki(X, k, distance='euclidean'):

    centroids = []
    centroids.append(X[np.random.randint(X.shape[0]), :])
    plot(X, np.array(centroids))

    for c_id in range(k - 1):
        dist = []
        for i in range(X.shape[0]):
            point = X[i, :]
            d = sys.maxsize

            for j in range(len(centroids)):
                if distance == 'euclidean':
                    temp_dist = distp(point, centroids[j][np.newaxis, :])
                elif distance == 'mahalanobis':
                    temp_dist = distm(point, X)
                d = min(d, temp_dist)
            dist.append(d)

        dist = np.array(dist)
        next_centroid = X[np.argmax(dist), :]
        centroids.append(next_centroid)
        plot(X, np.array(centroids))

    return np.array(centroids)


def plot(data, centroids):
    plt.scatter(data[:, 0], data[:, 1], marker='.',
                color='gray', label='data points')
    plt.scatter(centroids[:-1, 0], centroids[:-1, 1],
                color='black', label='previously selected centroids')
    plt.scatter(centroids[-1, 0], centroids[-1, 1],
                color='red', label='next centroid')
    plt.title('Wybierz % d centroid' % (centroids.shape[0]))

    plt.legend()
    plt.show()


# Wczytywanie danych
file_path = "I:\\auto.csv"
X = pd.read_csv(file_path)[['wheel-base', 'length']].to_numpy()

k = 4
centroids = ksrodki(X, k)

plot(X,centroids)