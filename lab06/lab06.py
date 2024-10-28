import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA



np.random.seed(100)
rng = np.random.RandomState(1)
dane = np.dot(rng.rand(2, 2), rng.randn(2, 200)).T # transponowana
#generuje macierz 2,2

plt.scatter(dane[:, 0], dane[:, 1])
plt.axis('equal')
plt.title('wykres 1b')
plt.show()

def draw_vector(v0, v1, ax=None):
    ax = ax or plt.gca()
    arrowprops=dict(arrowstyle='->',
                    linewidth=2,
                    shrinkA=0, shrinkB=0)
    ax.annotate('', v1, v0, arrowprops=arrowprops)


def wiPCA(org_data, k=1):
    # Obliczanie średniej dla każdej kolumny
    mean = np.mean(org_data, axis=0)
    # Standaryzacja danych (odejmowanie średniej)
    mean_data = org_data - mean
    # Obliczanie macierzy kowariancji
    cov = np.cov(mean_data.T)
    # Obliczanie wartości własnych i wektorów własnych
    eig_val, eig_vec = np.linalg.eig(cov)
    indices = np.arange(0, len(eig_val), 1)
    # Sortowanie wartości własnych i wektorów własnych
    sorted_indices = np.argsort(indices)[::-1]
    eig_val = eig_val[indices]
    eig_vec = eig_vec[:, indices]
    # Wybieranie k największych wartości własnych i odpowiadających im wektorów własnych
    principal_components = eig_vec[:, :k]
    # Obliczanie zredukowanych danych
    pca_data = np.dot(mean_data, eig_vec)
    return pca_data, principal_components, eig_val

# Wywołanie funkcji PCA transformed shape: (200, 2)
pca = PCA(n_components=2)
pca.fit(dane)
# plot data
plt.scatter(dane[:, 0], dane[:, 1], alpha=0.2)
for length, vector in zip(pca.explained_variance_, pca.components_):
    v = vector * 3 * np.sqrt(length)
    draw_vector(pca.mean_, pca.mean_ + v)

# Wywołanie funkcji PCA transformed shape: (200, 1)
pca = PCA(n_components=1)
pca.fit(dane)
dane_pca = pca.transform(dane)
dane_new = pca.inverse_transform(dane_pca)
#plt.scatter(dane[:, 0], dane[:, 1], alpha=0.2) juz u gory wyswietlam
plt.scatter(dane_new[:, 0], dane_new[:, 1], alpha=0.8)


print("original shape:   ", dane.shape)
print("transformed shape:", dane_pca.shape)

# Wykres wizualizacja przestrzeni cech i wektorów własnych
plt.axis('equal')
plt.title('1c: Wizualizacja przestrzeni cech i wektorów własnych')
plt.show()

#zad 2

iris = datasets.load_iris()
X = iris.data
y = iris.target

X_pca, pc, eig_vals = wiPCA(X, k=2)

#2b Wykres redukcji wymiarowości obiektów na zbiore iris
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis')
plt.title('Wizualizacja danych z bazy iris')
plt.show()

#2c ---

#3a
digits = datasets.load_digits()
X = digits.data
y = digits.target
#3b
X_pca, pc, eig_vals = wiPCA(X, k=2)

plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis')
plt.show()

#3c
pca = PCA().fit(X)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.show()

#3d
plt.scatter(X_pca[:, 0], X_pca[:, 1],cmap='viridis', c=y )
plt.colorbar()
plt.show()

#3e ----
