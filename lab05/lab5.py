import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import scipy
import matplotlib.pyplot as plt
import random
from collections import defaultdict

from scipy.stats import gaussian_kde


def freq(x, prob=True):
    if isinstance(x, pd.Series): # sprawdza czy pd.Series
        x = x.tolist()

    frequencies = {}
    unique_values = list(set(x))
    freqList = []
    countList = []
    totalCount = len(x)

    for val in unique_values:
        if prob:
            freq = x.count(val) / totalCount
        else:
            freq = x.count(val)
        freqList.append(freq)
        countList.append(x.count(val))

    frequencies = {'unique_values': unique_values, 'freqList': freqList, 'countList': countList}
    return frequencies

x = [1,2,2,3,3,3,4,4,4,4]

result = freq(x, prob=True)
print("Unikalne wartosci xi")
print(result['unique_values'])
print("Prawdopodobienstwa xi")
print(result['freqList'])
print("Czestosci xi")
print(result['countList'])

x=[10,20,30,40,50,10,20,30,35]
y=['a','b','b','c','d','a','b','e','e']


def freq2(x, y, prob=False):
    unique_pairs = list(zip(x, y))
    unique_pairs2 = set(zip(x, y))
    total_pairs = len(unique_pairs)

    if prob:
        frequencies = {}
        for pair in unique_pairs:
            if pair in frequencies:
                frequencies[pair] += 1 / total_pairs
            else:
                frequencies[pair] = 1 / total_pairs
        return frequencies, unique_pairs2
    else:
        frequencies = {}
        for pair in unique_pairs:
            if pair in frequencies:
                frequencies[pair] += 1
            else:
                frequencies[pair] = 1
        return frequencies, unique_pairs2

x = [10, 20, 30, 40, 50, 10, 20, 30, 35]
y = ['a', 'b', 'b', 'c', 'd', 'a', 'b', 'e', 'e']

result = freq2(x, y, prob=True)
print("Rozkład:")
print(result[0])
unique_x = [pair[0] for pair in result[1]]
unique_y = [pair[1] for pair in result[1]]

print("Unikalne wartości x:", unique_x)
print("Unikalne wartości y:", unique_y)

x =  [1,5,7,5,2]


def entropy(x):
    total = len(x)
    result = freq(x, prob=True)
    freqList = result['freqList']

    entropy_value = 0
    for prob in freqList:
        entropy_value += -prob * np.log2(prob)
    return entropy_value

print(entropy(x))

y = ['a','a','a','b','b']
def infogain(x, y):
    entropy_y = entropy(y)

    hYX = 0
    result, unique_pairs = freq2(x, y, prob=False)
    total_y = len(y)

    for pair, count in result.items():
        val, _ = pair
        listaY = []

        for i in range(total_y):
            if x[i] == val:
                listaY.append(y[i])

        entropyListaY = entropy(listaY)
        hYX += count / total_y * entropyListaY

    # Oblicz infogain jako entropią Y  -  entropią warunkową H(Y|X)
    info_gain = entropy_y - hYX
    return info_gain


# Przykładowe dane
x = [1, 5, 7, 5, 2]
y = ['a', 'a', 'a', 'b', 'b']

# Obliczenia
info_gain_xy = infogain(x, y)

# Wynik
print("Przyrost informacji (Info Gain) między X a Y:", info_gain_xy)


data = pd.read_csv('C:\\Users\\kf53844\\Downloads\\zoo.csv')

info_gains = {}
target = 'type'
for col in data.columns:
    if col != target:
        info_gains[col] = infogain(data[col], data[target])


results = pd.DataFrame({
    'kolumna': [col for col in data.columns if col != target],
    'entropy': [entropy(data[col]) for col in data.columns if col != target],
    'infogain': [info_gains[col] for col in info_gains.keys()]
})

results_sorted = results[results['kolumna'] != 'animal'].sort_values(by='infogain', ascending=False)

print(results_sorted)

