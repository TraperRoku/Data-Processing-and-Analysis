import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import scipy
import matplotlib.pyplot as plt
import random

from scipy.stats import gaussian_kde


#zad 1

df = pd.DataFrame({"x": [1, 2, 3, 4, 5], 'y': ['a', 'b',
                                               'a', 'b', 'b']})
print(df)

print(df.groupby('y').mean())


#zad 2
print(df.value_counts())


#zad 3



data = pd.read_csv('C:\\Users\\kf53844\\PycharmProjects\\zadanie1\\autos.csv')

loadTXT = np.loadtxt('C:\\Users\\kf53844\\PycharmProjects\\zadanie1\\autos.csv', delimiter=',',dtype='str')
print(loadTXT)
#zad 4
fuel = data.groupby('make')[['city-mpg', 'highway-mpg']].mean()

print(fuel)

#zad 5
print()
print()
print()

fuelType = data.groupby('make')['fuel-type'].value_counts()

print(fuelType)


#zad 6

wykres1 = np.polyfit(data['length'],data['city-mpg'],1)
wykres2 = np.polyfit(data['length'],data['city-mpg'],2)
print(wykres1)
print(wykres2)

prognozujący = np.polyval(wykres1, data['length'])
prognozujący2 = np.polyval(wykres2, data['length'])


#zad 7

korelacjaLiniowa, _ = pearsonr(prognozujący, data['city-mpg'])
print("Współczynnik korelacji liniowej dla 1 korelacji liniowej", korelacjaLiniowa)

#zad 8

plt.figure(figsize=(10, 6))

plt.scatter(data['length'],data['city-mpg'], color='blue', label='Dane')
plt.plot(data['length'], prognozujący, color='red', label='Model liniowy 1')

plt.xlabel('lenght')
plt.ylabel('city-mpg')

plt.legend()

plt.show()


#zad 9

daneLength = data['length']
gaussian_kde1 = gaussian_kde(daneLength)

wartosciX = np.linspace(min(daneLength), max(daneLength), 1000)

wartoscFunkcjiGestosci = gaussian_kde1(wartosciX)

plt.figure(figsize=(10, 6))

plt.plot(wartosciX,wartoscFunkcjiGestosci, color='green', label='Estymator Gaussa')
plt.scatter(data['length'],gaussian_kde1(data['length']), color='blue', label='Probki')
plt.legend()
plt.xlabel('length')
plt.ylabel('gestosc')

plt.show()

#zad10
plt.subplot(2,1,1)
plt.plot(wartosciX,wartoscFunkcjiGestosci, color='green')
plt.scatter(data['length'],gaussian_kde1(data['length']), color='blue')



daneWidth = data['width']
gaussian_kde2 = gaussian_kde(daneWidth)
wartosciX2 = np.linspace(min(daneWidth), max(daneWidth), 1000)


wartoscFunkcjiGestosci2 = gaussian_kde2(wartosciX2)

plt.subplot(2,1,2)

plt.plot(wartosciX2, wartoscFunkcjiGestosci2, color='green')
plt.scatter(data['width'], gaussian_kde2(data['width']), color='blue')
plt.show()

#zad 11
length = data['length']
width = data['width']
kde_2d = gaussian_kde([length, width])

x, y = np.meshgrid(np.linspace(min(length), max(length), 100), np.linspace(min(width), max(width), 100))

grid = np.vstack([x.ravel(), y.ravel()])

valuesGestosc = kde_2d(grid).reshape(x.shape)

plt.figure()
plt.contour(x, y, valuesGestosc, cmap='Blues')

plt.scatter(length, width, color='black', s=15)

plt.savefig('plot.png')
plt.savefig('plot.pdf')
plt.show()