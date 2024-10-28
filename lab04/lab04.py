#img tutorial
import matplotlib.pyplot as plt
import numpy as np
import pylab as py
from skimage import data
from skimage import filters
from skimage import exposure
from PIL import Image
import matplotlib.image

#zad 2.1 i 2.2



def sinus(f,fs):
    t = np.arange(0,1,1/fs)
    s = np.sin(2*np.pi*f*t)
    return t,s

# czas,sygnal = sinus(10,20)
# plt.plot(czas,sygnal)
# plt.show()
#
# czas,sygnal = sinus(10,21)
# plt.plot(czas,sygnal)
# plt.show()
#
# czas,sygnal = sinus(10,30)
# plt.plot(czas,sygnal)
# plt.show()
#
#
# czas,sygnal = sinus(10,45)
# plt.plot(czas,sygnal)
# plt.show()
#
# czas,sygnal = sinus(10,50)
# plt.plot(czas,sygnal)
# plt.show()
#
# czas,sygnal = sinus(10,100)
# plt.plot(czas,sygnal)
# plt.show()
#
# czas,sygnal = sinus(10,150)
# plt.plot(czas,sygnal)
# plt.show()
#
# czas,sygnal = sinus(10,200)
# plt.plot(czas,sygnal)
# plt.show()
#
# czas,sygnal = sinus(10,250)
# plt.plot(czas,sygnal)
# plt.show()
#
# czas,sygnal = sinus(10,1000)
# plt.plot(czas,sygnal)
# plt.show()

#zad 2.4
#Twierdzenie o próbkowaniu Nyquista-Shannona,
# czy też Kotielnikowa mówi, że do prawidłowego odtworzenia sygnału analogowego
# na podstawie ciągu utworzonego z jego próbek konieczne jest zapewnienie warunku,
# aby częstotliwość próbkowania była co najmniej dwukrotnie wyższa
# od najwyższej składowej widma sygnału analogowego

#zad 2.5
#Zjawisko, które występuje w wyniku błędnie dobranej częstotliwości
# próbkowania i prowadzi do błędnej interpretacji sygnału, nazywa się aliasingiem



img = np.asarray(Image.open(r'C:\Users\kf53844\Downloads\robal.png'))

print(repr(img))



#zad 3.2
print("Liczba wymiarów obrazka:", img.ndim)

#zad 3.3
print("Liczba wartości w pojedynczym pikselu:", img.shape[2]) # wysokosc, szerokosc, rgb
#zad 3.4
r = img[:, :, 0]
# plt.imshow(r, cmap='gray')
# plt.show()
#
luminacja = 0.21 * r + 0.72 * img[:,:,1] + 0.07 * img[:,:,2]
# plt.imshow(luminacja, cmap='gray')
# plt.show()
#
usrednianieWartosciPiksela = (img[:,:,0] + img[:,:,1] + img[:,:,2]) / 3
# plt.imshow(usrednianieWartosciPiksela, cmap='gray')
# plt.show()
#
#
wyznaczenieJasnosciPiksela = (np.maximum(img[:,:,0], np.maximum(img[:,:,1], img[:,:,2]))
                               + np.minimum(img[:,:,0], np.minimum(img[:,:,1], img[:,:,2]))) / 2
# plt.imshow(wyznaczenieJasnosciPiksela, cmap='gray')
# plt.show()

#zad 3.5
lumHis, bins = np.histogram(luminacja, bins=256, range=(0, 256))
plt.plot(bins[:-1], lumHis, color='gray')
plt.show()

wartoscPikselaHis, bins = np.histogram(usrednianieWartosciPiksela, bins=256, range=(0, 256))
plt.plot(bins[:-1], wartoscPikselaHis, color='gray')
plt.title('3.5')
plt.show()

jasnoscPikselaHis, bins = np.histogram(wyznaczenieJasnosciPiksela, bins=256, range=(0, 256))
plt.plot(bins[:-1], jasnoscPikselaHis, color='gray')
plt.show()
#zad 3.6
lumHis, bins = np.histogram(luminacja, bins=16, range=(0, 256))
plt.plot(bins[:-1], lumHis, color='gray')
plt.title('3.6')
plt.show()

#4.1
img = np.asarray(Image.open(r'C:\Users\kf53844\Downloads\zdj.jpg'))

# Wyodrębnienie składowej czerwonej
r = img[:, :, 0]

luminacja = 0.21 * r + 0.72 * img[:,:,1] + 0.07 * img[:,:,2]

hist, bins = np.histogram(luminacja.flatten(), bins=256, range=(0, 256))

plt.plot(hist, color='black')
plt.title('4.2 Histogram')
plt.show()


#4.3
val = filters.threshold_otsu(luminacja)

plt.plot(hist, color='black')
plt.axvline(val, color='r', linestyle='--')
plt.title('METODA Otsu')
plt.show()

#4.4
binaryzowany_obraz = luminacja > val
plt.imshow(binaryzowany_obraz, cmap='gray')
plt.show()

#5.5
#X


