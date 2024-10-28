from IPython.display import Audio
import IPython
import sounddevice as sd
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
import librosa

file = r'C:\Users\kf53844\Downloads\jestemStudentemZut.wav'


IPython.display.Audio(file)

s, fs = sf.read(file, dtype='float32')

s = s[:, 0]

#sd.play(s, fs)

status = sd.wait()

time = np.linspace(0, len(s) / fs * 1000, len(s))


#NORMALIZACJA



max_value = np.max(s)
min_value = np.min(s)
scaled_s = s/max_value
scaled_s = scaled_s * -1


plt.plot(time, scaled_s)
plt.xlabel('czas [ms]')
plt.ylabel('amp')
plt.title('1.3')
plt.grid()
plt.show()


#2.1

def calculate_energy(frame):
    return np.sum(np.power(frame, 2))

def calculate_zero_crossings(frame):
    return np.sum(np.abs(np.diff(np.sign(frame)))) / len(frame)



def obliczWartosci(dlugoscOkna,fs,string):
    energia = []
    funkcjaPrzejsciaPrzezZero = []

    frame_length_samples = int(fs * dlugoscOkna / 1000)
    num_frames = len(s) // frame_length_samples
    frames = np.array_split(s[:num_frames * frame_length_samples], num_frames)

    energia = [calculate_energy(frame) for frame in frames]
    funkcjaPrzejsciaPrzezZero = [calculate_zero_crossings(frame) for frame in frames]

    max_energy = np.max(energia)
    normalized_energy = energia / max_energy
    normalized_zero_crossings = funkcjaPrzejsciaPrzezZero / np.max(funkcjaPrzejsciaPrzezZero)

    # Wykreślenie sygnału dźwiękowego
    plt.plot(time, scaled_s)

    # Wykreślenie znormalizowanej funkcji energii (czerwona linia) i znormalizowanej funkcji przejść przez zero (niebieska linia)
    plt.plot(np.arange(num_frames) * frame_length_samples / fs * 1000, normalized_energy, 'r')
    plt.plot(np.arange(num_frames) * frame_length_samples / fs * 1000, normalized_zero_crossings, 'b')

    plt.xlabel('czas [ms]')
    plt.ylabel('amp')
    plt.title(string)
    plt.show()



energia = []
funkcjaPrzejsciaPrzezZero = []

frame_length_samples = int(fs * 10 / 1000)
num_frames = len(s) // frame_length_samples
frames = np.array_split(s[:num_frames * frame_length_samples], num_frames)

energia = [calculate_energy(frame) for frame in frames]
funkcjaPrzejsciaPrzezZero = [calculate_zero_crossings(frame) for frame in frames]


#------------------------------------------------------------------
#--------------------------------2.2--------------------------------
#------------------------------------------------------------------
# Normalizacja funkcji energii i liczby przejść przez zero
max_energy = np.max(energia)
normalized_energy = energia / max_energy
normalized_zero_crossings = funkcjaPrzejsciaPrzezZero / np.max(funkcjaPrzejsciaPrzezZero)

# Wykreślenie sygnału dźwiękowego
plt.plot(time, scaled_s)

# Wykreślenie znormalizowanej funkcji energii (czerwona linia) i znormalizowanej funkcji przejść przez zero (niebieska linia)
plt.plot(np.arange(num_frames) * frame_length_samples / fs * 1000, normalized_energy, 'r')
plt.plot(np.arange(num_frames) * frame_length_samples / fs * 1000, normalized_zero_crossings, 'b')

plt.xlabel('czas [ms]')
plt.ylabel('amp')
plt.title('2.2 Funkcje E i Z w czasie')

plt.show()

#------------------------------------------------------------------
#--------------------------------2.3--------------------------------
#------------------------------------------------------------------

#------------------------------------------------------------------
#--------------------------------2.4--------------------------------
#------------------------------------------------------------------
obliczWartosci(5,fs,'2.4 5')
obliczWartosci(20,fs,'2.4 20')
obliczWartosci(50,fs,'2.4 50')

#------------------------------------------------------------------
#--------------------------------2.5--------------------------------
#------------------------------------------------------------------

#---------------------------------------------------------------------
#--------------------------------3.1------------------------------------
#---------------------------------------------------------------------


# Wyodrębnienie fragmentu samogłoski
fragment_samogloski = s[6000:6000 + 2048]



from scipy.fft import fft
#okno maskowania  = wektor samogloski * hamming(liczba probek)

#---------------------------------------------------------------------
#--------------------------------3.2------------------------------------
#---------------------------------------------------------------------


maska = np.hamming(len(fragment_samogloski))
maska = maska * fragment_samogloski

time = len(s) / fs * 1000

x = np.linspace(0, time, len(fragment_samogloski))

N = len(maska)

T = 1.0 / fs

plt.plot(x, maska)
plt.xlabel('Time ');
plt.show()

#---------------------------------------------------------------------
#--------------------------------3.3------------------------------------
#---------------------------------------------------------------------


widmo = fft(maska)
logWidmo = np.log(np.abs(widmo))


#---------------------------------------------------------------------
#--------------------------------3.4------------------------------------
#---------------------------------------------------------------------


czestotliwosci = np.linspace(0, fs, len(logWidmo))

plt.plot(czestotliwosci, logWidmo)
plt.xlim(0, 10000)
plt.xlabel('Freq ')
plt.title('Logarytmiczne widmo amplitudowe');
plt.show()

#---------------------------------------------------------------------
#--------------------------------3.5------------------------------------
#---------------------------------------------------------------------


F0= czestotliwosci[np.argmax(logWidmo)]
print(F0)

# F0 = 211.04054714215926


#---------------------------------------------------------------------
#--------------------------------3.6------------------------------------
#---------------------------------------------------------------------


#---------------------------------------------------------------------
#--------------------------------4.1------------------------------------
#---------------------------------------------------------------------

okno = fragment_samogloski


#---------------------------------------------------------------------
#--------------------------------4.2-----------------------------------
#---------------------------------------------------------------------


p = 20
#a = librosa.lpc(okno, p)
#Nie dziala mi :(

#---------------------------------------------------------------------
#--------------------------------4.3-----------------------------------
#---------------------------------------------------------------------
#dowiedziec sie
#---------------------------------------------------------------------
#--------------------------------4.4-----------------------------------
#---------------------------------------------------------------------



#---------------------------------------------------------------------
#--------------------------------4.5-----------------------------------
#---------------------------------------------------------------------
