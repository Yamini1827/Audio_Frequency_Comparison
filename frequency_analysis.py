import numpy as np
from scipy.io import wavfile
from numpy.linalg import norm

rate1, audio1 = wavfile.read('audio1.wav')
rate2, audio2 = wavfile.read('audio2.wav')

if len(audio1.shape) > 1:
    audio1 = audio1.mean(axis=1)
if len(audio2.shape) > 1:
    audio2 = audio2.mean(axis=1)

fft_audio1 = np.abs(np.fft.fft(audio1))
fft_audio2 = np.abs(np.fft.fft(audio2))

min_len = min(len(fft_audio1), len(fft_audio2))
fft1 = fft_audio1[:min_len]
fft2 = fft_audio2[:min_len]

similarity = np.dot(fft1, fft2) / (norm(fft1) * norm(fft2))

print("Similarity Score:", similarity)

from numpy.linalg import norm

rate1, audio1 = wavfile.read("audio1.wav")
rate2, audio2 = wavfile.read("audio2.wav")

if len(audio1.shape) > 1:
    audio1 = audio1.mean(axis=1)
if len(audio2.shape) > 1:
    audio2 = audio2.mean(axis=1)

fft_audio1 = np.abs(np.fft.fft(audio1))
fft_audio2 = np.abs(np.fft.fft(audio2))

min_len = min(len(fft_audio1), len(fft_audio2))
fft1 = fft_audio1[:min_len]
fft2 = fft_audio2[:min_len]

similarity = np.dot(fft1, fft2) / (norm(fft1) * norm(fft2))

print("Similarity Score:", similarity)

import matplotlib.pyplot as plt
plt.show()