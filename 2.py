import glob
import os
from train import *
from wavegan import *
from loader import *

samples = decode_audio("./data/sc09/train/Eight_00b01445_nohash_0.wav")
plot_wave(samples, 'raw wave')

# samples = np.load("./data/sc09/train_wave_data.npy")[0]

window_size = 256
log_eps = 1e-10
window = signal.get_window('hann', window_size)
frequencies, times, spectrogram = signal.stft(samples, 16000, window=window, nperseg=window_size,
                                                  noverlap=window_size // 2, padded=True)

frequencies = frequencies[:-1]
times = times[:-1]
spectrogram = spectrogram[:-1, :-1]
#  spectrogram shaped (128, 128)

real = np.log(spectrogram.real + 10 + log_eps)
imag = np.log(spectrogram.imag + 10 + log_eps)
plot_spec(times, frequencies, real, 'real')
plot_spec(times, frequencies, imag, 'imag')

amplitude = np.abs(spectrogram)
amplitude = np.log(amplitude + log_eps)
plot_spec(times, frequencies, amplitude, 'log amplitude')

angle = np.arctan2(spectrogram.imag, (spectrogram.real + log_eps))
angle /= np.pi
plot_spec(times, frequencies, angle, 'phases')

os.system("pause")

amplitude = np.exp(amplitude) - log_eps
angle *= np.pi
reconstructed_spectrogram = amplitude * np.exp(angle * 1j)
reconstructed_spectrogram = np.pad(reconstructed_spectrogram, ((0,1),(0,1)), 'constant')

time, audio = signal.istft(reconstructed_spectrogram, fs=16000, window=window, noverlap=window_size//2,
                           nperseg=window_size, nfft=window_size)
print(len(audio))
plot_wave(audio, 'reconstructed wave')
wavwrite('1.wav', 16000, audio)

