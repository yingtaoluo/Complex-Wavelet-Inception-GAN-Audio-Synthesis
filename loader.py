from scipy.io.wavfile import read as wavread
from scipy import signal
import numpy as np
import glob
import os

window_size = 256
log_eps = 1e-3
window = signal.get_window('hann', window_size)


def decode_audio(fp, fs=16000, normalize=True, fast_wav=True):
    # Decodes audio file paths into 32-bit waveform floating point vectors.
    if fast_wav:
        # Read with scipy wavread (fast), return with sample rate and data
        _fs, _wav = wavread(fp)
        if fs is not None and fs != _fs:
            raise NotImplementedError('Scipy cannot resample audio to this rate.')
        if _wav.dtype == np.int16:
            _wav = _wav.astype(np.float32)
            _wav /= 32768.
        elif _wav.dtype == np.float32:
            _wav = np.copy(_wav)
        else:
            raise NotImplementedError('Scipy cannot process atypical WAV files.')
    else:
        # Decode with librosa load (slow but supports file formats like mp3).
        import librosa
        _wav, _fs = librosa.core.load(fp, sr=fs, mono=False)
        if _wav.ndim == 2:
            _wav = np.swapaxes(_wav, 0, 1)

    assert _wav.dtype == np.float32

    # At this point, _wav is np.float32 either [nsamps,] or [nsamps, nch].
    if _wav.ndim != 1:
        nch = _wav.shape[1]
    else:
        nch = 1

    # If nch does not match num_channels, average (mono) or expand (stereo) channels
    if nch != 1:
        _wav = np.mean(_wav, 2, keepdims=True)

    if normalize:
        factor = np.max(np.abs(_wav))
        if factor > 0:
            _wav /= factor

    return _wav


def plot_wave(audio_data, title):
    import matplotlib.pyplot as plt
    x = np.linspace(0, audio_data.shape[0] - 1, audio_data.shape[0])
    plt.plot(x, audio_data)
    plt.title(title, fontsize=15)
    plt.show()


def plot_spec(times, frequencies, spectrogram, title):
    import matplotlib.pyplot as plt
    plt.pcolormesh(times, frequencies, spectrogram, cmap='Greys_r')
    # plt.imshow(spectrogram)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.title(title, fontsize=15)
    plt.show()


def wav_to_spec(audio):
    frequencies, times, spectrogram = signal.stft(audio, 16000, window=window, nperseg=window_size,
                                                  noverlap=window_size // 2, padded=True)
    frequencies = frequencies[:-1]
    times = times[:-1]
    spectrogram = spectrogram[:-1, :-1]
    #  spectrogram shaped (128, 128)

    amplitude = np.abs(spectrogram)
    amplitude = np.log(amplitude + log_eps)
    # plot_spec(times, frequencies, amplitude, 'amplitude')

    angle = np.arctan2(spectrogram.imag, (spectrogram.real + log_eps))
    angle /= np.pi
    # plot_spec(times, frequencies, angle, 'phases')
    return amplitude, angle


def spec_to_wav(amplitude, angle):
    amplitude = np.exp(amplitude) - log_eps
    angle *= np.pi
    reconstructed_spectrogram = amplitude * np.exp(angle * 1j)
    reconstructed_spectrogram = np.pad(reconstructed_spectrogram, ((0, 1), (0, 1)), 'constant')

    time, audio = signal.istft(reconstructed_spectrogram, fs=16000, window=window, noverlap=window_size // 2,
                               nperseg=window_size, nfft=window_size)
    print(len(audio))
    # audio shaped (16384,)

    # plot_wave(audio, 'reconstructed wave')
    return audio


# from audio files create numpy wave data
def form_wave_data(dir, fs=16000, norm=True, fast=True, slice_len=16384, batch_size=64):

    print("decoding audio files & forming data set...")

    # extract audio files in the directory fps
    fps = glob.glob(os.path.join(dir, '*'))

    # build the first element for concatenation
    files = np.zeros(slice_len)
    files = np.expand_dims(files, axis=0)

    # concatenate file with length fs
    for i, fp in enumerate(fps):
        file = decode_audio(fp, fs=fs, fast_wav=fast, normalize=norm)
        print("{}/{} audio shape {}".format(i + 1, len(fps), np.shape(file)))

        if file.shape[0] % 2 == 0 and file.shape[0] >= 10000:
            # padding number
            pad = int((slice_len - file.shape[0]) / 2)

            # pad file to length slice_len
            file = np.pad(file, (pad,), 'constant')
            file = np.expand_dims(file, axis=0)

            # concatenation
            files = np.concatenate((files, file))

            print("files shape:{}".format(np.shape(files)))

    # throw away the file element
    files = files[1:]

    # trim the files to be the multiple of batch_size
    length = files.shape[0]
    remainder = length % batch_size
    files = files[remainder:]

    # enforce float32 to save memory when reading it
    files.astype(np.float32)

    return files


# from wave create magnitude & phase data
def form_spec_data(fp):
    amplitudes = []
    angles = []
    wav_data = np.load(fp)

    for i in range(wav_data.shape[0]):
        amplitude, angle = wav_to_spec(wav_data[i])
        amplitudes.append(amplitude)
        angles.append(angle)

    # enforce float32 to save memory when reading it
    amplitudes = np.array(amplitudes, dtype=np.float32)
    angles = np.array(angles, dtype=np.float32)

    return amplitudes, angles


if __name__ == '__main__':

    audio_dir = './data/sc09/train/'
    train_data = form_numpy_data(audio_dir, fs=16000, slice_len=16384, batch_size=64)
    np.save("./data/sc09/train_wave_data.npy", train_data)

    '''
    data_path = "./data/sc09/train_wave_data.npy"
    amp, ang = form_spec_data(data_path)
    np.save("./data/sc09/train_amplitudes.npy", amp)
    np.save("./data/sc09/train_phases.npy", ang)
    '''




