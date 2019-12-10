from scipy.io.wavfile import read as wavread
from scipy import signal
import numpy as np
import glob
import os

window_size = 256
log_eps = 1e-3
window = signal.get_window('hann', window_size)


# Decodes audio file paths into 32-bit waveform floating point vectors.
def decode_audio(fp, fs=16000, normalize=True, fast_wav=True):
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


# from audio files create numpy wave data
def collect_wave_data(dir, fs=16000, norm=True, fast=True, slice_len=16384, batch_size=64):

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


def plot_wave(audio_data, title):
    import matplotlib.pyplot as plt
    x = np.linspace(0, audio_data.shape[0] - 1, audio_data.shape[0])
    plt.plot(x, audio_data)
    plt.title(title, fontsize=15)
    plt.show()
    plt.close()


def plot_spec(spectrogram, title):
    import matplotlib.pyplot as plt
    times = np.linspace(1, 128, 128)
    frequencies = np.linspace(1, 128, 128)
    plt.pcolormesh(times, frequencies, spectrogram.real, cmap='Greys_r')
    # plt.imshow(spectrogram)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.title(title, fontsize=15)
    plt.show()
    plt.close()


def wav_to_spec(audio):
    frequencies, times, spectrogram = signal.stft(audio, 16000, window=window, nperseg=window_size,
                                                  noverlap=window_size // 2, padded=True)
    frequencies = frequencies[:-1]
    times = times[:-1]
    spectrogram = spectrogram[:-1, :-1]
    #  spectrogram shaped (128, 128)

    return spectrogram


def spec_to_wav(spectrogram):
    spectrogram = np.pad(spectrogram, ((0, 1), (0, 1)), 'constant')

    time, audio = signal.istft(spectrogram, fs=16000, window=window, noverlap=window_size // 2,
                               nperseg=window_size, nfft=window_size)
    # print(len(audio))
    # audio shaped (16384,)
    # plot_wave(audio, 'reconstructed wave')
    return audio


# from wave create complex spectrogram
def form_spec_data(fp):
    complex_data = []
    wav_data = np.load(fp)

    for i in range(wav_data.shape[0]):
        spectrogram = wav_to_spec(wav_data[i])
        complex_data.append(spectrogram)

    # enforce float32 to save memory when reading it
    complex_data = np.array(complex_data, dtype=np.complex64)
    print(complex_data.real.dtype)

    return complex_data


if __name__ == '__main__':

    dir = 'E:/audio/data/sc09/test_wave_data.npy'
    test_data = form_spec_data(dir)
    print(test_data.shape)
    np.save("E:/audio/data/sc09/complex_test_data.npy", test_data)
    '''
    audio_dir = 'E:/audio/data/sc09/test'
    test_data = collect_wave_data(audio_dir, fs=16000)
    # train_data = form_spec_data(dir)
    print(test_data.shape)
    np.save("E:/audio/data/sc09/test_wave_data.npy", test_data)
    '''




