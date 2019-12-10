# Complex-Wavelet-Inception-GAN-Audio-Synthesis
This is a github project of an unpublished manuscript improving the quality of unsupervised audio synthesis.
## Introduction
The adversarial audio synthesis first started from the work of Chris Donahua et al.(https://github.com/chrisdonahue/wavegan). Following studies used various strategies to further improve the fidelity of generated audio, such as Phase Gradient Heap Integration introduced by Andr´es Maraﬁoti et al.(https://tifgan.github.io/#S-E), IF-Phase learning proposed by Jesse Engel et al.(https://github.com/tensorflow/magenta/tree/master/magenta/models/gansynth), inverse transform of Melspectrogram to raw wave audio (https://github.com/descriptinc/melgan-neurips), etc. 
We used different original approaches including complex-valued convolution, wavelet transform, and inception generator to improve.
## 1. Complex-valued convolution for complex spectrogram
The learning representation of complex spectrogram can avoid scattering the information of phase. The complex spectrogram, however, can hardly be processed by real-valued neural network. Introducing the architecture of DEEP COMPLEX NETWORKS (Chiheb Trabelsi et al.), we learned complex spectrogram directly.
## 2. Wavelet transform to replace short time Fourier Transform
The short time Fourier Transform is not flexible enough to filter audio features at different frequency scales. This is not a significant improvement in experiments, since the major problems are the quality of generation and phase recovery.
## 3. Inception generator to capture features at different frequency scales
Inspired by wavelet transform and inception network, we designed a generator architecture that provides convolution filters of flexible sizes to capture audio features at different frequency scales. Audio signal is a mixture of various foundamental waves and harmonic waves. Different foundamental waves are at different frequency ranges, and their harmonic waves are at their multiple frequencies. All these together are too complicated and twisted for a single-sized convolution to process. Therefore, we proposed to use multi-sized filters to generate waveform audio signal, just like an inversed inception network. This modification can boost the quality of generated audio drastically, following the direction of WaveGAN, without generating the intermediate spectrogram.
## Author Information
Yingtao Luo, Huazhong University of Science and Technology
