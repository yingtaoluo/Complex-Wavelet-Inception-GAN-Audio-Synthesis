import torch
from torch import nn, optim
import torch.utils.data as data
from torch.autograd import Variable, grad
import torch.nn.functional as F
from scipy.io.wavfile import write as wavwrite
from scipy.stats import entropy
from scipy import linalg
from complexLoader import spec_to_wav, plot_spec, plot_wave
import numpy as np
import os
import time


'''cuda option'''
torch.manual_seed(1)


class AudioDataSet(data.Dataset):
    def __init__(self, train_real, train_imag):
        self.train_real = train_real
        self.train_imag = train_imag

    def __getitem__(self, index):
        real_tensor = self.train_real[index]
        imag_tensor = self.train_imag[index]
        return real_tensor, imag_tensor

    def __len__(self):
        return self.train_real.shape[0]


def to_numpy(x):
    return x.cpu().data.numpy()


class Trainer:
    def __init__(self, sample_rate, model, classifier, batch_size, gpu, it):
        self.fs = sample_rate
        self.model = model.cuda(gpu)
        self.classifier = classifier.cuda(gpu)
        self.loss = {'m': [], 'latent': []}
        self.lamb = 10
        self.batch_size = batch_size
        self.gpu = gpu
        self.init_iteration = it
        self.lr = 1e-4  # learning_rate
        self.iterations = 10000

        # real data loader iterator
        train_numpy = np.load("./complex_train_data.npy")
        train_numpy = np.expand_dims(train_numpy, axis=1)  # 17920, 1, 128, 128
        train_real_tensor = torch.FloatTensor(train_numpy.real)
        train_imag_tensor = torch.FloatTensor(train_numpy.imag)
        self.train_set = AudioDataSet(train_real_tensor, train_imag_tensor)
        self.loader = data.DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=0)
        self.data_iter = iter(self.loader)

        # test data
        test_numpy = np.load('./complex_test_data.npy')
        np.random.shuffle(test_numpy)
        self.test_r = torch.FloatTensor(test_numpy.real[0:self.batch_size]).cuda(self.gpu)
        self.test_i = torch.FloatTensor(test_numpy.imag[0:self.batch_size]).cuda(self.gpu)

        # fixed noise data
        self.noise_r = torch.randn(self.batch_size, 100).cuda(self.gpu)
        self.noise_i = torch.randn(self.batch_size, 100).cuda(self.gpu)

        # optimization strategies
        self.opt = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=0)

    def generate_audio(self, fake_r, fake_i, order):
        audio_dir = './audio/' + str(order) + '/'
        if not os.path.isdir(audio_dir):
            os.makedirs(audio_dir)
        for i in range(self.batch_size):
            audio_fp = os.path.join(audio_dir, '{}.wav'.format(str(i)))
            fake = fake_r[i, 0] + 1j * fake_i[i, 0]
            audio = spec_to_wav(fake)
            wavwrite(audio_fp, self.fs, audio)
        print("Done generating audio :)")

    def preview(self):
        fake_r, fake_i = self.model(self.noise_r, self.noise_i)
        fake_r, fake_i = to_numpy(fake_r), to_numpy(fake_i)
        self.generate_audio(fake_r, fake_i, 'preview')

    def inception_score(self, num_batch=10, splits=1):
        def softmax_predict(x_r, x_i):
            x_r, x_i = self.model.decoder(x_r, x_i)
            x_r = x_r.view(self.batch_size, 1, 128, 128)
            x_i = x_i.view(self.batch_size, 1, 128, 128)
            features, x = self.classifier(x_r, x_i)
            return F.softmax(x).data.cpu().numpy()

        data_len = self.batch_size*num_batch
        noise_r = torch.randn(data_len, 100).cuda(self.gpu)
        noise_i = torch.randn(data_len, 100).cuda(self.gpu)
        predicts = np.zeros((data_len, 10))

        r, i = self.model.decoder(noise_r[0:1], noise_i[0:1])
        plot_spec(to_numpy(r.view(128, 128)), 'real')
        plot_spec(to_numpy(i.view(128, 128)), 'image')
        fake = to_numpy(r.view(128, 128)) + 1j * to_numpy(i.view(128, 128))
        audio = spec_to_wav(fake)
        plot_wave(audio, 'generation')

        for i in range(num_batch):
            noise_batch_r = noise_r[i * self.batch_size:(i + 1) * self.batch_size]
            noise_batch_i = noise_i[i * self.batch_size:(i + 1) * self.batch_size]
            predicts[i*self.batch_size:(i+1)*self.batch_size] = softmax_predict(noise_batch_r, noise_batch_i)

        # compute the mean kl-div
        split_scores = []

        for k in range(splits):
            part = predicts[k * (data_len // splits): (k+1) * (data_len // splits), :]
            py = np.mean(part, axis=0)
            scores = []
            for i in range(part.shape[0]):
                pyx = part[i, :]
                scores.append(entropy(pyx, py))
            split_scores.append(np.exp(np.mean(scores)))

        return np.mean(split_scores), np.std(split_scores)

    def calculate_fid(self, num_batch=10):
        def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
            mu1 = np.atleast_1d(mu1)
            mu2 = np.atleast_1d(mu2)

            sigma1 = np.atleast_2d(sigma1)
            sigma2 = np.atleast_2d(sigma2)

            assert mu1.shape == mu2.shape, \
                'Training and test mean vectors have different lengths'
            assert sigma1.shape == sigma2.shape, \
                'Training and test covariances have different dimensions'

            diff = mu1 - mu2

            # Product might be almost singular
            covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
            if not np.isfinite(covmean).all():
                msg = ('fid calculation produces singular product; '
                       'adding %s to diagonal of cov estimates') % eps
                print(msg)
                offset = np.eye(sigma1.shape[0]) * eps
                covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

            # Numerical error might give slight imaginary component
            if np.iscomplexobj(covmean):
                if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                    m = np.max(np.abs(covmean.imag))
                    raise ValueError('Imaginary component {}'.format(m))
                covmean = covmean.real

            tr_covmean = np.trace(covmean)

            return (diff.dot(diff) + np.trace(sigma1) +
                    np.trace(sigma2) - 2 * tr_covmean)

        def calculate_statistics(fake_r, fake_i):
            act_r = np.empty((len(fake_r), 1024))
            act_i = np.empty((len(fake_i), 1024))

            for i in range(len(fake_r) // self.batch_size):
                start = i * self.batch_size
                end = start + self.batch_size
                batch_r = torch.FloatTensor(fake_r[start:end]).cuda(self.gpu)
                batch_i = torch.FloatTensor(fake_i[start:end]).cuda(self.gpu)
                features, x = self.classifier(batch_r, batch_i)
                act_r[start:end] = features[0].cpu().data.numpy().reshape(self.batch_size, -1)
                act_i[start:end] = features[1].cpu().data.numpy().reshape(self.batch_size, -1)

            mu_r, mu_i = np.mean(act_r, axis=0), np.mean(act_i, axis=0)
            sigma_r, sigma_i = np.cov(act_r, rowvar=False), np.cov(act_i, rowvar=False)
            return mu_r, mu_i, sigma_r, sigma_i

        data_len = self.batch_size * num_batch
        noise_r = torch.randn(data_len, 100).cuda(self.gpu)
        noise_i = torch.randn(data_len, 100).cuda(self.gpu)
        fake_spec_r = np.empty((data_len, 1, 128, 128), dtype=np.float32)
        fake_spec_i = np.empty((data_len, 1, 128, 128), dtype=np.float32)

        real_spec = np.load("./complex_train_data.npy")
        real_spec_r = np.expand_dims(real_spec.real, axis=1)  # 17920, 1, 128, 128
        real_spec_i = np.expand_dims(real_spec.imag, axis=1)  # 17920, 1, 128, 128

        for i in range(num_batch):
            noise_batch_r = noise_r[i * self.batch_size:(i + 1) * self.batch_size]
            noise_batch_i = noise_i[i * self.batch_size:(i + 1) * self.batch_size]
            temp = self.model.decoder(noise_batch_r, noise_batch_i)
            fake_spec_r[i * self.batch_size:(i + 1) * self.batch_size],\
                fake_spec_i[i * self.batch_size:(i + 1) * self.batch_size] \
                = to_numpy(temp[0]), to_numpy(temp[1])

        m1_r, m1_i, s1_r, s1_i = calculate_statistics(fake_spec_r, fake_spec_i)
        m2_r, m2_i, s2_r, s2_i = calculate_statistics(real_spec_r, real_spec_i)
        fid_value_r = calculate_frechet_distance(m1_r, s1_r, m2_r, s2_r)
        fid_value_i = calculate_frechet_distance(m1_i, s1_i, m2_i, s2_i)
        fid_value = fid_value_r + fid_value_i

        return fid_value

    def train(self):
        for i in range(self.init_iteration, self.iterations + self.init_iteration):
            # start = time.time()
            def criterion(outputs_r, outputs_i, inputs_r, inputs_i):
                loss_r = torch.pow(outputs_r-inputs_r, 2)/self.batch_size
                loss_i = torch.pow(outputs_i-inputs_i, 2)/self.batch_size
                return torch.sum(loss_r) + torch.sum(loss_i)

            try:
                data_r, data_i = self.data_iter.next()
            except StopIteration:
                data_iter = iter(self.loader)
                data_r, data_i = data_iter.next()
            data_r, data_i = data_r.cuda(self.gpu), data_i.cuda(self.gpu)

            output_r, output_i, latent_loss = self.model(data_r, data_i)
            loss_train = criterion(output_r, output_i, data_r, data_i) + self.lamb * latent_loss
            plot_spec(to_numpy(output_r[0,0])+to_numpy(output_i[0,0]) * 1j, 'spectrogram')
            self.generate_audio(to_numpy(output_r), to_numpy(output_i), 'audio')

            if i % 10 == 0:
                test_out_r, test_out_i, latent_loss = self.model(self.test_r, self.test_i)
                loss_test = criterion(test_out_r, test_out_i, self.test_r, self.test_i) + self.lamb * latent_loss
                print('iteration: {}'.format(i))
                print('train_loss: {:.4}, test_loss: {:.4}'.
                      format(to_numpy(loss_train), to_numpy(loss_test)))
                print('latent_loss: {:.4}'.format(latent_loss))
                print('                        ')

            # end = time.time()

            # save model
            if (i+1) % 1000 == 0:
                torch.save({'iteration': i,
                            'state_dict': self.model.state_dict()},
                           'checkpoint_VAE.pth')

            # evaluate the model
            if (i+1) % 1000 == 0:
                inception_mean, inception_std = self.inception_score()
                print("Inception_score: {}±{}".format(inception_mean, inception_std))
                fid = self.calculate_fid()
                print("FID:{}".format(fid))

            # write audio files
            if (i+1) % 1000 == 0:
                fake_r, fake_i = self.model.decoder(self.noise_r, self.noise_i)
                fake_r, fake_i = to_numpy(fake_r), to_numpy(fake_i)
                self.generate_audio(fake_r, fake_i, i + 1)

            # update parameters
            self.opt.zero_grad()
            loss_train.backward()
            self.opt.step()

        # write losses' changes during training
        with open("./train/VAE_losses.txt", 'w') as f:
            f.write(str(self.loss))


if __name__ == '__main__':
    load = True

    from complexVAE import *

    encoder = Encoder()
    decoder = Decoder()
    m = VAE(encoder, decoder)
    if load:
        checkpoint = torch.load('checkpoint_VAE.pth')
        m.load_state_dict(checkpoint['state_dict'])
        iteration = checkpoint['iteration']
    else:
        iteration = 0

    from ComplexTrain_classifier import complexClassifier
    c = complexClassifier()
    check = torch.load('checkpoint_complexClassifier.pth')
    c.load_state_dict(check['classifier'])
    c.eval()

    trainer = Trainer(sample_rate=16000, model=m, classifier=c,
                      batch_size=32, gpu=0, it=iteration)
    trainer.train()
    # fid = trainer.calculate_fid()
    # print(fid)






