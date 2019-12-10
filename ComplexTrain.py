import torch
from torch import nn, optim
import torch.utils.data as data
from torch.autograd import Variable, grad
import torch.nn.functional as F
from scipy.io.wavfile import write as wavwrite
from scipy.stats import entropy
from scipy import linalg
from complexLoader import spec_to_wav, plot_spec
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
    def __init__(self, sample_rate, generator, discriminator, classifier,
                 critic, batch_size, gpu, it):
        self.fs = sample_rate
        self.G = generator.cuda(gpu)
        self.D = discriminator.cuda(gpu)
        self.classifier = classifier.cuda(gpu)
        self.losses = {'G': [], 'D': [], 'GP': [], 'grad_norm': [], 'distance': []}
        self.lamb = 10
        self.critic = critic
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

        # fixed noise data
        self.noise_r = np.random.uniform(high=1, low=-1, size=(self.batch_size, 100))
        self.noise_r = torch.FloatTensor(self.noise_r).cuda(self.gpu)
        self.noise_i = np.random.uniform(high=1, low=-1, size=(self.batch_size, 100))
        self.noise_i = torch.FloatTensor(self.noise_i).cuda(self.gpu)

        # optimization strategies
        self.g_opt = optim.Adam(self.G.parameters(), lr=self.lr, betas=(0.5, 0.9))
        self.d_opt = optim.Adam(self.D.parameters(), lr=self.lr*5, betas=(0.5, 0.9))

    def calc_gradient_penalty(self, real_r, real_i, fake_r, fake_i):
        # real part
        alpha_r = torch.rand(self.batch_size, 1, 1, 1)
        alpha_r = alpha_r.expand(real_r.size())
        alpha_r = alpha_r.cuda(self.gpu)

        interpolates_r = alpha_r * real_r + ((1 - alpha_r) * fake_r)
        interpolates_r = interpolates_r.cuda(self.gpu)
        interpolates_r = Variable(interpolates_r, requires_grad=True)

        # image part
        alpha_i = torch.rand(self.batch_size, 1, 1, 1)
        alpha_i = alpha_i.expand(real_i.size())
        alpha_i = alpha_i.cuda(self.gpu)

        interpolates_i = alpha_i * real_i + ((1 - alpha_i) * fake_i)
        interpolates_i = interpolates_i.cuda(self.gpu)
        interpolates_i = Variable(interpolates_i, requires_grad=True)

        # Calculate probability of interpolated examples
        disc_interpolates = self.D(interpolates_r, interpolates_i)

        # Calculate gradients of probabilities w.r.t examples
        gradients = grad(outputs=disc_interpolates, inputs=(interpolates_r, interpolates_i),
                         grad_outputs=torch.ones(disc_interpolates.size()).cuda(self.gpu),
                         create_graph=True, retain_graph=True)[0]

        # Gradients have shape (batch_size, num_channels, img_width, img_height)
        # Flatten gradients to easily take norm per example in batch
        gradients = gradients.view(self.batch_size, -1)
        self.losses['grad_norm'].append(gradients.norm(2, dim=1).mean().item())

        # Derivatives of the gradient close to 0 can cause problems because of
        # the square root, thus manually calculate norm and add epsilon
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
        gradient_penalty = self.lamb * ((gradients_norm - 1) ** 2).mean()

        return gradient_penalty

    def critic_train_iteration(self):
        # acquire next batch_size of real data
        try:
            real_r, real_i = self.data_iter.next()
        except StopIteration:
            data_iter = iter(self.loader)
            real_r, real_i = data_iter.next()

        real_r, real_i = real_r.cuda(self.gpu), real_i.cuda(self.gpu)

        # empty gradients in discriminator
        self.D.zero_grad()

        # load noise data
        noise_r = np.random.uniform(high=1, low=-1, size=(self.batch_size, 100))
        noise_i = np.random.uniform(high=1, low=-1, size=(self.batch_size, 100))
        noise_r = torch.FloatTensor(noise_r).cuda(self.gpu)
        noise_i = torch.FloatTensor(noise_i).cuda(self.gpu)

        # calculate probabilities on real data
        d_out_real = self.D(real_r, real_i)
        d_real = d_out_real.mean()

        # calculate probabilities on fake data
        fake_r, fake_i = self.G(noise_r, noise_i)
        d_out_fake = self.D(fake_r, fake_i)
        d_fake = d_out_fake.mean()

        # get and record gradient penalty
        gradient_penalty = self.calc_gradient_penalty(real_r.data, real_i.data, fake_r.data, fake_i.data)
        self.losses['GP'].append(gradient_penalty.item())

        # get total discriminator loss and wasserstein distance
        d_loss = d_fake - d_real + gradient_penalty

        self.losses['D'].append(d_loss.item())
        w_distance = d_real - d_fake
        self.losses['distance'].append(w_distance.item())

        # optimize
        d_loss.backward()
        self.d_opt.step()

    def generator_train_iteration(self):
        # empty gradients in generator
        self.G.zero_grad()

        # load noise data
        noise_r = np.random.uniform(high=1, low=-1, size=(self.batch_size, 100))
        noise_i = np.random.uniform(high=1, low=-1, size=(self.batch_size, 100))
        noise_r = torch.FloatTensor(noise_r).cuda(self.gpu)
        noise_i = torch.FloatTensor(noise_i).cuda(self.gpu)

        # calculate probabilities on fake data
        fake_r, fake_i = self.G(noise_r, noise_i)
        d_fake = self.D(fake_r, fake_i)
        d_fake = d_fake.mean()

        # get loss, record, and optimize
        g_loss = -d_fake
        self.losses['G'].append(g_loss.item())
        g_loss.backward()
        self.g_opt.step()

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
        fake_r, fake_i = self.G(self.noise_r, self.noise_r)
        fake_r, fake_i = to_numpy(fake_r), to_numpy(fake_i)
        self.generate_audio(fake_r, fake_i, 'preview')

    def inception_score(self, num_batch=100, splits=1):
        def softmax_predict(x_r, x_i):
            x_r, x_i = self.G(x_r, x_i)
            x_r = x_r.view(self.batch_size, 1, 128, 128)
            x_i = x_i.view(self.batch_size, 1, 128, 128)
            features, x = self.classifier(x_r, x_i)
            return F.softmax(x).data.cpu().numpy()

        data_len = self.batch_size*num_batch
        noise_r = np.random.uniform(high=1, low=-1, size=(data_len, 100))
        noise_i = np.random.uniform(high=1, low=-1, size=(data_len, 100))
        predicts = np.zeros((data_len, 10))

        for i in range(num_batch):
            noise_batch_r = noise_r[i * self.batch_size:(i + 1) * self.batch_size]
            noise_batch_i = noise_i[i * self.batch_size:(i + 1) * self.batch_size]
            noise_batch_r = torch.FloatTensor(noise_batch_r).cuda(self.gpu)
            noise_batch_i = torch.FloatTensor(noise_batch_i).cuda(self.gpu)
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

    def calculate_fid(self, num_batch=100):
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

        noise_r = np.random.uniform(high=1, low=-1, size=(data_len, 100))
        noise_i = np.random.uniform(high=1, low=-1, size=(data_len, 100))
        fake_spec_r = np.empty((data_len, 1, 128, 128), dtype=np.float32)
        fake_spec_i = np.empty((data_len, 1, 128, 128), dtype=np.float32)

        real_spec = np.load("./complex_train_data.npy")
        real_spec_r = np.expand_dims(real_spec.real, axis=1)  # 17920, 1, 128, 128
        real_spec_i = np.expand_dims(real_spec.imag, axis=1)  # 17920, 1, 128, 128

        for i in range(num_batch):
            noise_batch_r = noise_r[i * self.batch_size:(i + 1) * self.batch_size]
            noise_batch_i = noise_i[i * self.batch_size:(i + 1) * self.batch_size]
            noise_batch_r = torch.FloatTensor(noise_batch_r).cuda(self.gpu)
            noise_batch_i = torch.FloatTensor(noise_batch_i).cuda(self.gpu)
            temp = self.G(noise_batch_r, noise_batch_i)
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
        # save the noise data for preview
        # np.save('./train/noise_input.npy', self.noise_data)

        for i in range(self.init_iteration, self.iterations + self.init_iteration):
            # start = time.time()

            # update D network
            for p in self.D.parameters():
                p.requires_grad = True
            for d_iteration in range(self.critic):
                self.critic_train_iteration()

            # update G network
            for p in self.D.parameters():
                p.requires_grad = False
            self.generator_train_iteration()

            # end = time.time()
            # print("You need to wait {} minutes".format(int((end-start)*self.iterations/60)))

            # output loss every 10 iterations
            if (i+1) % 10 == 0:
                print("iteration: {}".format(i + 1))
                print("D loss: {}".format(self.losses['D'][-1]))
                print("G loss: {}".format(self.losses['G'][-1]))
                print("GP: {}".format(self.losses['GP'][-1]))
                print("Gradient norm: {}".format(self.losses['grad_norm'][-1]))
                print("Wasserstein distance: {}".format(self.losses['distance'][-1]))

            # save model
            if (i+1) % 100 == 0:
                torch.save({'iteration': i,
                            'G_state_dict': self.G.state_dict(),
                            'D_state_dict': self.D.state_dict()},
                           'checkpoint_complex.pth')

            # evaluate the model
            if (i+1) % 100 == 0:
                inception_mean, inception_std = self.inception_score()
                print("Inception_score: {}±{}".format(inception_mean, inception_std))
                fid = self.calculate_fid()
                print("FID:{}".format(fid))

            # write audio files
            if (i+1) % 1000 == 0:
                fake_r, fake_i = self.G(self.noise_r, self.noise_i)
                # if self.gpu is not None:
                #     fake = fake.cpu().data
                fake_r, fake_i = to_numpy(fake_r), to_numpy(fake_i)
                self.generate_audio(fake_r, fake_i, i + 1)

        # write losses' changes during training
        with open("./train/losses.txt", 'w') as f:
            f.write(str(self.losses))


if __name__ == '__main__':
    load = True

    from complexSpecgan import *
    if load:
        g = ComplexGenerator()
        d = ComplexDiscriminator()
        checkpoint = torch.load('checkpoint_complex.pth')
        g.load_state_dict(checkpoint['G_state_dict'])
        d.load_state_dict(checkpoint['D_state_dict'])
        iteration = checkpoint['iteration']
    else:
        g = ComplexGenerator()
        d = ComplexDiscriminator()
        iteration = 0

    from ComplexTrain_classifier import complexClassifier
    c = complexClassifier()
    check = torch.load('checkpoint_complexClassifier.pth')
    c.load_state_dict(check['classifier'])
    c.eval()

    trainer = Trainer(sample_rate=16000, generator=g, discriminator=d, classifier=c,
                      critic=1, batch_size=32, gpu=0, it=iteration)
    trainer.train()
    # fid = trainer.calculate_fid()
    # print(fid)






