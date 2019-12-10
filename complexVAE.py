import torch
import torch.nn as nn
from complexLayers import ComplexLinear
from complexFunctions import complex_relu, complex_leakyrelu
from complexFunctions import complex_tanh
from torch.autograd import Variable

latent_dim = 100


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.layer0 = ComplexLinear(16384, 4000)
        self.layer1 = ComplexLinear(4000, 2000)
        self.layer2 = ComplexLinear(2000, 1000)
        self.layer3 = ComplexLinear(1000, 500)
        self.layer4 = ComplexLinear(500, 200)
        self.activation = complex_relu

    def forward(self, xr, xi):
        xr, xi = xr.view(-1, 16384), xi.view(-1, 16384)
        xr, xi = self.layer0(xr, xi)
        xr, xi = self.activation(xr, xi)
        xr, xi = self.layer1(xr, xi)
        xr, xi = self.activation(xr, xi)
        xr, xi = self.layer2(xr, xi)
        xr, xi = self.activation(xr, xi)
        xr, xi = self.layer3(xr, xi)
        xr, xi = self.activation(xr, xi)
        xr, xi = self.layer4(xr, xi)
        return xr, xi


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.layer0 = ComplexLinear(latent_dim, 1000)
        self.layer1 = ComplexLinear(1000, 2000)
        self.layer2 = ComplexLinear(2000, 4000)
        self.layer3 = ComplexLinear(4000, 8000)
        self.layer4 = ComplexLinear(8000, 16384)
        self.activation = complex_leakyrelu

    def forward(self, xr, xi):
        xr, xi = self.layer0(xr, xi)
        xr, xi = self.activation(xr, xi, 0.1)
        xr, xi = self.layer1(xr, xi)
        xr, xi = self.activation(xr, xi, 0.1)
        xr, xi = self.layer2(xr, xi)
        xr, xi = self.activation(xr, xi, 0.1)
        xr, xi = self.layer3(xr, xi)
        xr, xi = self.activation(xr, xi, 0.1)
        xr, xi = self.layer4(xr, xi)
        xr, xi = xr.view(-1, 1, 128, 128), xi.view(-1, 1, 128, 128)
        return xr, xi


class VAE(nn.Module):
    def __init__(self, encoder, decoder):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self._enc_mu = ComplexLinear(200, latent_dim)
        self._enc_log_sigma = ComplexLinear(200, latent_dim)

    def re_param(self, h_enc_r, h_enc_i):
        """
        Return the latent normal sample z ~ N(mu, sigma^2)
        """
        mu_r, mu_i = self._enc_mu(h_enc_r, h_enc_i)
        log_sigma_r, log_sigma_i = self._enc_log_sigma(h_enc_r, h_enc_i)
        sigma_r, sigma_i = log_sigma_r.exp_(), log_sigma_i.exp_()
        return mu_r, mu_i, sigma_r, sigma_i

    def sample_z(self, mu, sigma):
        std = Variable(torch.randn(*mu.size()), requires_grad=False).cuda(0)
        z = mu + sigma * std
        return z

    def latent_loss(self, z_mean, z_std):
        mean_sq = z_mean * z_mean
        std_sq = z_std * z_std
        return 0.5 * torch.mean(mean_sq + std_sq - torch.log(std_sq) - 1)

    def forward(self, xr, xi):
        latent_r, latent_i = self.encoder(xr, xi)
        mu_r, mu_i, sigma_r, sigma_i = self.re_param(latent_r, latent_i)
        zr = self.sample_z(mu_r, sigma_r)
        zi = self.sample_z(mu_i, sigma_i)
        xr, xi = self.decoder(zr, zi)
        llr = self.latent_loss(mu_r, sigma_r)
        lli = self.latent_loss(mu_i, sigma_i)
        ll = llr + lli
        return xr, xi, ll
