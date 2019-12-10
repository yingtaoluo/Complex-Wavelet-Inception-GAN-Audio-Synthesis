import torch
import torch.nn as nn
from complexLayers import ComplexConvTranspose2d, SpectralNormComplexLinear, ComplexLinear
from complexLayers import ComplexBatchNorm2d, SpectralNormComplexConv2d
from complexFunctions import complex_relu, complex_leakyrelu
from complexFunctions import complex_tanh, complex_dropout
import math


class ComplexGenerator(nn.Module):
    def __init__(self):
        super(ComplexGenerator, self).__init__()
        # torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=1,
        # padding=0, output_padding=0, groups=1, bias=True, dilation=1)

        # input(N, C_in, H_in, W_in), output(N, C_out, H_out, W_out)
        # H_out=(H_in−1)×stride[0]−2×padding[0] + kernel_size[0] + output_padding[0]

        self.dense = ComplexLinear(100, 1024*4*4)
        self.deconv1 = ComplexConvTranspose2d(1024, 512, 4, 2, 1)  # (1024, 4, 4) --> (512, 8, 8)
        self.batchnorm1 = ComplexBatchNorm2d(512)
        self.deconv2 = ComplexConvTranspose2d(512, 256, 4, 2, 1)  # (512, 8, 8) --> (256, 16, 16)
        self.batchnorm2 = ComplexBatchNorm2d(256)
        self.deconv3 = ComplexConvTranspose2d(256, 128, 4, 2, 1)  # (256, 16, 16) --> (128, 32, 32)
        self.batchnorm3 = ComplexBatchNorm2d(128)
        self.deconv4 = ComplexConvTranspose2d(128, 64, 4, 2, 1)  # (128, 32, 32) --> (64, 64, 64)
        self.batchnorm4 = ComplexBatchNorm2d(64)
        self.deconv5 = ComplexConvTranspose2d(64, 1, 4, 2, 1)  # (64, 64, 64) --> (1, 128, 128)

    def forward(self, xr, xi):
        # inputs shaped (batch_size, 100)
        xr, xi = self.dense(xr, xi)
        xr, xi = complex_relu(xr, xi)
        xr = xr.view(xr.size(0), 1024, 4, 4)
        xi = xi.view(xi.size(0), 1024, 4, 4)
        xr, xi = complex_relu(xr, xi)  # (batch_size, 1024, 4, 4)
        xr, xi = self.deconv1(xr, xi)  # (batch_size, 512, 8, 8)
        xr, xi = self.batchnorm1(xr, xi)
        xr, xi = complex_relu(xr, xi)
        xr, xi = self.deconv2(xr, xi)  # (batch_size, 256, 16, 16)
        xr, xi = self.batchnorm2(xr, xi)
        xr, xi = complex_relu(xr, xi)
        xr, xi = self.deconv3(xr, xi)  # (batch_size, 128 ,32, 32)
        xr, xi = self.batchnorm3(xr, xi)
        xr, xi = complex_relu(xr, xi)
        xr, xi = self.deconv4(xr, xi)  # (batch_size, 64, 64, 64)
        xr, xi = self.batchnorm4(xr, xi)
        xr, xi = complex_relu(xr, xi)
        xr, xi = self.deconv5(xr, xi)  # (batch_size, 1, 128, 128)
        xr, xi = complex_tanh(xr, xi)
        return xr / math.sqrt(2), xi / math.sqrt(2)  # (batch_size, 1, 128, 128)


class ComplexDiscriminator(nn.Module):
    def __init__(self):
        super(ComplexDiscriminator, self).__init__()
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1,
        # padding=0, dilation=1, groups=1, bias=True)

        # input(N, C_in, H_in, W_in), output(N, C_out, H_out, W_out)
        # H_out=[H_in + 2×padding[0] - dilation[0]×(kernel_size[0]−1) − 1]/stride[0] + 1

        self.conv1 = SpectralNormComplexConv2d(1, 64, 4, 2, 1)
        self.conv2 = SpectralNormComplexConv2d(64, 128, 4, 2, 1)
        self.conv3 = SpectralNormComplexConv2d(128, 256, 4, 2, 1)
        self.conv4 = SpectralNormComplexConv2d(256, 512, 4, 2, 1)
        self.conv5 = SpectralNormComplexConv2d(512, 1024, 4, 2, 1)
        self.dense = SpectralNormComplexLinear(1024*4*4, 1)

    def forward(self, xr, xi):
        # input shaped(batch_size, 1. 128, 128)
        batch = xr.size()[0]
        xr, xi = self.conv1(xr, xi)  # (batch_size, 64, 64, 64)
        xr, xi = complex_leakyrelu(xr, xi, 0.2)
        xr, xi = self.conv2(xr, xi)  # (batch_size, 128, 32, 32)
        xr, xi = complex_leakyrelu(xr, xi, 0.2)
        xr, xi = self.conv3(xr, xi)  # (batch_size, 256, 16, 16)
        xr, xi = complex_leakyrelu(xr, xi, 0.2)
        xr, xi = self.conv4(xr, xi)  # (batch_size, 512, 8, 8)
        xr, xi = complex_leakyrelu(xr, xi, 0.2)
        xr, xi = self.conv5(xr, xi)  # (batch_size, 1024, 4, 4)
        xr, xi = complex_leakyrelu(xr, xi, 0.2)
        xr, xi = xr.view(batch, -1), xi.view(batch, -1)
        xr, xi = self.dense(xr, xi)  # (batch_size, 1)
        x = torch.sqrt(torch.pow(xr, 2) + torch.pow(xi, 2))
        return x.view(batch)  # (batch_size)




