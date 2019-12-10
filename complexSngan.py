import torch
import torch.nn as nn
from complexLayers import SpectralNormComplexLinear, ComplexLinear
from complexLayers import ComplexBatchNorm1d, SpectralNormComplexConv2d
from complexFunctions import complex_relu, complex_leakyrelu
from complexFunctions import complex_tanh


class ComplexGenerator(nn.Module):
    def __init__(self):
        super(ComplexGenerator, self).__init__()
        # torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=1,
        # padding=0, output_padding=0, groups=1, bias=True, dilation=1)

        # input(N, C_in, H_in, W_in), output(N, C_out, H_out, W_out)
        # H_out=(H_in−1)×stride[0]−2×padding[0] + kernel_size[0] + output_padding[0]
        self.size = 16384
        self.size1 = int(self.size * 0.125)
        self.size2 = int(self.size * 0.25)
        self.size3 = int(self.size * 0.5)

        self.dense1 = ComplexLinear(100, self.size1)
        self.batchnorm1 = ComplexBatchNorm1d(self.size1)
        self.dense2 = ComplexLinear(self.size1, self.size2)
        self.batchnorm2 = ComplexBatchNorm1d(self.size2)
        self.dense3 = ComplexLinear(self.size2, self.size3)
        self.batchnorm3 = ComplexBatchNorm1d(self.size3)
        self.dense4 = ComplexLinear(self.size3, self.size)

    def forward(self, xr, xi):
        # inputs shaped (batch_size, 100)
        xr, xi = self.dense1(xr, xi)
        xr, xi = self.batchnorm1(xr, xi)
        xr, xi = complex_relu(xr, xi)
        xr, xi = self.dense2(xr, xi)
        xr, xi = self.batchnorm2(xr, xi)
        xr, xi = complex_relu(xr, xi)
        xr, xi = self.dense3(xr, xi)
        xr, xi = self.batchnorm3(xr, xi)
        xr, xi = complex_relu(xr, xi)
        xr, xi = self.dense4(xr, xi)
        xr, xi = complex_tanh(xr, xi)
        xr, xi = xr.view(-1, 1, 128, 128), xi.view(-1, 1, 128, 128)
        return xr, xi


class ComplexDiscriminator(nn.Module):
    def __init__(self):
        super(ComplexDiscriminator, self).__init__()
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1,
        # padding=0, dilation=1, groups=1, bias=True)

        # input(N, C_in, H_in, W_in), output(N, C_out, H_out, W_out)
        # H_out=[H_in + 2×padding[0] - dilation[0]×(kernel_size[0]−1) − 1]/stride[0] + 1
        self.size = 16384
        self.size1 = int(self.size / 4)
        self.size2 = int(self.size / 16)
        self.size3 = int(self.size / 64)
        self.size4 = int(self.size / 256)

        self.dense1 = SpectralNormComplexLinear(self.size, self. size1)
        self.dense2 = SpectralNormComplexLinear(self.size1, self. size2)
        self.dense3 = SpectralNormComplexLinear(self.size2, self. size3)
        self.dense4 = SpectralNormComplexLinear(self.size3, self. size4)
        self.dense5 = SpectralNormComplexLinear(self.size4, 1)

    def forward(self, xr, xi):
        # input shaped(batch_size, 1. 128, 128)
        batch = xr.size()[0]
        xr, xi = xr.view(batch, -1), xi.view(batch, -1)
        xr, xi = self.dense1(xr, xi)
        xr, xi = complex_leakyrelu(xr, xi, 0.2)
        xr, xi = self.dense2(xr, xi)
        xr, xi = complex_leakyrelu(xr, xi, 0.2)
        xr, xi = self.dense3(xr, xi)
        xr, xi = complex_leakyrelu(xr, xi, 0.2)
        xr, xi = self.dense4(xr, xi)
        xr, xi = complex_leakyrelu(xr, xi, 0.2)
        xr, xi = self.dense5(xr, xi)
        x = torch.sqrt(torch.pow(xr, 2) + torch.pow(xi, 2))
        return x.view(batch)  # (batch_size)




