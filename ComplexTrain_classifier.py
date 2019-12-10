import torch
import torch.utils.data as data
from torch import nn, optim
from complexLayers import ComplexAdaptiveAvgPool2d, ComplexConv2d, ComplexLinear
from complexFunctions import complex_leakyrelu
import numpy as np


class LabeledComplexDataSet(data.Dataset):
    def __init__(self, train, gpu):
        # train_data (cut, 16384+1)
        self.len = train.shape[0]
        self.input_length = train.shape[1] - 1
        self.inputs = train[:, 0:self.input_length].reshape(self.len, 1, 128, 128)
        self.labels = train[:, self.input_length:]
        self.gpu = gpu

    def __getitem__(self, index):
        input_real_tensor = torch.FloatTensor(self.inputs[index].real)
        input_imag_tensor = torch.FloatTensor(self.inputs[index].imag)
        label_tensor = torch.FloatTensor(self.labels[index].real).long()
        return input_real_tensor, input_imag_tensor, label_tensor

    def __len__(self):
        return self.len


class complexClassifier(nn.Module):
    def __init__(self):
        super(complexClassifier, self).__init__()
        # torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride=1,
        # padding=0, dilation=1, groups=1, bias=True)

        # input(N, C_in, H_in, W_in), output(N, C_out, H_out, W_out)
        # H_out=[H_in + 2×padding[0] - dilation[0]×(kernel_size[0]−1) − 1]/stride[0] + 1

        self.conv1 = ComplexConv2d(1, 64, 4, 2, 1)
        self.conv2 = ComplexConv2d(64, 128, 4, 2, 1)
        self.conv3 = ComplexConv2d(128, 256, 4, 2, 1)
        self.conv4 = ComplexConv2d(256, 512, 4, 2, 1)
        self.conv5 = ComplexConv2d(512, 1024, 4, 2, 1)
        self.dense = ComplexLinear(1024 * 4 * 4, 10)
        self.pool = ComplexAdaptiveAvgPool2d(output_size=(1, 1))

    def forward(self, xr, xi):
        # inputs shaped (batch_size, 1, 128, 128)
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
        features = self.pool(xr, xi)
        xr, xi = xr.view(batch, -1), xi.view(batch, -1)  # (batch_size, 1024*4*4)
        xr, xi = self.dense(xr, xi)  # (batch_size, 10)
        x = torch.sqrt(torch.pow(xr, 2) + torch.pow(xi, 2))
        return features, x


def load_wave_data(cut):
    # load data
    directory = 'E:/audio/sc09/wave/'
    zeros = np.load(directory + 'train_zero.npy')
    ones = np.load(directory + 'train_one.npy')
    twos = np.load(directory + 'train_two.npy')
    threes = np.load(directory + 'train_three.npy')
    fours = np.load(directory + 'train_four.npy')
    fives = np.load(directory + 'train_five.npy')
    sixes = np.load(directory + 'train_six.npy')
    sevens = np.load(directory + 'train_seven.npy')
    eights = np.load(directory + 'train_eight.npy')
    nines = np.load(directory + 'train_nine.npy')

    zero_labels = np.zeros((len(zeros), 1))
    one_labels = np.zeros((len(ones), 1)) + 1
    two_labels = np.zeros((len(twos), 1)) + 2
    three_labels = np.zeros((len(threes), 1)) + 3
    four_labels = np.zeros((len(fours), 1)) + 4
    five_labels = np.zeros((len(fives), 1)) + 5
    six_labels = np.zeros((len(sixes), 1)) + 6
    seven_labels = np.zeros((len(sevens), 1)) + 7
    eight_labels = np.zeros((len(eights), 1)) + 8
    nine_labels = np.zeros((len(nines), 1)) + 9

    # form inputs, labels, and raw_data
    inputs = zeros
    inputs = np.concatenate((inputs, ones))
    inputs = np.concatenate((inputs, twos))
    inputs = np.concatenate((inputs, threes))
    inputs = np.concatenate((inputs, fours))
    inputs = np.concatenate((inputs, fives))
    inputs = np.concatenate((inputs, sixes))
    inputs = np.concatenate((inputs, sevens))
    inputs = np.concatenate((inputs, eights))
    inputs = np.concatenate((inputs, nines))

    labels = zero_labels
    labels = np.concatenate((labels, one_labels))
    labels = np.concatenate((labels, two_labels))
    labels = np.concatenate((labels, three_labels))
    labels = np.concatenate((labels, four_labels))
    labels = np.concatenate((labels, five_labels))
    labels = np.concatenate((labels, six_labels))
    labels = np.concatenate((labels, seven_labels))
    labels = np.concatenate((labels, eight_labels))
    labels = np.concatenate((labels, nine_labels))

    raw_data = np.concatenate((inputs, labels), axis=1)
    np.random.shuffle(raw_data)

    # enforce float32 to save memory when reading it
    raw_data = np.array(raw_data, dtype=np.float32)

    np.save(directory + "train_labeled_data.npy", raw_data[0:cut*size])
    np.save(directory + "test_labeled_data.npy", raw_data[cut*size:])
    print(cut)
    print(raw_data.shape[0]/size - cut)


def load_complex_data():
    from complexLoader import wav_to_spec

    train_wave = np.load("E:/audio/sc09/wave/train_labeled_data.npy")  # (cut, 16385)
    test_wave = np.load("E:/audio/sc09/wave/test_labeled_data.npy")  # (17920-cut, 16385)
    train_wave_inputs = train_wave[:, 0:16384]
    train_labels = train_wave[:, 16384:16385] + 0j
    test_wave_inputs = test_wave[:, 0:16384]
    test_labels = test_wave[:, 16384:16385] + 0j

    train_spec_inputs = np.empty((train_wave.shape[0], 1, 128, 128))
    train_spec_inputs = np.array(train_spec_inputs, dtype=np.complex64)
    test_spec_inputs = np.empty((test_wave.shape[0], 1, 128, 128))
    test_spec_inputs = np.array(test_spec_inputs, dtype=np.complex64)

    # transform train inputs from wave to spec
    for i in range(train_wave_inputs.shape[0]):
        train_spec_inputs[i, 0] = wav_to_spec(train_wave_inputs[i])
    train_spec_inputs = np.reshape(train_spec_inputs, (train_spec_inputs.shape[0], 16384))

    # transform test inputs from wave to spec
    for i in range(test_wave_inputs.shape[0]):
        test_spec_inputs[i, 0] = wav_to_spec(test_wave_inputs[i])
    test_spec_inputs = np.reshape(test_spec_inputs, (test_spec_inputs.shape[0], 16384))

    # concatenate labels with inputs (batch, 16384+1)
    train_spec = np.concatenate((train_spec_inputs, train_labels), axis=1)
    test_spec = np.concatenate((test_spec_inputs, test_labels), axis=1)

    np.random.shuffle(train_spec), np.random.shuffle(test_spec)

    # enforce float32 to save memory when reading it
    train_spec = np.array(train_spec, dtype=np.complex64)
    test_spec = np.array(test_spec, dtype=np.complex64)

    np.save("E:/audio/sc09/complex/train_labeled_data.npy", train_spec)  # (cut, 16384+1)
    np.save("E:/audio/sc09/complex/test_labeled_data.npy", test_spec)  # (17920-cut, 16384+1)


def train_classifier(gpu, batch_size, epoch, regular):
    classifier = complexClassifier().cuda(gpu)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=1e-4, weight_decay=regular)

    train_data = np.load("E:/audio/sc09/complex/train_labeled_data.npy")  # (None, 16384+1) complex
    train_dataset = LabeledComplexDataSet(train_data, gpu=0)
    train_loader = data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    train_iterator = iter(train_loader)

    test_data = np.load("E:/audio/sc09/complex/test_labeled_data.npy")  # (None, 16384+1) complex
    test_input = np.reshape((test_data[:, 0:test_data.shape[1] - 1]), (test_data.shape[0], 1, 128, 128))
    test_input_real = torch.FloatTensor(test_input.real).cuda(gpu)  # (None, 1, 128, 128)
    test_input_imag = torch.FloatTensor(test_input.imag).cuda(gpu)  # (None, 1, 128, 128)
    test_label = torch.FloatTensor(test_data[:, test_data.shape[1] - 1:].real.reshape(-1)).long().cuda(gpu)

    threshold = 0.9
    for i in range(epoch):
        # train
        for t in range(int(train_data.shape[0] / size)):
            train_batch = next(train_iterator, None)
            if train_batch is None:
                train_iterator = iter(train_loader)
                train_batch = train_iterator.next()
            train_real_input, train_imag_input, train_label = train_batch
            train_input_real = train_real_input.cuda(gpu)
            train_input_imag = train_imag_input.cuda(gpu)
            train_label = train_label.view(-1).cuda(gpu)

            features, train_predict = classifier(train_input_real, train_input_imag)
            loss_train = criterion(train_predict, train_label)

            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()

        # test
        features, test_predict = classifier(test_input_real, test_input_imag)
        loss_test = criterion(test_predict, test_label)

        # precision
        predict = torch.max(test_predict.cpu(), 1)[1]  # get the index of the max for dim=1
        correct = predict.eq(test_label.cpu().view_as(predict)).sum()
        precision = correct.detach().numpy() / 64  # alter with the value of "cut"

        # print losses and precision
        print('epoch: {}, train loss: {:.4}, test loss: {:.4}, precision: {:}%'
              .format(i, loss_train, loss_test, precision*100))

        # save model
        if threshold <= precision:
            torch.save({'classifier': classifier.state_dict()},
                       'checkpoint_complexClassifier.pth')
            threshold = precision


if __name__ == '__main__':
    size = 16

    # load_wave_data(cut=350)
    # load_complex_data()
    #  #  22464 samples in total

    # train_classifier(gpu=0, batch_size=size, epoch=100, regular=0)


