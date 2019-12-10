from train import *
from wavegan import *
from loader import *

amp = np.load("./data/sc09/train_amplitudes.npy")
ph = np.load("./data/sc09/train_phases.npy")
batch_size = 64
num_batch = 100
splits = 1

from train_classifier import Classifier_Spec

classifier = Classifier_Spec()
check = torch.load('checkpoint_classifier_spec.pth')
classifier.load_state_dict(check['classifier'])
classifier = classifier.cuda(0)
classifier.eval()


def softmax_predict(a, b):
    a = np.reshape(a, (batch_size, 1, 128, 128))
    b = np.reshape(b, (batch_size, 1, 128, 128))
    x = np.concatenate((a, b), axis=1)
    x = torch.FloatTensor(x).cuda(0)
    x = x.view(batch_size, 2, 128, 128)
    x = classifier(x)
    return F.softmax(x).data.cpu().numpy()


data_len = batch_size * num_batch
predicts = np.zeros((data_len, 10))

for i in range(num_batch):
    a = amp[i * batch_size:(i + 1) * batch_size]
    b = ph[i * batch_size:(i + 1) * batch_size]
    predicts[i * batch_size:(i + 1) * batch_size] = softmax_predict(a, b)

# compute the mean kl-div
split_scores = []

for k in range(splits):
    part = predicts[k * (data_len // splits): (k + 1) * (data_len // splits), :]
    py = np.mean(part, axis=0)
    scores = []
    for i in range(part.shape[0]):
        pyx = part[i, :]
        scores.append(entropy(pyx, py))
    split_scores.append(np.exp(np.mean(scores)))

print("mean:{}, std:{}".format(np.mean(split_scores), np.std(split_scores)))