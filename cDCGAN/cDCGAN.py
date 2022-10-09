import torch
from Discriminator import *
from Generator import *
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(1)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def to_onehot(x, num_classes=10):
    '''
    One-Hot encoding of MNIST classes to use as model input (with noise vector).

    returns:
    - one-hot encoding of input vector with class labels.
    '''
    assert isinstance(x, int) or isinstance(x, (torch.LongTensor, torch.cuda.LongTensor))
    if isinstance(x, int):
        c = torch.zeros(1, num_classes).long()
        c[0][x] = 1
    else:
        x = x.cpu()
        c = torch.LongTensor(x.size(0), num_classes)
        c.zero_()
        c.scatter_(1, x, 1) # dim, index, src value
    return c


def get_sample_images(G, num_samples):
    '''
    Generate num_samples noise vector from latent space and pass to Generator
    as input in order to get sample images.

    returns:
    num_samples generate images
    '''
    samples_per_line = int(np.sqrt(num_samples))
    matrix_img = np.zeros([28 * samples_per_line, 28 * samples_per_line])
    for j in range(10):
        c = torch.zeros([10, 10]).to(DEVICE)
        c[:, j] = 1
        z = torch.randn(samples_per_line, NOISE_VECTOR_DIM).to(DEVICE)
        y_hat = G(z,c).view(samples_per_line, 28, 28)
        result = y_hat.cpu().data.numpy()
        matrix_img[j*28:(j+1)*28] = np.concatenate([x for x in result], axis=-1)

    return matrix_img


#Models creation
num_samples = 100
# D = Discriminator().to(DEVICE) is not actually neede
G = Generator().to(DEVICE)
G.load_state_dict(torch.load('trained_model/Generator.pth',  map_location=torch.device('cpu')), strict=False)
print(G.eval())

# generation to image
plt.figure(figsize=(7, 4)) 
plt.imshow(get_sample_images(G, num_samples), cmap='gray')
plt.show()