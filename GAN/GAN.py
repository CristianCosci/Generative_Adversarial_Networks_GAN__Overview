import torch
from Discriminator import *
from Generator import *
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(1)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_sample_images(G, num_samples):
    '''
    Generate num_samples noise vector from latent space and pass to Generator
    as input in order to get sample images.

    returns:
    num_samples generate images
    '''
    assert int(np.sqrt(num_samples)) * int(np.sqrt(num_samples)) == num_samples  , "num_samples square is not integer"

    z = torch.randn(num_samples, NOISE_VECTOR_DIM).to(DEVICE)  #z.shape = torch.Size([num_samples, 100])
    y_hat = G(z).view(num_samples, 28, 28) # (100, 28, 28)
    samples_img = y_hat.cpu().data.numpy()

    #Put generated samples in a matrix of size sqrt(num_samples) x sqrt(num_samples)
    #in order to plot all samples togheter
    samples_per_line = int(np.sqrt(num_samples))
    matrix_img = np.zeros([28 * samples_per_line, 28 * samples_per_line])
    for j in range(samples_per_line): 
        matrix_img[j*28:(j+1)*28] = np.concatenate([x for x in samples_img[j*samples_per_line:(j+1)*samples_per_line]], axis=-1)
        
    return matrix_img


#Models creation
num_samples = 100
# D = Discriminator().to(DEVICE) is not actually neede
G = Generator().to(DEVICE)
G.load_state_dict(torch.load('trained_model/G_epoch_199.pth',  map_location=torch.device('cpu')), strict=False)
print(G.eval())

# generation to image
plt.figure(figsize=(10,10)) 
plt.imshow(get_sample_images(G, num_samples), cmap='gray')
plt.show()