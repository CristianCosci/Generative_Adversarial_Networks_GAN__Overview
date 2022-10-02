import torch
from Discriminator import *
from Generator import *
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def show_images(images, column=5, fig_size=(5, 5)):
    '''
    Function used to plot images (input as a batch).
    - column parameter is for choose the layout of plot.
    - fig_size is used for select the size (n x m) of the single batch's images
    '''
    images = images / 2 + 0.5
    fig, ax = plt.subplots(figsize=fig_size)
    ax.set_xticks([]); ax.set_yticks([])
    ax.imshow(make_grid(images.detach(), nrow=column).permute(1, 2, 0))
    plt.show()


#Models creation
num_samples = 100
# D = Discriminator().to(DEVICE) is not actually neede
G = Generator().to(DEVICE)
G.load_state_dict(torch.load('trained_model/Generator.pth',  map_location=torch.device('cpu')), strict=False)
print(G.eval())

z = torch.randn(25, NOISE_VECTOR_DIM, 1, 1).to(DEVICE)
generated_image = G(z)
show_images(generated_image.cpu(), column=5, fig_size=(6,6))