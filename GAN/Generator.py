import torch.nn as nn
import torch.nn.functional as F

NOISE_VECTOR_DIM = 100

class Generator(nn.Module):
    '''
    Generator model definition.
    In the Vanilla GAN it is defined as an MLP model with input size equal to noise vector.
    The output size is the same as images we want to generate.
    '''
    def __init__(self, input_size=NOISE_VECTOR_DIM, output_size=784):
        super(Generator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, output_size),
            nn.Tanh()   #The tanh activation at the output layer ensures that the pixel 
                        #values are mapped in line with its own output, i.e., between (-1, 1)
            )


    def forward(self, noise_vector):
        generated_img = self.model(noise_vector)
        generated_img = generated_img.view(generated_img.size(0), 1, 28, 28)
        return generated_img