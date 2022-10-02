import torch.nn as nn
import torch.nn.functional as F

NOISE_VECTOR_DIM = 100

# Generator Model Class Definition      
class Generator(nn.Module):
    '''
    Generator model definition.
    In the Deep Convolutional GAN it is defined as an CNN model with input size 
    equal to noise vector.
    The output size is the same as images we want to generate
    (in this case is 3 x 64 x 64).
    The model has been divided into 5 blocks, and each block consists of:
    - A Convolution 2D Transpose Layer
    - Followed by a BatchNorm Layer and LeakyReLU Activation Function
    - A tanh Activation Function in the last block, instead of ReLU. 
    '''
    def __init__(self, input_size=NOISE_VECTOR_DIM):
        super(Generator, self).__init__()

        self.model = nn.Sequential(
            # Block 1:input is Z, going into a convolution
            nn.ConvTranspose2d(input_size, 64 * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(64 * 8),
            nn.LeakyReLU(0.2),

            # Block 2: (64 * 8) x 4 x 4
            nn.ConvTranspose2d(64 * 8, 64 * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 4),
            nn.LeakyReLU(0.2),

            # Block 3: (64 * 4) x 8 x 8
            nn.ConvTranspose2d(64 * 4, 64 * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 2),
            nn.LeakyReLU(0.2),

            # Block 4: (64 * 2) x 16 x 16
            nn.ConvTranspose2d(64 * 2, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            # Block 5: (64) x 32 x 32
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
            # Output: (3) x 64 x 64
        )

    def forward(self, noise_vector):
        generated_img = self.model(noise_vector)
        return generated_img