import torch
import torch.nn as nn
import torch.nn.functional as F

NOISE_VECTOR_DIM = 100

class Generator(nn.Module):
    '''
    Generator model definition.
    In the Conditional Deep Convolutional GAN it is defined as an CNN model with
    input size equal to noise vector plus the 10-class one-ho-encoding vector.
    The output size is the same as images we want to generate.
    (in this case is 1 x 28 x 28).
    The model has been divided into 3 blocks, and each block consists of:
    - A Convolution 2D Transpose Layer
    - Followed by a BatchNorm Layer and LeakyReLU Activation Function
    - A tanh Activation Function in the last block, instead of ReLU.

    Before process data through the CNN network, the input is preprocessed 
    in a fully connected layer that produce an output of size 4x4x512.
    '''
    def __init__(self, input_size=100, condition_size=10):
        super(Generator, self).__init__()
        
        self.fully_connected = nn.Sequential(
            nn.Linear(input_size+condition_size, 4*4*512),
            nn.ReLU(),
        )

        self.convolutional_network = nn.Sequential(
            # input: 4 by 4, output: 7 by 7
            nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            # input: 7 by 7, output: 14 by 14
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            # input: 14 by 14, output: 28 by 28
            nn.ConvTranspose2d(128, 1, 4, stride=2, padding=1, bias=False),
            nn.Tanh(),
        )
        
    def forward(self, x, c):
        # x: (N, 100), c: (N, 10)
        x, c = x.view(x.size(0), -1), c.float() # may not need
        noise_vector_with_class = torch.cat((x, c), 1) # v: (N, 110)
        y_ = self.fully_connected(noise_vector_with_class)
        y_ = y_.view(y_.size(0), 512, 4, 4)
        generated_img = self.convolutional_network(y_) # (N, 28, 28)
        return generated_img