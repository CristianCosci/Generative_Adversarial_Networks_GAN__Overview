import torch.nn as nn
import torch.nn.functional as F

# Discriminator Model Class Definition
class Discriminator(nn.Module):
    '''
    Discriminator model definition as a binary classifier.
    In the Deep Convolutional GAN it is defined as an CNN model with input size equal to
    flattened image size.
    The output size is the 1 (i.e. the probability of a binary problem -> real or fake).
    '''
    def __init__(self, img_size=64, output_size=1):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            # Block 1: (3) x 64 x 64
            nn.Conv2d(3, img_size, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # Block 2: (64) x 32 x 32
            nn.Conv2d(img_size, img_size * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(img_size * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # Block 3: (64*2) x 16 x 16
            nn.Conv2d(img_size * 2, img_size * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(img_size * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # Block 4: (64*4) x 8 x 8
            nn.Conv2d(img_size * 4, img_size * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(img_size * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # Block 5: (64*8) x 4 x 4
            nn.Conv2d(img_size * 8, output_size, 4, 1, 0, bias=False),
            nn.Sigmoid(),
            nn.Flatten()
            # Output: 1
        )

    def forward(self, input):
        output = self.model(input)
        return output