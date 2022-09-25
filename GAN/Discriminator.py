import torch.nn as nn
import torch.nn.functional as F

class Discriminator(nn.Module):
    '''
    Discriminator model definition as a binary classifier.
    In the Vanilla GAN it is defined as an MLP model with input size equal to
    flattened image size.
    The output size is the 1 (i.e. the probability of a binary problem -> real or fake).
    '''

    def __init__(self, input_size=784, num_classes=1):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, num_classes),
            nn.Sigmoid(), #The final layer has the sigmoid activation function, 
                        #which squashes the output value between 0 (fake) and 1 (real).
            )


    def forward(self, image):
        image_flattened = image.view(image.size(0), -1)
        result = self.model(image_flattened)
        return result