import torch
import torch.nn as nn
import torch.nn.functional as F

class Discriminator(nn.Module):
    '''
    Discriminator model definition as a binary classifier.
    In the Conditional Deep Convolutional GAN it is defined as an CNN model with
    input size equal to flattened image size plus the one-hot-ecnoding.
    The output size is the 1 (i.e. the probability of a binary problem -> real or fake).
    Before process data through the CNN network, the input is preprocessed 
    in a fully connected layer that produce an output of size 784 (1 x 28 x 28).
    '''
    def __init__(self, in_channel=1, input_size=784, condition_size=10, num_classes=1):
        super(Discriminator, self).__init__()

        self.transform = nn.Sequential(
            nn.Linear(input_size+condition_size, 784),
            nn.LeakyReLU(0.2),
        )

        self.convolutional_network = nn.Sequential(
            # 28 -> 14
            nn.Conv2d(in_channel, 512, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            # 14 -> 7
            nn.Conv2d(512, 256, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            # 7 -> 4
            nn.Conv2d(256, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.AvgPool2d(4),
        )

        self.fully_connected = nn.Sequential(
            # reshape input, 128 -> 1
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, x, c):
        # x: (N, 1, 28, 28), c: (N, 10)
        x, c = x.view(x.size(0), -1), c.float() # may not need
        input = torch.cat((x, c), 1) # v: (N, 794)
        image_flattened_with_class = self.transform(input) # (N, 784)
        image_flattened_with_class = image_flattened_with_class.view(image_flattened_with_class.shape[0], 1, 28, 28) # (N, 1, 28, 28)
        y_ = self.convolutional_network(image_flattened_with_class)
        y_ = y_.view(y_.size(0), -1)
        output = self.fully_connected(y_)
        return output