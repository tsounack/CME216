import torch
import torch.nn as nn

class CAE(nn.Module):
    """
    This class creates an autoencoder comprised of an encoder, a decoder and a 
    hidden/bottleneck layer
    """

    def __init__(self, dim: int, h_dim: int) -> None:
        """
        Instantiates a CAE object with:
        -  dim: the number of feature maps/channels in the first convolution layer
        - hdim: dimension of the hidden/bottleneck layer
        """
        super(CAE, self).__init__()

        self.dim            = dim
        self.h_dim          = h_dim
        self.negative_slope = 0.25

        # Encoder
        self.down_scale = nn.Sequential(
            # block 1
            nn.Conv2d(1, self.dim, 4, 2, 1),
            nn.LeakyReLU(self.negative_slope),
            nn.BatchNorm2d(self.dim),
            # block 2
            nn.Conv2d(self.dim, 2 * self.dim, 4, 2, 1),
            nn.LeakyReLU(self.negative_slope),
            nn.BatchNorm2d(2 * self.dim),
            # block 3
            nn.Conv2d(2 * self.dim, 4 * self.dim, 4, 2, 1),
            nn.LeakyReLU(self.negative_slope),
            nn.BatchNorm2d(4 * self.dim),
            # block 4
            nn.Conv2d(4 * self.dim, 8 * self.dim, 4, 2, 1),
            nn.LeakyReLU(self.negative_slope),
            nn.BatchNorm2d(8 * self.dim)
        )

        # Hidden layer
        self.hidden_layer = nn.Sequential(
            nn.Linear(2 * 2 * 8 * self.dim, self.h_dim),
            nn.LeakyReLU(self.negative_slope),
            nn.BatchNorm1d(self.h_dim),
            nn.Linear(self.h_dim, 2 * 2 * 8 * self.dim),
            nn.LeakyReLU(self.negative_slope),
            nn.BatchNorm1d(2 * 2 * 8 * self.dim)
        )

        # Decoder
        self.up_scale = nn.Sequential(
            # block 1
            nn.ConvTranspose2d(8 * self.dim, 4 * self.dim, 4, 2, 1),
            nn.LeakyReLU(self.negative_slope),
            nn.BatchNorm2d(4 * self.dim),
            # block 2
            nn.ConvTranspose2d(4 * self.dim, 2 * self.dim, 4, 2, 1),
            nn.LeakyReLU(self.negative_slope),
            nn.BatchNorm2d(2 * self.dim),
            # block 3
            nn.ConvTranspose2d(2 * self.dim, self.dim, 4, 2, 1),
            nn.LeakyReLU(self.negative_slope),
            nn.BatchNorm2d(self.dim),
            # block 4
            nn.ConvTranspose2d(self.dim, 1, 4, 2, 1),
            nn.LeakyReLU(self.negative_slope)
          )
        
    def forward(self, x):
        x            = self.down_scale(x)
        reshaped_x   = torch.reshape(x, (-1, 2 * 2 * 8 * self.dim))
        hidden       = self.hidden_layer(reshaped_x)
        reshaped_hid = torch.reshape(hidden, (-1, 8 * self.dim, 2, 2))
        output       = self.up_scale(reshaped_hid)
        return output