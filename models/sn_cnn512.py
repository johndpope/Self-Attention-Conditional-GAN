import torch.nn as nn
import torch.nn.utils.spectral_norm as SpectralNorm


class Generator(nn.Module):
    def __init__(self,z_dim=128,channels=3,n_classes = 0):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        self.model = nn.Sequential(
            nn.ConvTranspose2d(z_dim, 512, 4, stride=1, bias=False),  # 4x4 (dense)
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=(1, 1), bias=False),  # 2x
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=(1, 1), bias=False),  # 2x
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=(1, 1), bias=False),  # 2x
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=(1, 1), bias=False),  # 2x
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.ConvTranspose2d(32, channels, 3, stride=1, padding=(1, 1)),  # 1x (conv)
            nn.Tanh())

    def forward(self, input,y = None):
        return self.model(input.view(-1,self.z_dim, 1, 1))


class Discriminator(nn.Module):
    def __init__(self,channels=3,leak =0.1,n_classes = 0):
        super(Discriminator, self).__init__()
        self.leak = leak
        self.conv1 = SpectralNorm(nn.Conv2d(channels, 64, 3, stride=1, padding=(1, 1)))
        self.conv2 = SpectralNorm(nn.Conv2d(64, 64, 4, stride=2, padding=(1, 1)))  # x/2

        self.conv3 = SpectralNorm(nn.Conv2d(64, 128, 3, stride=1, padding=(1, 1)))
        self.conv4 = SpectralNorm(nn.Conv2d(128, 128, 4, stride=2, padding=(1, 1)))  # x/2

        self.conv5 = SpectralNorm(nn.Conv2d(128, 256, 3, stride=1, padding=(1, 1)))
        self.conv6 = SpectralNorm(nn.Conv2d(256, 256, 4, stride=2, padding=(1, 1)))  # x/2

        self.conv7 = SpectralNorm(nn.Conv2d(256, 512, 3, stride=1, padding=(1, 1)))  # x

        self.fc = SpectralNorm(nn.Linear(8 * 8 * 512, 1))  # dens

    def forward(self, input,y=None):
        m = input
        m = nn.LeakyReLU(self.leak)(self.conv1(m))
        m = nn.LeakyReLU(self.leak)(self.conv2(m))
        m = nn.LeakyReLU(self.leak)(self.conv3(m))
        m = nn.LeakyReLU(self.leak)(self.conv4(m))
        m = nn.LeakyReLU(self.leak)(self.conv5(m))
        m = nn.LeakyReLU(self.leak)(self.conv6(m))
        m = nn.LeakyReLU(self.leak)(self.conv7(m))

        return self.fc(m.view(-1, 8 * 8  * 512))