import torch.nn as nn
from Regularization import *
from sklearn.cluster import KMeans

class Encoder_layer(nn.Module):
    def __init__(self, in_channels, out_channels, normalize=True, dropout=0.0):
        super(Encoder_layer, self).__init__()
        # width and height is decreased by 2x
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class Latent_space_layer(nn.Module):
  def __init__(self, in_features, out_features):
    super(Latent_space_layer, self).__init__()
    layers = [nn.Linear(in_features, out_features)]
    self.model = nn.Sequential(*layers)

  def forward(self, x):
    return self.model(x)

class Decoder_layer(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super(Decoder_layer, self).__init__()
        # width and height are increased 2x
        layers = [nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)]
        layers.append(nn.InstanceNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
      return self.model(x)

class Encoder(nn.Module):
  def __init__(self, in_channels, latent_space):
    super(Encoder, self).__init__()
    self.latent_space = latent_space
    self.down1 = Encoder_layer(in_channels, 64, normalize=False) # [64 X 128 X 128]
    self.down2 = Encoder_layer(64, 128) # [128 X 64 X 64]
    self.down3 = Encoder_layer(128, 256) # [256 X 32 X 32]
    self.down4 = Encoder_layer(256, 512, dropout=0.5) # [512 X 16 X 16]
    self.down5 = Encoder_layer(512, 512, dropout=0.5) # [512 X 8 X 8]
    self.down6 = Encoder_layer(512, 512, dropout=0.5) # [512 X 4 X 4]
    self.down7 = Encoder_layer(512, 512, dropout=0.5) # [512 X 2 X 2]
    self.down8 = Encoder_layer(512, 512, normalize=False, dropout=0.5) # [512 X 1 X 1]
    self.fc1 = Latent_space_layer(512, 256)
    self.fc2 = Latent_space_layer(256, self.latent_space)
    # self.fc1 = Latent_space_layer(8192, 4096)
    # self.fc2 = Latent_space_layer(4096, self.latent_space)
    self.csd = 0
    self.mu = torch.tensor([[ 10.5407, -1.0915], [ 3.4946, -3.6491]])
    self.var = torch.tensor([[1., 1.], [1., 1.]])
    # self.g = Gaussian_mix_2d(classes=2, dim=2, mu=self.mu, var=self.var).sample([1000])

  def forward(self, x):
    d1 = self.down1(x)
    d2 = self.down2(d1)
    d3 = self.down3(d2)
    d4 = self.down4(d3)
    d5 = self.down5(d4)
    d6 = self.down6(d5)
    d7 = self.down7(d6)
    d8 = self.down8(d7)
    x = d8.view(d8.size(0), -1)
    # x = d6.view(d6.size(0), -1)
    x = self.fc1(x)
    x = self.fc2(x)
    g = Gaussian_mix_2d(classes=2, dim=2, mu=self.mu, var=self.var).sample([1000])
    # g = Gaussian_mix_2d(classes=2, dim=2).sample([1000])
    vxx = Regularizer().get_kernel(x, x)
    vzz = Regularizer().get_kernel(g, g)
    vxz = Regularizer().get_kernel(x, g)
    self.csd = torch.log(torch.sqrt(torch.mean(vxx) * torch.mean(vzz)) /
                         torch.mean(vxz))
    return x

class Decoder(nn.Module):
  def __init__(self, out_channels, latent_space):
    super(Decoder, self).__init__()
    self.latent_space = latent_space
    self.fc3 = Latent_space_layer(self.latent_space, 256)
    self.fc4 = Latent_space_layer(256, 512)
    # self.fc3 = Latent_space_layer(self.latent_space, 4096)
    # self.fc4 = Latent_space_layer(4096, 8192)
    self.up1 = Decoder_layer(512, 512, dropout=0.5) # [512 X 2 X 2]
    self.up2 = Decoder_layer(512, 512, dropout=0.5) # [512 X 4 X 4]
    self.up3 = Decoder_layer(512, 512, dropout=0.5) # [512 X 8 X 8]
    self.up4 = Decoder_layer(512, 512, dropout=0.5) # [512 X 16 X 16]
    self.up5 = Decoder_layer(512, 256) # [256 X 32 X 32]
    self.up6 = Decoder_layer(256, 128) # [128 X 64 X 64]
    self.up7 = Decoder_layer(128, 64) # [64 X 128 X 128]
    self.final = nn.Sequential(
            nn.Upsample(scale_factor=2), # [128 X 256 X 256]
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(64, out_channels, kernel_size=4, padding=1), # [out_channels X 256 X 256]
            nn.Tanh(),
        )

  def forward(self, x):
    x = self.fc3(x)
    x = self.fc4(x)
    # x = x.view(-1, 512, 4, 4)
    x = x.view(-1, 512, 1, 1)
    u1 = self.up1(x)
    u2 = self.up2(u1)
    u3 = self.up3(u2)
    # u3 = self.up3(x)
    u4 = self.up4(u3)
    u5 = self.up5(u4)
    u6 = self.up6(u5)
    u7 = self.up7(u6)
    return self.final(u7)

class Autoencoder(nn.Module):
    def __init__(self, in_channels, out_channels, latent_space):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(in_channels, latent_space)
        self.decoder = Decoder(out_channels, latent_space)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

class Kmeans:
    def __init__(self):
        self.kmeans = KMeans(n_clusters=2)
