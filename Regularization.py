import torch
import numpy as np
from torch import distributions as D
from torch.distributions.mixture_same_family import MixtureSameFamily

device = 'cuda' if torch.cuda.is_available() else 'cpu'
class Regularizer():
    def __init__(self, scale=1., ksize=100):
        super(Regularizer, self).__init__()
        self.scale = scale
        self.ksize = ksize

    def get_kernel(self, X, Z):
        X = X.to(device)
        Z = Z.to(device)
        G = torch.sum((torch.unsqueeze(X, dim=1) - Z) ** 2, axis=-1)
        G = torch.exp(-G / (2 * (self.ksize ** 2))) / (np.sqrt(2 * np.pi) * self.ksize)
        return G

mu = torch.tensor([[ 10.5407, -1.0915],
        [ 3.4946, -3.6491]])
var = torch.tensor([[1., 1.],
        [1., 1.]])

def Gaussian_mix_2d(classes, dim, mu, var):
  mix = D.Categorical(torch.ones(classes,))
  comp = D.Independent(D.Normal(mu, var), 1)
  gmm = MixtureSameFamily(mix, comp)
  return gmm