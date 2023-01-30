import torch
import torch.nn.functional as F
import torch.nn as nn

class VariationalEncoder(nn.Module):
    def __init__(self, in_channels, latent_dims):
        super(VariationalEncoder, self).__init__()
        self.linear1 = nn.Linear(in_channels, 512)
        self.linear2 = nn.Linear(512, latent_dims)
        self.linear3 = nn.Linear(512, latent_dims)
        
        self.N = torch.distributions.Normal(0, 1)
        #self.N.loc = self.N.loc.cuda() # hack to get sampling on the GPU
        #self.N.scale = self.N.scale.cuda()
        self.kl = 0
    
    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.linear1(x))
        mu =  self.linear2(x)
        sigma = torch.exp(self.linear3(x))
        z = mu + sigma*self.N.sample(mu.shape)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        return z
    
class Decoder(nn.Module):
    def __init__(self, latent_dims, out_channels):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(latent_dims, 512)
        self.linear2 = nn.Linear(512, out_channels)
        
    def forward(self, z):
        z = F.relu(self.linear1(z))
        z = torch.sigmoid(self.linear2(z))
        return z.reshape((32, 1024))

class VariationalAutoencoder(nn.Module):
    def __init__(self, in_channels, latent_dims, out_channels):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = VariationalEncoder(in_channels, latent_dims)
        self.decoder = Decoder(latent_dims, out_channels)
    
    def load_on(self, device):
        self.encoder = self.encoder.to(device)
        self.decoder = self.decoder.to(device)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)