import torch
import torch.nn.functional as F
import torch.nn as nn


class Encoder(torch.nn.Module):
    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 categorical_dim: int,
                 hidden_dims = None,
                 temperature: float = 0.5,
    ):
        super(Encoder, self).__init__()
        self.latent_dim = latent_dim
        self.categorical_dim = categorical_dim
        self.temp = temperature
        
        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size= 3, stride= 2, padding  = 1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1]*4, self.latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1]*4, self.latent_dim)
        self.fc_z = nn.Linear(hidden_dims[-1]*4, self.categorical_dim)
   
    def encode(self, input):
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)
        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        z = self.fc_z(result)
        z = z.view(-1, self.categorical_dim)
        return [mu, log_var, z]

class Decoder(torch.nn.Module):
    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 categorical_dim: int,
                 hidden_dims = None,
                 temperature: float = 0.5):
        super(Decoder, self).__init__()
        self.latent_dim = latent_dim
        self.categorical_dim = categorical_dim
        self.temp = temperature
        
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]
        # Build Decoder
        modules = []
        self.decoder_input = nn.Linear(self.latent_dim + self.categorical_dim,
                                       hidden_dims[-1] * 4)
        hidden_dims.reverse()
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride = 2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )
        self.decoder = nn.Sequential(*modules)
        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(hidden_dims[-1],
                                               hidden_dims[-1],
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               output_padding=1),
                            nn.BatchNorm2d(hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv2d(hidden_dims[-1], out_channels= 3,
                                      kernel_size= 3, padding= 1),
                            nn.Tanh())
        self.sampling_dist = torch.distributions.OneHotCategorical(1. / categorical_dim * torch.ones((self.categorical_dim, 1)))

    def decode(self, z):
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D x Q]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        result = result.view(-1, 512, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result


class VAE(torch.nn.Module):
    def __init__(self, input, latent, classes):
        super(VAE, self).__init__()
        self.encoder = Encoder(in_channels=input, latent_dim=latent, categorical_dim=classes)
        self.decoder = Decoder(in_channels=input, latent_dim=latent, categorical_dim=classes)

    def reparameterize(self,
                       mu,
                       log_var,
                       q,
                       eps:float = 1e-7):
        """
        Gumbel-softmax trick to sample from Categorical Distribution
        :param mu: (Tensor) mean of the latent Gaussian  [B x D]
        :param log_var: (Tensor) Log variance of the latent Gaussian [B x D]
        :param q: (Tensor) Categorical latent Codes [B x Q]
        :return: (Tensor) [B x (D + Q)]
        """

        std = torch.exp(0.5 * log_var)
        e = torch.randn_like(std)
        z = e * std + mu

        # Sample from Gumbel
        u = torch.rand_like(q)
        g = - torch.log(- torch.log(u + eps) + eps)

        # Gumbel-Softmax sample
        s = F.softmax((q + g) / self.temp, dim=-1)
        s = s.view(-1, self.categorical_dim)

        return torch.cat([z, s], dim=1)
    
    def forward(self, x):
        mu, log_var, q = self.encoder.encode(x)
        z = self.reparameterize(mu, log_var, q)
        return  [self.decoder.decode(z), x, q, mu, log_var]



#self.min_temp = temperature
        #self.anneal_rate = anneal_rate
        #self.anneal_interval = anneal_interval
        #self.alpha = alpha

        #self.cont_min = latent_min_capacity
        #self.cont_max = latent_max_capacity

        # self.disc_min = categorical_min_capacity
        # self.disc_max = categorical_max_capacity

        # self.cont_gamma = latent_gamma
        # self.disc_gamma = categorical_gamma

        # self.cont_iter = latent_num_iter
        # self.disc_iter = categorical_num_iter