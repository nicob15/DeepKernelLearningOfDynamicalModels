import torch
import torch.nn as nn
import torch.nn.functional as F
import gpytorch
import math

# for 84x84 inputs
OUT_DIM = {2: 39, 4: 35, 6: 31}

class GaussianProcessLayer(gpytorch.models.ApproximateGP):
    def __init__(self, num_dim, grid_bounds=(-10., 10.), grid_size=64):

        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            num_inducing_points=grid_size, batch_shape=torch.Size([num_dim])
        )
        # Our base variational strategy is a GridInterpolationVariationalStrategy,
        # which places variational inducing points on a Grid
        # We wrap it with a IndependentMultitaskVariationalStrategy so that our output is a vector-valued GP
        variational_strategy = gpytorch.variational.IndependentMultitaskVariationalStrategy(
            gpytorch.variational.GridInterpolationVariationalStrategy(
                self, grid_size=grid_size, grid_bounds=[grid_bounds],
                variational_distribution=variational_distribution,
            ), num_tasks=num_dim,
        )
        super().__init__(variational_strategy)

        self.covar_module = gpytorch.kernels.ScaleKernel(
            base_kernel=gpytorch.kernels.RBFKernel(ard_num_dims=num_dim,
                lengthscale_prior=gpytorch.priors.SmoothedBoxPrior(
                    math.exp(-1), math.exp(1), sigma=0.1, transform=torch.exp
                )
            )
        )
        self.mean_module = gpytorch.means.ConstantMean()
        self.grid_bounds = grid_bounds

        #self.normalizer = torch.nn.BatchNorm1d(1)

    def forward(self, x):
        #x = self.normalizer(x)
        #x = (x - x.mean(-2)) / x.std(-2)
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)

class SVDKL_AE_latent_dyn(gpytorch.Module):
    def __init__(self, num_dim, grid_bounds=(-10., 10.), a_dim=1, h_dim=32, grid_size=32):
        super(SVDKL_AE_latent_dyn, self).__init__()
        self.gp_layer = GaussianProcessLayer(num_dim=num_dim, grid_bounds=grid_bounds, grid_size=grid_size)
        self.grid_bounds = grid_bounds
        self.num_dim = num_dim

        self.encoder = Encoder(self.num_dim) # NN model
        self.decoder = StochasticDecoder(self.num_dim) # NN model
        self.fwd_model_DKL = Forward_DKLModel(num_dim=num_dim, grid_bounds=grid_bounds, h_dim=h_dim, a_dim=a_dim, grid_size=grid_size) # DKL forward model

        # This module will scale the NN features so that they're nice values
        self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(self.grid_bounds[0], self.grid_bounds[1])

    def forward_encoder_DKL(self, x):
        features = self.encoder(x)
        features = self.scale_to_bounds(features)
        # This next line makes it so that we learn a GP for each feature
        features = features.transpose(-1, -2).unsqueeze(-1)
        if self.training:
        # The next three lines are required to clear the GP test time caches since the GP parameters will change each time
            with gpytorch.settings.detach_test_caches(False):
                self.gp_layer.train()
                self.gp_layer.eval()
                res = self.gp_layer(features)
        else:
            # If we aren't in training mode, we don't expect the GP parameters to change each iteration so we don't need to clear the caches.
            res = self.gp_layer(features)
        mean = res.mean
        var = res.variance
        z = res.rsample()
        return res, mean, var, z


    def forward(self, x, a, x_next):
        res, mu, var, z = self.forward_encoder_DKL(x)
        res_target, mu_target, var_target, z_target = self.forward_encoder_DKL(x_next)
        mu_x, var_x = self.decoder.decoder(z)
        res_fwd, mu_fwd, var_fwd, z_fwd = self.fwd_model_DKL(z, a)
        return mu_x, var_x, mu, var, z, res, mu_target, var_target, res_target, mu_fwd, var_fwd, res_fwd


    def predict_dynamics(self, x, a, likelihood_fwd, likelihood):

        n_samples = 1

        res, mu, var, z = self.forward_encoder_DKL(x)
        z = likelihood(res).sample(sample_shape=torch.Size([n_samples])).view(n_samples, self.num_dim).mean(0).view(1, self.num_dim)
        # predicted distribution
        res_fwd, mu_fwd, var_fwd, z_fwd = self.fwd_model_DKL(z, a)

        #mu_x, var_x = self.decoder(likelihood_fwd(res_fwd).sample())
        mu_x = self.decoder(likelihood(res).sample(sample_shape=torch.Size([n_samples])).view(n_samples, self.num_dim))[0].mean(0)

        mu_x_2 = self.decoder(likelihood_fwd(res_fwd).sample(sample_shape=torch.Size([n_samples])).view(n_samples, self.num_dim))[0].mean(0)

        return mu_x, mu_x_2

    def predict_trajectory(self, x, a, likelihood_fwd, likelihood):

        n_samples = 1
        res, mu, var, z = self.forward_encoder_DKL(x)
        z = likelihood(res).sample(sample_shape=torch.Size([n_samples])).view(n_samples, self.num_dim).mean(0).view(1, self.num_dim)
        # predicted distribution
        res_fwd, mu_fwd, var_fwd, z_fwd = self.fwd_model_DKL(z, a)

        var = likelihood(res).variance
        lower, upper = mu - torch.sqrt(var), mu + torch.sqrt(var)#likelihood(res).confidence_region()
        lower_fwd, upper_fwd = likelihood_fwd(res_fwd).confidence_region()

        mu_x = self.decoder(likelihood(res).sample(sample_shape=torch.Size([n_samples])).view(n_samples, self.num_dim))[0].mean(0)

        mu_x_2 = self.decoder(likelihood_fwd(res_fwd).sample(sample_shape=torch.Size([n_samples])).view(n_samples, self.num_dim))[0].mean(0)

        return mu, mu_fwd, lower, upper, lower_fwd, upper_fwd, mu_x, mu_x_2, z

    def predict_latent_dynamics(self, z, a, likelihood_fwd):

        n_samples = 1
        # predicted distribution
        mu = 1
        var = 1
        res_fwd, mu_fwd, var_fwd, z_fwd = self.fwd_model_DKL(z, a)
        lower, upper = likelihood_fwd(res_fwd).confidence_region()
        z_fwd = likelihood_fwd(res_fwd).sample()

        mu_x, var_x = self.decoder(z_fwd)

        return mu_x, mu_fwd, lower, upper, z_fwd

class Forward_DKLModel(gpytorch.Module):
    def __init__(self, num_dim, grid_bounds=(-100., 100.), h_dim=256, a_dim=1, grid_size=32):
        super(Forward_DKLModel, self).__init__()
        self.gp_layer_2 = GaussianProcessLayer(num_dim=num_dim, grid_bounds=grid_bounds, grid_size=grid_size)
        self.grid_bounds = grid_bounds
        self.num_dim = num_dim

        self.fwd_model = ForwardModel(num_dim, h_dim, a_dim) # NN model

        # This module will scale the NN features so that they're nice values
        self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(self.grid_bounds[0], self.grid_bounds[1])

    def forward(self, x, a):
        features = self.fwd_model(x, a)
        features = self.scale_to_bounds(features)
        # This next line makes it so that we learn a GP for each feature
        features = features.transpose(-1, -2).unsqueeze(-1)
        if self.training:
            # The next three lines are required to clear the GP test time caches since the GP parameters will change each time
            with gpytorch.settings.detach_test_caches(False):
                self.gp_layer_2.train()
                self.gp_layer_2.eval()
                res = self.gp_layer_2(features)
        else:
            # If we aren't in training mode, we don't expect the GP parameters to change each iteration so we don't need to clear the caches.
            res = self.gp_layer_2(features)
        mean = res.mean
        var = res.variance
        z = res.rsample()
        return res, mean, var, z

class ForwardModel(nn.Module):
    def __init__(self, z_dim=20, h_dim=256, a_dim=1):
        super(ForwardModel, self).__init__()

        self.fc = nn.Linear(z_dim + a_dim, h_dim)
        self.fc1 = nn.Linear(h_dim, h_dim)
        self.fc12 = nn.Linear(h_dim, h_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)

        self.batch = nn.BatchNorm1d(z_dim)

    def forward(self, z, a):
        za = torch.cat([z, a], dim=1)
        za = F.elu(self.fc(za))
        za = F.elu(self.fc1(za))
        features = self.fc2(za)
        return features

class Encoder(nn.Module):
    def __init__(self, z_dim=20):
        super(Encoder, self).__init__()

        self.conv1 = nn.Conv2d(6, 32, (3, 3), stride=(2, 2))
        self.conv2 = nn.Conv2d(32, 32, (3, 3), stride=(1, 1))
        self.batch1 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, (3, 3), stride=(1, 1))
        self.conv4 = nn.Conv2d(32, 32, (3, 3), stride=(1, 1))
        self.batch2 = nn.BatchNorm2d(32)
        out_dim = OUT_DIM[4]
        self.fc = nn.Linear(32 * out_dim * out_dim, 256)
        self.fc1 = nn.Linear(256, z_dim)

    def encoder(self, x):
        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))
        x = self.batch1(x)
        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x))
        x = self.batch2(x)
        x = torch.flatten(x, start_dim=1)
        x = F.elu(self.fc(x))
        x = self.fc1(x)
        return x

    def forward(self, x):
        return self.encoder(x)

class ForwardModelVAE(nn.Module):
    def __init__(self, z_dim=20, h_dim=256, a_dim=1, fixed_std=True):
        super(ForwardModelVAE, self).__init__()

        self.fixed_std = fixed_std

        self.fc = nn.Linear(z_dim + a_dim, h_dim)
        self.fc1 = nn.Linear(h_dim, h_dim)
        self.fcmu = nn.Linear(h_dim, z_dim)
        self.fcvar = nn.Linear(h_dim, z_dim)

    def forward(self, z, a):
        za = torch.cat([z, a], dim=1)
        za = F.elu(self.fc(za))
        za = F.elu(self.fc1(za))
        z_next_mu = self.fcmu(za)
        if self.fixed_std:
            z_next_std = torch.ones_like(z_next_mu).detach()
        else:
            z_next_std = F.relu(z_next_mu) + 1e-4
        return z_next_mu, z_next_std

class EncoderVAE(nn.Module):
    def __init__(self, z_dim=20, h_dim=256):
        super(EncoderVAE, self).__init__()

        self.conv1 = nn.Conv2d(6, 32, (3, 3), stride=(2, 2))
        self.conv2 = nn.Conv2d(32, 32, (3, 3), stride=(1, 1))
        self.batch1 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, (3, 3), stride=(1, 1))
        self.conv4 = nn.Conv2d(32, 32, (3, 3), stride=(1, 1))
        self.batch2 = nn.BatchNorm2d(32)
        out_dim = OUT_DIM[4]
        self.fc0 = nn.Linear(32 * out_dim * out_dim, h_dim)
        self.fc = nn.Linear(h_dim, z_dim)
        self.fc1 = nn.Linear(h_dim, z_dim)

    def sampling(self, mu, std):
        std = torch.exp(0.5 * torch.log(torch.square(std)))
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)  # return z sample

    def encoder(self, x):
        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))
        x = self.batch1(x)
        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x))
        x = self.batch2(x)
        x = torch.flatten(x, start_dim=1)
        x = F.elu(self.fc0(x))
        mu = self.fc(x)
        std = F.relu(self.fc1(x)) + 1e-4
        return mu, std

    def forward(self, x):
        mu, std = self.encoder(x)
        return mu, std, self.sampling(mu, std)


class Decoder(nn.Module):
    def __init__(self, z_dim=20):
        super(Decoder, self).__init__()

        # decoder part
        out_dim = OUT_DIM[4]
        self.fcz = nn.Linear(z_dim, 32 * out_dim * out_dim)
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(32, out_dim, out_dim))
        self.deconv1 = nn.ConvTranspose2d(32, 32, (3, 3), stride=(1, 1))
        self.deconv2 = nn.ConvTranspose2d(32, 32, (3, 3), stride=(1, 1))
        self.batch3 = nn.BatchNorm2d(32)
        self.deconv3 = nn.ConvTranspose2d(32, 32, (3, 3), stride=(1, 1))
        self.deconv4 = nn.ConvTranspose2d(32, 6, (3, 3), stride=(2, 2), output_padding=(1, 1))
        self.batch4 = nn.BatchNorm2d(32)

    def sampling(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)  # return z sample

    def decoder(self, z):
        z = F.elu(self.fcz(z))
        z = self.unflatten(z)
        z = self.batch3(z)
        z = F.elu(self.deconv1(z))
        z = F.elu(self.deconv2(z))
        z = self.batch4(z)
        z = F.elu(self.deconv3(z))
        x = F.sigmoid(self.deconv4(z))
        return x

    def forward(self, x):
        return self.decoder(x)

class StochasticDecoder(nn.Module):
    def __init__(self, z_dim=20):
        super(StochasticDecoder, self).__init__()

        # decoder part
        out_dim = OUT_DIM[4]
        self.fcz = nn.Linear(z_dim, 32 * out_dim * out_dim)
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(32, out_dim, out_dim))
        self.deconv1 = nn.ConvTranspose2d(32, 32, (3, 3), stride=(1, 1))
        self.deconv2 = nn.ConvTranspose2d(32, 32, (3, 3), stride=(1, 1))
        self.batch3 = nn.BatchNorm2d(32)
        self.deconv3 = nn.ConvTranspose2d(32, 32, (3, 3), stride=(1, 1))
        self.deconv4 = nn.ConvTranspose2d(32, 6, (3, 3), stride=(2, 2), output_padding=(1, 1))
        self.batch4 = nn.BatchNorm2d(32)

    def sampling(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)  # return z sample

    def decoder(self, z):
        z = F.elu(self.fcz(z))
        z = self.unflatten(z)
        z = self.batch3(z)
        z = F.elu(self.deconv1(z))
        z = F.elu(self.deconv2(z))
        z = self.batch4(z)
        z = F.elu(self.deconv3(z))
        #mu = F.sigmoid(self.deconv4(z))
        mu = self.deconv4(z)
        std = torch.ones_like(mu).detach()
        return mu, std

    def forward(self, x):
        return self.decoder(x)

class StochasticVAE(nn.Module):
    def __init__(self, z_dim, h_dim, a_dim, fixed_std):
        super(StochasticVAE, self).__init__()

        self.encoder = EncoderVAE(z_dim)
        self.decoder = StochasticDecoder(z_dim)

        self.fwd_model = ForwardModelVAE(z_dim, h_dim, a_dim, fixed_std=fixed_std)

    def sampling(self, mu, std):
        std = torch.exp(0.5 * torch.log(torch.square(std)))
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)  # return z sample

    def forward(self, x, a, x_next):
        mu, std, _ = self.encoder(x)
        mu_target, std_target, _ = self.encoder(x_next)
        z = self.sampling(mu, std)
        z_target = self.sampling(mu_target, std_target)
        mu_next, std_next = self.fwd_model(z, a)
        z_next = self.sampling(mu_next, std_next)
        mu_x, std_x = self.decoder(z)
        return mu_x, std_x, mu, std, z, mu_target, std_target, z_target, mu_next, std_next, z_next

    def predict_dynamics(self, x, a):

        n_samples = 1

        _, _, z = self.encoder(x)

        mu_next, std_next = self.fwd_model(z, a)
        z_next = self.sampling(mu_next, std_next)

        mu_x, _ = self.decoder(z)

        mu_x_2, _ = self.decoder(z_next)

        return mu_x, mu_x_2


    def predict_trajectory(self, x, a):

        n_samples = 1

        mu, std, z = self.encoder(x)
        mu_fwd, std_next = self.fwd_model(z, a)
        z_fwd = self.sampling(mu_fwd, std_next)

        lower, upper = mu - std, mu + std
        lower_fwd, upper_fwd = mu_fwd - std, mu_fwd + std

        mu_x, _ = self.decoder(z)

        mu_x_2, _ = self.decoder(z_fwd)


        return mu, mu_fwd, lower, upper, lower_fwd, upper_fwd, mu_x, mu_x, z

    def predict_latent_dynamics(self, z, a):


        n_samples = 1
        # predicted distribution

        mu_fwd, std_fwd = self.fwd_model(z, a)
        z_fwd = self.sampling(mu_fwd, std_fwd)

        lower, upper = mu_fwd - std_fwd, mu_fwd + std_fwd



        mu_x, var_x = self.decoder(z_fwd)

        return mu_x, mu_fwd, lower, upper, z_fwd

class VAE(nn.Module):
    def __init__(self, z_dim, h_dim, a_dim):
        super(VAE, self).__init__()

        self.encoder = EncoderVAE(z_dim)
        self.decoder = Decoder(z_dim)

        self.fwd_model = ForwardModelVAE(z_dim, h_dim, a_dim)

    def sampling(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, x, a, x_next):
        mu, std = self.encoder(x)
        mu_target, std_target = self.encoder(x_next)
        z = self.sampling(mu, torch.log(torch.square(std)))
        mu_next, std_next = self.fwd_model(z, a)
        return self.decoder(z), mu, std, z, mu_target, std_target, mu_next, std_next