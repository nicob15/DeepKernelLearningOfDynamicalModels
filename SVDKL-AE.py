import torch
import torch.utils

import os
from utils import load_pickle, add_gaussian_noise
from Replay_Buffer import ReplayBuffer
import gpytorch
import numpy as np

from Plotter import plot_results
from Models import DKLModel
from VariationalTraining import VariationalKL
from Trainer import train_DKL as train

from modified_gaussian_likelihood import GaussianLikelihood

import gc

from datetime import datetime
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--batch-size', type=int, default=50,
                    help='Batch size.')
parser.add_argument('--num-epochs', type=int, default=100,
                    help='Number of training epochs.')
parser.add_argument('--learning-rate', type=float, default=3e-4,
                    help='Learning rate.')
parser.add_argument('--learning-rate-gp', type=float, default=1e-2,
                    help='Learning rate GP.')
parser.add_argument('--learning-rate-gp-var', type=float, default=1e-2,
                    help='Learning rate GP variational.')
parser.add_argument('--learning-rate-gp-lik', type=float, default=1e-2,
                    help='Learning rate GP likelihood.')
parser.add_argument('--reg_coefficient', type=float, default=1e-2,
                    help='L2 regularization coefficient.')
parser.add_argument('--coefficient-recon-loss', type=float, default=1.0,
                    help='Coefficient reconstrustruction loss.')
parser.add_argument('--coefficient-fwd-kl-loss', type=float, default=1.0,
                    help='Coefficient KL-divergence forward loss.')
parser.add_argument('--grid-size', type=int, default=32,
                    help='Grid size variational inference GP.')

parser.add_argument('--training', default=True,
                    help='Train the models.')
parser.add_argument('--plotting', default=False,
                    help='Plot the results.')
parser.add_argument('--num-samples-plot', type=int, default=5,
                    help='Number of independent sampling from the distribution.')

parser.add_argument('--hidden-dim', type=int, default=512,
                    help='Number of hidden units in MLPs.')
parser.add_argument('--latent-state-dim', type=int, default=20,
                    help='Dimensionality of the latent state space.')
parser.add_argument('--action-dim', type=int, default=1,
                    help='Dimensionality of the action space.')
parser.add_argument('--state-dim', type=int, default=3,
                    help='Dimensionality of the true state space.')
parser.add_argument('--observation-dim-w', type=int, default=84,
                    help='Width of the input measurements (RGB images).')
parser.add_argument('--observation-dim-h', type=int, default=84,
                    help='Height of the input measurements (RGB images).')
parser.add_argument('--observation-channels', type=int, default=6,
                    help='Channels of the RGB images (3*2 frames).')

parser.add_argument('--measurement-noise-level', type=float, default=0.0,
                    help='Level of noise of the input measurements.')
parser.add_argument('--actuation-noise-level', type=float, default=0.0,
                    help='Level of noise of the input actions.')

parser.add_argument('--experiment', type=str, default='Pendulum',
                    help='Experiment.')
parser.add_argument('--model-type', type=str, default='DKL',
                    help='Model type.')
parser.add_argument('--training-dataset', type=str, default='pendulum-mid.pkl',
                    help='Training dataset.')
parser.add_argument('--testing-dataset', type=str, default='pendulum_test.pkl',
                    help='Testing dataset.')

parser.add_argument('--log-interval', type=int, default=10,
                    help='How many batches to wait before saving and plotting')

parser.add_argument('--seed', type=int, default=1,
                    help='Random seed (default: 1).')
parser.add_argument('--jitter', type=float, default=1e-1,
                    help='Cholesky jitter.')

args = parser.parse_args()

# for 84x84 inputs
OUT_DIM = {2: 39, 4: 35, 6: 31}

seed = args.seed
torch.manual_seed(seed)
np.random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print('CUDA available:', torch.cuda.is_available())
    torch.cuda.manual_seed(args.seed)

# training hyperparameters
batch_size = args.batch_size
max_epoch = args.num_epochs

training = args.training
plotting = args.plotting
num_samples_plot = args.num_samples_plot

# learning rate
lr = args.learning_rate
lr_gp = args.learning_rate_gp
lr_gp_var = args.learning_rate_gp_var
lr_gp_lik = args.learning_rate_gp_lik
reg_coef = args.reg_coefficient
k1 = args.coefficient_recon_loss
k2 = args.coefficient_fwd_kl_loss
grid_size = args.grid_size

# build model
latent_dim = args.latent_state_dim
act_dim = args.action_dim
state_dim = args.state_dim
obs_dim_1 = args.observation_dim_w
obs_dim_2 = args.observation_dim_h
obs_dim_3 = args.observation_channels
h_dim = args.hidden_dim


# noise level on observations
noise_level = args.measurement_noise_level

# noise level on dynamics (actions)
noise_level_act = args.actuation_noise_level

# experiment and model type
exp = args.experiment
mtype = args.model_type
training_dataset = args.training_dataset
testing_dataset = args.testing_dataset

log_interval = args.log_interval

jitter = args.jitter


def main(exp='Pendulum', mtype='DKL', noise_level=0.0, training_dataset='pendulum-mid.pkl',
         testing_dataset='pendulum_test.pkl'):
    # load data
    directory = os.path.dirname(os.path.abspath(__file__))

    folder = os.path.join(directory + '/Data', training_dataset)
    folder_test = os.path.join(directory + '/Data', testing_dataset)

    data = load_pickle(folder)
    data_test = load_pickle(folder_test)

    model = DKLModel(num_dim=latent_dim, a_dim=act_dim, h_dim=h_dim, exp=exp, grid_size=grid_size)

    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(latent_dim)
    likelihood_fwd = gpytorch.likelihoods.MultitaskGaussianLikelihood(latent_dim)

    #likelihood = GaussianLikelihood()
    #likelihood_fwd = GaussianLikelihood()

    variational_kl_term = VariationalKL(likelihood, model.gp_layer, num_data=len(data))
    variational_kl_term_fwd = VariationalKL(likelihood_fwd, model.fwd_model_DKL.gp_layer_2, num_data=len(data))

    # If you run this example without CUDA, I hope you like waiting!
    if torch.cuda.is_available():
        model = model.cuda()
        likelihood = likelihood.cuda()
        likelihood_fwd = likelihood_fwd.cuda()
        variational_kl_term = variational_kl_term.cuda()
        variational_kl_term_fwd = variational_kl_term_fwd.cuda()

        # NOTE: Critical to avoid GPU leak
        gc.collect()

        # Use the adam optimizer
    optimizer = torch.optim.Adam([
        {'params': model.encoder.parameters()},
        {'params': model.decoder.parameters()},
        {'params': model.fwd_model_DKL.fwd_model.parameters()},
        {'params': model.gp_layer.hyperparameters(), 'lr': lr_gp},
        {'params': model.gp_layer.variational_parameters(), 'lr': lr_gp_var},
        {'params': model.fwd_model_DKL.gp_layer_2.hyperparameters(), 'lr': lr_gp},
        {'params': model.fwd_model_DKL.gp_layer_2.variational_parameters(), 'lr': lr_gp_var},
        {'params': likelihood.parameters(), 'lr': lr_gp_lik},
        {'params': likelihood_fwd.parameters(), 'lr': lr_gp_lik},
        ], lr=lr, weight_decay=reg_coef)

    counter = 0
    train_loader = ReplayBuffer(obs_dim=(obs_dim_1, obs_dim_2, 6), act_dim=act_dim, size=len(data), state_dim=state_dim)
    for d in data:
        train_loader.store(add_gaussian_noise(d[0]/255, noise_level=noise_level, clip=True).astype('float32'),
                           add_gaussian_noise(d[1], noise_level=noise_level_act).astype('float32'),
                           d[2].astype('float32'),
                           add_gaussian_noise(d[3]/255, noise_level=noise_level, clip=True).astype('float32'),
                           d[4], d[5].astype('float32'))
        counter += 1

    print(counter)

    test_loader = ReplayBuffer(obs_dim=(obs_dim_1, obs_dim_2, obs_dim_3), act_dim=act_dim, size=len(data_test),
                               state_dim=state_dim)
    for dt in data_test:
        test_loader.store(add_gaussian_noise(dt[0] / 255, noise_level=noise_level, clip=True).astype('float32'),
                          add_gaussian_noise(dt[1], noise_level=noise_level_act).astype('float32'),
                          dt[2].astype('float32'),
                          add_gaussian_noise(dt[3] / 255, noise_level=noise_level, clip=True).astype('float32'),
                          dt[4], dt[5].astype('float32'))

    now = datetime.now()
    date_string = now.strftime("%d-%m-%Y_%Hh-%Mm-%Ss")
    if training:
        for epoch in range(1, max_epoch):
            with gpytorch.settings.cholesky_jitter(jitter):
                train(epoch, batch_size, counter, train_loader, model, likelihood, likelihood_fwd,
                    optimizer, max_epoch, variational_kl_term, variational_kl_term_fwd, k1, k2)

            if epoch % log_interval == 0:

                torch.save({'model': model.state_dict(), 'likelihood': likelihood.state_dict(),
                            'likelihood_fwd': likelihood_fwd.state_dict()}, './DKL_Model_'+ date_string+'.pth')
                #if plotting:
                #    model.eval()
                #    likelihood.eval()
                #    likelihood_fwd.eval()
                #    with torch.no_grad(), gpytorch.settings.use_toeplitz(False), gpytorch.settings.fast_pred_var(), \
                #            gpytorch.settings.cholesky_jitter(1e-1):
                #        plot_results(model=model, likelihood=likelihood, likelihood_fwd=likelihood_fwd,
                #                     test_loader=test_loader, exp=exp, mtype=mtype,latent_dim=latent_dim,
                #                     noise_level=noise_level, state_dim=state_dim, obs_dim_1=obs_dim_1,
                #                     obs_dim_2=obs_dim_2, num_samples_plot=num_samples_plot, batch_size=batch_size)


    torch.save({'model': model.state_dict(), 'likelihood': likelihood.state_dict(),
                'likelihood_fwd': likelihood_fwd.state_dict()}, './DKL_Model_'+ date_string+'.pth')

    model.eval()
    likelihood.eval()
    likelihood_fwd.eval()
    with torch.no_grad(), gpytorch.settings.use_toeplitz(False), gpytorch.settings.fast_pred_var(), \
            gpytorch.settings.cholesky_jitter(jitter):
        plot_results(model=model, likelihood=likelihood, likelihood_fwd=likelihood_fwd,
                     test_loader=test_loader, exp=exp, mtype=mtype, latent_dim=latent_dim,
                     noise_level=noise_level, state_dim=state_dim, obs_dim_1=obs_dim_1,
                     obs_dim_2=obs_dim_2, num_samples_plot=num_samples_plot, batch_size=batch_size)

if __name__ == "__main__":

    with gpytorch.settings.use_toeplitz(False):#, gpytorch.settings.fast_pred_var():
        main(exp=exp, mtype=mtype, noise_level=noise_level, training_dataset=training_dataset,
             testing_dataset=testing_dataset)
    print('Finished Training the Representation Model!')
