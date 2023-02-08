import torch
import torch.utils

import os
from utils import load_pickle, add_gaussian_noise
from replay_buffer import ReplayBuffer
import gpytorch
import numpy as np

from plotter import plot_results
from models import SVDKL_AE_latent_dyn
from variational_inference import VariationalKL
from trainer import train_DKL as train

import gc

from datetime import datetime
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--batch-size', type=int, default=50,
                    help='Batch size.')
parser.add_argument('--num-epochs', type=int, default=100,
                    help='Number of training epochs.')
parser.add_argument('--learning-rate', type=float, default=1e-4,
                    help='Learning rate.')
parser.add_argument('--learning-rate-gp', type=float, default=1e-2,
                    help='Learning rate GP.')
parser.add_argument('--learning-rate-gp-var', type=float, default=1e-2,
                    help='Learning rate GP variational.')
parser.add_argument('--learning-rate-gp-lik', type=float, default=1e-2,
                    help='Learning rate GP likelihood.')
parser.add_argument('--reg-coefficient', type=float, default=1e-2,
                    help='L2 regularization coefficient.')
parser.add_argument('--coefficient-recon-loss', type=float, default=1.0,
                    help='Coefficient reconstrustruction loss.')
parser.add_argument('--coefficient-fwd-kl-loss', type=float, default=1.0,
                    help='Coefficient KL-divergence forward loss.')
parser.add_argument('--grid-size', type=int, default=32,
                    help='Grid size variational inference GP.')

parser.add_argument('--training', default=True,
                    help='Train the models.')
parser.add_argument('--plotting', default=True,
                    help='Plot the results.')
parser.add_argument('--num-samples-plot', type=int, default=200,
                    help='Number of independent sampling from the distribution.')

parser.add_argument('--hidden-dim', type=int, default=256,
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
parser.add_argument('--training-dataset', type=str, default='pendulum_train.pkl',
                    help='Training dataset.')
parser.add_argument('--testing-dataset', type=str, default='pendulum_test.pkl',
                    help='Testing dataset.')

parser.add_argument('--log-interval', type=int, default=50,
                    help='How many batches to wait before saving and plotting')

parser.add_argument('--seed', type=int, default=1,
                    help='Random seed (default: 1).')
parser.add_argument('--jitter', type=float, default=1e-8,
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


def main(exp='Pendulum', mtype='DKL', noise_level=0.0, training_dataset='pendulum_train.pkl',
         testing_dataset='pendulum_test.pkl'):

    # load data
    directory = os.path.dirname(os.path.abspath(__file__))

    folder = os.path.join(directory + '/Data', training_dataset)
    folder_test = os.path.join(directory + '/Data', testing_dataset)

    data = load_pickle(folder)
    data_test = load_pickle(folder_test)

    # likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(latent_dim, rank=0, has_task_noise=False,
    #                                                               has_global_noise=True)
    # likelihood_fwd = gpytorch.likelihoods.MultitaskGaussianLikelihood(latent_dim, rank=0, has_task_noise=False,
    #                                                                   has_global_noise=True)

    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(latent_dim, rank=0, has_task_noise=True,
                                                                  has_global_noise=False)
    likelihood_fwd = gpytorch.likelihoods.MultitaskGaussianLikelihood(latent_dim, rank=0, has_task_noise=True,
                                                                      has_global_noise=False)

    model = SVDKL_AE_latent_dyn(num_dim=latent_dim, a_dim=act_dim, h_dim=h_dim, grid_size=grid_size, lik=likelihood,
                                lik_fwd=likelihood_fwd, grid_bounds=(-10.0, 10.0))

    variational_kl_term = VariationalKL(model.AE_DKL.likelihood, model.AE_DKL.gp_layer, num_data=batch_size) #len(data)
    variational_kl_term_fwd = VariationalKL(model.fwd_model_DKL.likelihood, model.fwd_model_DKL.gp_layer_2, num_data=batch_size)

    # If you run this example without CUDA, I hope you like waiting!
    if torch.cuda.is_available():
        model = model.cuda()
        variational_kl_term = variational_kl_term.cuda()
        variational_kl_term_fwd = variational_kl_term_fwd.cuda()

        # NOTE: Critical to avoid GPU leak
        gc.collect()

    # Use the adam optimizer
    optimizer = torch.optim.Adam([
        {'params': model.AE_DKL.encoder.parameters()},
        {'params': model.AE_DKL.decoder.parameters()},
        {'params': model.fwd_model_DKL.fwd_model.parameters()},
        {'params': model.AE_DKL.gp_layer.hyperparameters(), 'lr': lr_gp},
        #{'params': model.AE_DKL.gp_layer.variational_parameters(), 'lr': lr_gp_var},
        {'params': model.fwd_model_DKL.gp_layer_2.hyperparameters(), 'lr': lr_gp},
        #{'params': model.fwd_model_DKL.gp_layer_2.variational_parameters(), 'lr': lr_gp_var},
        {'params': model.AE_DKL.likelihood.parameters(), 'lr': lr_gp_lik},
        {'params': model.fwd_model_DKL.likelihood.parameters(), 'lr': lr_gp_lik},
        ], lr=lr, weight_decay=reg_coef)

    optimizer_var1 = torch.optim.SGD([
        {'params': model.AE_DKL.gp_layer.variational_parameters(), 'lr': lr_gp_var},
        ], lr=lr_gp_var, momentum=0.9, nesterov=True, weight_decay=0) # momentum=0.9, nesterov=True, weight_decay=0
    optimizer_var2 = torch.optim.SGD([
        {'params': model.fwd_model_DKL.gp_layer_2.variational_parameters(), 'lr': lr_gp_var},
        ], lr=lr_gp_var, momentum=0.9, nesterov=True, weight_decay=0)
    scheduler_1 = torch.optim.lr_scheduler.MultiStepLR(optimizer_var1, milestones=[0.5 * max_epoch, 0.75 * max_epoch],
                                                       gamma=0.1)
    scheduler_2 = torch.optim.lr_scheduler.MultiStepLR(optimizer_var2, milestones=[0.5 * max_epoch, 0.75 * max_epoch],
                                                       gamma=0.1)

    optimizer = [optimizer, optimizer_var1, optimizer_var2]

    counter = 0
    train_loader = ReplayBuffer(obs_dim=(obs_dim_1, obs_dim_2, 6), act_dim=act_dim, size=len(data), state_dim=state_dim)
    for d in data[:15000]:
        train_loader.store(add_gaussian_noise(d[0] / 255, noise_level=noise_level, clip=True).astype('float32'),
                           add_gaussian_noise(d[1], noise_level=noise_level_act).astype('float32'),
                           add_gaussian_noise(d[3] / 255, noise_level=noise_level, clip=True).astype('float32'),
                           d[4],
                           d[5].astype('float32'))
        counter += 1

    print(counter)

    test_loader = ReplayBuffer(obs_dim=(obs_dim_1, obs_dim_2, obs_dim_3), act_dim=act_dim, size=len(data_test),
                               state_dim=state_dim)
    counter_t = 0
    for dt in data_test:
        test_loader.store(add_gaussian_noise(dt[0] / 255, noise_level=noise_level, clip=True).astype('float32'),
                          add_gaussian_noise(dt[1], noise_level=noise_level_act).astype('float32'),
                          add_gaussian_noise(dt[3] / 255, noise_level=noise_level, clip=True).astype('float32'),
                          dt[4],
                          dt[5].astype('float32'))
        counter_t += 1
    print(counter_t)

    now = datetime.now()
    date_string = now.strftime("%d-%m-%Y_%Hh-%Mm-%Ss")
    save_pth_dir = directory + '/Results/' + str(exp) + '/' + str(mtype) + '/Noise_level_' + str(noise_level)
    if not os.path.exists(save_pth_dir):
        os.makedirs(save_pth_dir)

    training = False
    if training:
        for epoch in range(1, max_epoch):
            with gpytorch.settings.cholesky_jitter(jitter):
                train(epoch=epoch, batch_size=batch_size, nr_data=counter, train_loader=train_loader, model=model,
                      optimizers=optimizer, variational_kl_term=variational_kl_term,
                      variational_kl_term_fwd=variational_kl_term_fwd, k1=k1, beta=k2)

                scheduler_1.step()
                scheduler_2.step()

            if epoch % log_interval == 0:
                torch.save({'model': model.state_dict(), 'likelihood': model.AE_DKL.likelihood.state_dict(),
                            'likelihood_fwd': model.fwd_model_DKL.likelihood.state_dict()}, save_pth_dir +'/DKL_Model_' + date_string+'.pth')

        torch.save({'model': model.state_dict(), 'likelihood': model.AE_DKL.likelihood.state_dict(),
                    'likelihood_fwd': model.fwd_model_DKL.likelihood.state_dict()}, save_pth_dir + '/DKL_Model_' + date_string + '.pth')

    # checkpoint = torch.load(save_pth_dir + '/DKL_Model_07-02-2023_20h-02m-31s.pth')
    # model.load_state_dict(checkpoint['model'])
    # model.AE_DKL.likelihood.load_state_dict(checkpoint['likelihood'])
    # model.fwd_model_DKL.likelihood.load_state_dict(checkpoint['likelihood_fwd'])
    if plotting:
        model.eval()
        model.AE_DKL.likelihood.eval()
        model.fwd_model_DKL.likelihood.eval()
        del train_loader
        with torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.settings.cholesky_jitter(jitter):
            plot_results(model=model, test_loader=test_loader, mtype=mtype, save_dir=save_pth_dir, PCA=False,
                         obs_dim_1=obs_dim_1, obs_dim_2=obs_dim_2, num_samples_plot=num_samples_plot, latent_dim=latent_dim)

if __name__ == "__main__":

    with gpytorch.settings.use_toeplitz(False):
        main(exp=exp, mtype=mtype, noise_level=noise_level, training_dataset=training_dataset,
             testing_dataset=testing_dataset)
    print('Finished Training the Representation Model!')
