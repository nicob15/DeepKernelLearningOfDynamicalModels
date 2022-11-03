import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np
import torch
from torchvision.utils import save_image
import os
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.preprocessing import Normalizer

def compute_PCA(input, dim=2):
    pca = PCA(n_components=dim)
    return pca.fit_transform(input)

def saveMultipage(filename, figs=None, dpi=100):
    pp = PdfPages(filename)
    if figs is None:
        figs = [plt.figure(n) for n in plt.get_fignums()]
    for fig in figs:
        fig.savefig(pp, format='pdf')
    pp.close()

def normalize(x):
    transformer = Normalizer().fit(x)
    return transformer.transform(x)

def closeAll():
    plt.close('all')

def plot_reconstruction(obs, obs_rec, save_dir, mtype, nr_samples=24, obs_dim_1=84, obs_dim_2=84):

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_dir1 = save_dir + '_obs_' + str(mtype)
    save_dir2 = save_dir + '_obs_rec' + str(mtype)

    obs = obs[:nr_samples, 3:6, :, :]
    obs_rec = obs_rec[:nr_samples, 3:6, :, :]

    save_image(tensor=obs.view(nr_samples, 3, obs_dim_1, obs_dim_2), fp=save_dir1 + '.png')
    save_image(tensor=obs_rec.view(nr_samples, 3, obs_dim_1, obs_dim_2), fp=save_dir2 + '.png')


def plot_results(model, likelihood, likelihood_fwd, test_loader, mtype, save_dir, PCA=False, obs_dim_1=84, obs_dim_2=84,
                 num_samples_plot=100):

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    data = test_loader.sample_batch(batch_size=num_samples_plot)
    obs = torch.from_numpy(data['obs1']).permute(0, 3, 1, 2).cuda()
    act = torch.from_numpy(data['acts']).cuda()
    next_obs = torch.from_numpy(data['obs2']).permute(0, 3, 1, 2).cuda()
    s = torch.from_numpy(data['states'])

    if mtype == 'DKL':
        mu_x, _, mu, _, z, res, _, _, _, mu_fwd, _, res_fwd = model(obs, act, next_obs)
        z = likelihood(res).sample(sample_shape=torch.Size([1])).mean(0)
        z_next = likelihood_fwd(res_fwd).sample(sample_shape=torch.Size([1])).mean(0)
        plot_reconstruction(obs, mu_x, save_dir, mtype=mtype, obs_dim_1=obs_dim_1, obs_dim_2=obs_dim_2)

    if mtype == 'VAE':
        mu_x, _, mu, _, z, _, _, _, mu_fwd, _, z_next = model(obs, act, next_obs)
        plot_reconstruction(obs, mu_x, save_dir, mtype=mtype, obs_dim_1=obs_dim_1, obs_dim_2=obs_dim_2)

    z = z.cpu().numpy()
    mu = mu.cpu().numpy()
    z_next = z_next.cpu().numpy()
    mu_fwd = mu_fwd.cpu().numpy()
    if PCA:
        z_2d = compute_PCA(z, 2)
        z = compute_PCA(z, 3)
        mu_2d = compute_PCA(mu, 2)
        mu = compute_PCA(mu, 3)
        z_next_2d = compute_PCA(z_next, 2)
        z_next = compute_PCA(z_next, 3)
        mu_fwd_2d = compute_PCA(mu_fwd, 2)
        mu_fwd = compute_PCA(mu_fwd, 3)
    else:
        z_2d = TSNE(n_components=2, learning_rate='auto').fit_transform(z)
        z = TSNE(n_components=3, learning_rate='auto').fit_transform(z)
        mu_2d = TSNE(n_components=2, learning_rate='auto').fit_transform(mu)
        mu = TSNE(n_components=2, learning_rate='auto').fit_transform(mu)
        z_next_2d = TSNE(n_components=2, learning_rate='auto').fit_transform(z_next)
        z_next = TSNE(n_components=2, learning_rate='auto').fit_transform(z_next)
        mu_fwd_2d = TSNE(n_components=2, learning_rate='auto').fit_transform(mu_fwd)
        mu_fwd = TSNE(n_components=2, learning_rate='auto').fit_transform(mu_fwd)

    angle = np.arctan2(s[:, 1], s[:, 0])

    fig = plt.figure(dpi=300)
    ax = fig.add_subplot()
    p1 = ax.scatter(z_2d[:, 0], z_2d[:, 1], s=15, c=angle, cmap='hsv')
    ax.legend([p1], ['$z_{t}$ ~ p($z_{t}$|$x_{t}$)'])
    cbar = fig.colorbar(p1)
    cbar.set_label('angle', rotation=90)
    plt.savefig(save_dir + 'latent_states_2d.png')
    plt.close()

    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(projection='3d')
    p1 = ax.scatter(z[:, 0], z[:, 1], z[:, 2], s=15, c=angle, cmap='hsv')
    ax.legend([p1], ['$z_{t}$ ~ p($z_{t}$|$x_{t}$)'])
    cbar = fig.colorbar(p1)
    cbar.set_label('angle', rotation=90)
    plt.savefig(save_dir + 'latent_states.png')
    plt.close()

    fig = plt.figure(dpi=300)
    ax = fig.add_subplot()
    p1 = ax.scatter(z_next_2d[:, 0], z_next_2d[:, 1], s=15, c=angle, cmap='hsv')
    ax.legend([p1], ["$z_{t+1}$ ~ p($z_{t+1}$|$z_{t}$,$u_{t}$)"])
    cbar = fig.colorbar(p1)
    cbar.set_label('angle', rotation=90)
    plt.savefig(save_dir + 'latent_next_states_2d.png')
    plt.close()

    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(projection='3d')
    p1 = ax.scatter(z_next[:, 0], z_next[:, 1], z_next[:, 2], s=15, c=angle, cmap='hsv')
    ax.legend([p1], ["$z_{t+1}$ ~ p($z_{t+1}$|$z_{t}$,$u_{t}$)"])
    cbar = fig.colorbar(p1)
    cbar.set_label('angle', rotation=90)
    plt.savefig(save_dir + 'latent_next_states.png')
    plt.close()

    fig = plt.figure(dpi=300)
    ax = fig.add_subplot()
    p1 = ax.scatter(mu_2d[:, 0], mu_2d[:, 1], s=15, c=angle, cmap='hsv')
    ax.legend([p1], ['mean of p($z_{t}$|$x_{t}$)'])
    cbar = fig.colorbar(p1)
    cbar.set_label('angle', rotation=90)
    plt.savefig(save_dir + 'latent_states_2d_mean.png')
    plt.close()

    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(projection='3d')
    p1 = ax.scatter(mu[:, 0], mu[:, 1], mu[:, 2], s=15, c=angle, cmap='hsv')
    ax.legend([p1], ['mean of p($z_{t}$|$x_{t}$)'])
    cbar = fig.colorbar(p1)
    cbar.set_label('angle', rotation=90)
    plt.savefig(save_dir + 'latent_states_mean.png')
    plt.close()

    fig = plt.figure(dpi=300)
    ax = fig.add_subplot()
    p1 = ax.scatter(mu_fwd_2d[:, 0], mu_fwd_2d[:, 1], s=15, c=angle, cmap='hsv')
    ax.legend([p1], ['mean of p($z_{t+1}$|$z_{t}$,$u_{t}$)'])
    cbar = fig.colorbar(p1)
    cbar.set_label('angle', rotation=90)
    plt.savefig(save_dir + 'latent_next_states_2d_mean.png')
    plt.close()

    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(projection='3d')
    p1 = ax.scatter(mu_fwd[:, 0], mu_fwd[:, 1], mu_fwd[:, 2], s=15, c=angle, cmap='hsv')
    ax.legend([p1], ['mean of p($z_{t+1}$|$z_{t}$,$u_{t}$)'])
    cbar = fig.colorbar(p1)
    cbar.set_label('angle', rotation=90)
    plt.savefig(save_dir + 'latent_next_states_mean.png')
    plt.close()


