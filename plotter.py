import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np
import torch
from torchvision.utils import save_image
import os
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.preprocessing import Normalizer
from matplotlib.patches import Ellipse
from matplotlib import cm
from matplotlib import animation
import gc
import matplotlib
matplotlib.use('Agg')

import matplotlib as mpl
mpl.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'

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

def save_frames_as_gif(frames, save_dir, name):

    #to change frame size
    plt.figure(figsize=(42, 42), dpi=42)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])
        gc.collect()

    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=50, save_count=len(frames),
                                   cache_frame_data=True)

    FFwriter = animation.FFMpegWriter(fps=10)
    anim.save(name, writer=FFwriter)

    plt.close()

def plot_2d(z_2d, angle, timestep, upper_2d, save_dir, name='latent_states_2d',
            plt_ellipse=False, legend='$z_{t}$ ~ p($z_{t}$|$x_{t}$)', s=1):
    fig = plt.figure(dpi=300)
    ax = fig.add_subplot()
    if plt_ellipse:
        colors = cm.get_cmap('cool', z_2d.shape[0])
        for i in range(z_2d.shape[0]):
            e = Ellipse((z_2d[i, 0], z_2d[i, 1]), s * upper_2d[i, 0], s * upper_2d[i, 1], alpha=0.1,
                        color=colors(int(timestep[i])), edgecolor=None)
            ax.add_artist(e)
    p1 = ax.scatter(z_2d[:, 0], z_2d[:, 1], s=15, c=timestep, cmap='cool')
    ax.legend([p1], [legend])
    cbar = fig.colorbar(p1)
    cbar.set_label('timestep', rotation=90)
    plt.savefig(save_dir + '/' + name + '_timestep.png')
    plt.close()

    fig = plt.figure(dpi=300)
    ax = fig.add_subplot()
    if plt_ellipse:
        colors = cm.get_cmap('hsv', 360)
        for i in range(z_2d.shape[0]):
            e = Ellipse((z_2d[i, 0], z_2d[i, 1]), s * upper_2d[i, 0], s * upper_2d[i, 1], alpha=0.1,
                        color=colors(int(100*angle[i])), edgecolor=None)
            ax.add_artist(e)
    p1 = ax.scatter(z_2d[:, 0], z_2d[:, 1], s=15, c=angle, cmap='hsv')
    ax.legend([p1], [legend])
    cbar = fig.colorbar(p1)
    cbar.set_label('angle', rotation=90)
    plt.savefig(save_dir + '/' + name + '_angle.png')
    plt.close()

def plot_3d(z_3d, angle, timestep, upper_3d, save_dir, name='latent_states_3d',
            plt_ellipse=False, legend='$z_{t}$ ~ p($z_{t}$|$x_{t}$)', s=1):

    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(projection='3d')
    if plt_ellipse:
        colors = cm.get_cmap('cool', z_3d.shape[0])
        # Set of all spherical angles:
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 20)
        for i in range(z_3d.shape[0]):
            # Cartesian coordinates that correspond to the spherical angles:
            # (this is the equation of an ellipsoid):
            x = s * upper_3d[i, 0] * np.outer(np.cos(u), np.sin(v)) + z_3d[i, 0]
            y = s * upper_3d[i, 1] * np.outer(np.sin(u), np.sin(v)) + z_3d[i, 1]
            z = s * upper_3d[i, 2] * np.outer(np.ones_like(u), np.cos(v)) + z_3d[i, 2]
            # Plot:
            ax.plot_surface(x, y, z, color=colors(int(timestep[i])), alpha=0.1, shade=False)  # , antialiased=False)
    p1 = ax.scatter(z_3d[:, 0], z_3d[:, 1], z_3d[:, 2], s=15, c=timestep, cmap='cool')
    ax.legend([p1], [legend])
    cbar = fig.colorbar(p1)
    cbar.set_label('timestep', rotation=90)
    plt.savefig(save_dir + '/' + name + '_timestep.png')
    plt.close()

    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(projection='3d')
    if plt_ellipse:
        colors = cm.get_cmap('hsv', 720)
        # Set of all spherical angles:
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 20)
        for i in range(z_3d.shape[0]):
            # Cartesian coordinates that correspond to the spherical angles:
            # (this is the equation of an ellipsoid):
            x = s * upper_3d[i, 0] * np.outer(np.cos(u), np.sin(v)) + z_3d[i, 0]
            y = s * upper_3d[i, 1] * np.outer(np.sin(u), np.sin(v)) + z_3d[i, 1]
            z = s * upper_3d[i, 2] * np.outer(np.ones_like(u), np.cos(v)) + z_3d[i, 2]
            # Plot:
            ax.plot_surface(x, y, z, color=colors(int(100*angle[i])), alpha=0.1, shade=False)  # , antialiased=False)
    p1 = ax.scatter(z_3d[:, 0], z_3d[:, 1], z_3d[:, 2], s=15, c=angle, cmap='hsv')
    ax.legend([p1], [legend])
    cbar = fig.colorbar(p1)
    cbar.set_label('angle', rotation=90)
    plt.savefig(save_dir + '/' + name + '_angle.png')
    plt.close()

def plot_reconstruction_sequences(test_loader, model, save_dir, mtype, step=50, obs_dim_1=120, obs_dim_2=120, latent_dim=50, make_gif=False):

    save_dir = save_dir + '/sequences/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_dir1 = save_dir + '/obs_' + str(mtype)
    save_dir2 = save_dir + '/obs_rec_' + str(mtype)
    save_dir3 = save_dir + '/obs_rec_mu_' + str(mtype)

    ep_len = 100
    nr_samples = 100
    for i in range(1):
        data = test_loader.sample_sequence(start_idx=i+403, seq_len=ep_len)

        obs = torch.from_numpy(data['obs1']).permute(0, 3, 1, 2).cuda()
        act = torch.from_numpy(data['acts']).cuda()
        next_obs = torch.from_numpy(data['obs2']).permute(0, 3, 1, 2).cuda()

        s = data['states']
        angle = np.arctan2(s[:, 1], s[:, 0])
        timesteps = np.arange(0, nr_samples, 1)

        next_obs_rec = torch.zeros_like(obs)
        obs_rec = torch.zeros_like(obs)
        next_obs_rec_mu = torch.zeros_like(obs)
        z_next = torch.zeros((nr_samples, latent_dim)).cuda()
        mu_next = torch.zeros((nr_samples, latent_dim))
        mu_next_2 = torch.zeros((nr_samples, latent_dim)).cuda()
        upper = torch.zeros((nr_samples, latent_dim))
        upper_2 = torch.zeros((nr_samples, latent_dim))

        _, _, _, _, _, _, _, _, _, mu_next[0], _, res_next, z_next[0] = model(obs[0].unsqueeze(dim=0), act[0].unsqueeze(dim=0), next_obs[0].unsqueeze(dim=0))
        mu_next_2[0] = mu_next[0]
        _, upper[0] = model.fwd_model_DKL.likelihood(res_next).confidence_region()
        next_obs_rec[0], _ = model.AE_DKL.decoder(z_next[0].unsqueeze(dim=0))
        next_obs_rec_mu[0], _ = model.AE_DKL.decoder(mu_next_2[0].unsqueeze(dim=0))

        for j in range(ep_len-1):
            obs_rec[j], _, _, _, _, _, _, _, _, _, _, _, _ = model(obs[j].unsqueeze(dim=0), act[j].unsqueeze(dim=0),
                                                                   next_obs[j+1].unsqueeze(dim=0))

            next_obs_rec[j+1], z_next[j+1], mu_next[j+1], res_next = model.predict_dynamics(z_next[j].unsqueeze(dim=0), act[j+1].unsqueeze(dim=0), samples=10)
            _, upper[j+1] = model.fwd_model_DKL.likelihood(res_next).confidence_region()

            next_obs_rec_mu[j+1], _, mu_next_2[j+1], res_next = model.predict_dynamics_mean(mu_next_2[j].unsqueeze(dim=0), act[j+1].unsqueeze(dim=0))
            _, upper_2[j + 1] = model.fwd_model_DKL.likelihood(res_next).confidence_region()

        obs_rec = obs_rec[:nr_samples, 3:6, :, :]
        obs = obs[:nr_samples, 3:6, :, :]
        next_obs = next_obs[:nr_samples, 3:6, :, :]
        next_obs_rec = next_obs_rec[:nr_samples, 3:6, :, :]
        next_obs_rec_mu = next_obs_rec_mu[:nr_samples, 3:6, :, :]

        save_image(tensor=next_obs.view(nr_samples, 3, obs_dim_1, obs_dim_2),
                   fp=save_dir1 + '_sequence_' + str(i) +'_.png')
        save_image(tensor=next_obs_rec.view(nr_samples, 3, obs_dim_1, obs_dim_2),
                   fp=save_dir2 + '_sequence_' + str(i) +'_.png')
        save_image(tensor=next_obs_rec_mu.view(nr_samples, 3, obs_dim_1, obs_dim_2),
                   fp=save_dir3 + '_sequence_' + str(i) + '_.png')

        z_next = z_next.cpu().numpy()
        mu = mu_next.cpu().numpy()
        upper = upper.cpu().numpy()
        mu_2 = mu_next_2.cpu().numpy()
        upper_2 = upper_2.cpu().numpy()

        plot_traj_single_state_variables(z_next, None, save_dir, name='trajectory_over_time', plot_uq=False,
                                         name1='trajectory latent variable over time')
        plot_traj_single_state_variables(mu, upper, save_dir, name='trajectory_over_time_UQ', plot_uq=True,
                                         name1='trajectory latent variable over time with UQ')
        plot_traj_single_state_variables(angle.reshape(-1, 1), None, save_dir, name='true_angle_over_time',
                                         plot_uq=False, latent_var=False, name1='True angle over time')
        plot_traj_single_state_variables(s[:, 0].reshape(-1, 1), None, save_dir, name='true_x_over_time',
                                         plot_uq=False, latent_var=False, name1='True x position over time')
        plot_traj_single_state_variables(s[:, 1].reshape(-1, 1), None, save_dir, name='true_y_over_time',
                                         plot_uq=False, latent_var=False, name1='True y position over time')
        plot_traj_single_state_variables(s[:, 2].reshape(-1, 1), None, save_dir, name='true_vel_over_time',
                                         plot_uq=False, latent_var=False, name1='True velocity over time')

        z_next_2d = TSNE(n_components=2, learning_rate='auto').fit_transform(z_next)
        z_next_3d = TSNE(n_components=3, learning_rate='auto').fit_transform(z_next)
        mu_fwd_2d = TSNE(n_components=2, learning_rate='auto').fit_transform(mu)
        mu_fwd_3d = TSNE(n_components=3, learning_rate='auto').fit_transform(mu)
        upper_2d = TSNE(n_components=2, learning_rate='auto').fit_transform(upper / 2)
        upper_3d = TSNE(n_components=3, learning_rate='auto').fit_transform(upper / 2)

        upper_2d = upper_2d / 5
        upper_3d = upper_3d / 5

        plot_2d(z_next_2d, angle, timesteps, upper_2d, save_dir,
                name='latent_next_states_2d_' + str(i), plt_ellipse=False,
                legend='$z_{t+1}$ ~ p($z_{t+1}$|$z_{t}$)')
        plot_3d(z_next_3d, angle, timesteps, upper_3d, save_dir,
                name='latent_next_states_3d_' + str(i), plt_ellipse=False,
                legend='$z_{t+1}$ ~ p($z_{t+1}$|$z_{t}$)')
        plot_2d(mu_fwd_2d, angle, timesteps, upper_2d/10, save_dir,
                name='latent_next_states_2d_mean_' + str(i), plt_ellipse=True,
                legend='mean of p($z_{t+1}$|$z_{t}$)',
                s=5)
        plot_3d(mu_fwd_3d, angle, timesteps, upper_3d/20, save_dir,
                name='latent_next_states_3d_mean_' + str(i), plt_ellipse=True,
                legend='mean of p($z_{t+1}$|$z_{t}$)',
                s=15)

        mu_fwd_2d = TSNE(n_components=2, learning_rate='auto').fit_transform(mu_2)
        mu_fwd_3d = TSNE(n_components=3, learning_rate='auto').fit_transform(mu_2)
        upper_2d = TSNE(n_components=2, learning_rate='auto').fit_transform(upper_2 / 2)
        upper_3d = TSNE(n_components=3, learning_rate='auto').fit_transform(upper_2 / 2)

        upper_2d = upper_2d / 5
        upper_3d = upper_3d / 5

        plot_2d(mu_fwd_2d, angle, timesteps, upper_2d/10, save_dir,
                name='latent_next_states_2d_mean_over_time_' + str(i), plt_ellipse=True,
                legend='mean of p($z_{t+1}$|$z_{t}$)',
                s=5)
        plot_3d(mu_fwd_3d, angle, timesteps, upper_3d/20, save_dir,
                name='latent_next_states_3d_mean_over_time_' + str(i), plt_ellipse=True,
                legend='mean of p($z_{t+1}$|$z_{t}$)',
                s=15)

        if make_gif:
            print("make gif")
            save_frames_as_gif(obs.reshape(nr_samples, 3, obs_dim_1, obs_dim_2).permute(0, 2, 3, 1).cpu().numpy(),
                               save_dir=save_dir1 + '_obs_sequence_' + str(i) + '_.gif', name=save_dir1 + '_obs_sequence_' + str(i) + '_.mp4')
            save_frames_as_gif(obs_rec.reshape(nr_samples, 3, obs_dim_1, obs_dim_2).permute(0, 2, 3, 1).cpu().numpy(),
                               save_dir=save_dir1 + '_obs_rec_sequence_' + str(i) + '_.gif',
                               name=save_dir1 + '_obs_rec_sequence_' + str(i) + '_.mp4')
            save_frames_as_gif(next_obs.reshape(nr_samples, 3, obs_dim_1, obs_dim_2).permute(0, 2, 3, 1).cpu().numpy(),
                               save_dir=save_dir1 + '_sequence_' + str(i) + '_.gif', name=save_dir1 + '_sequence_' + str(i) + '_.mp4')
            save_frames_as_gif(next_obs_rec.reshape(nr_samples, 3, obs_dim_1, obs_dim_2).permute(0, 2, 3, 1).cpu().numpy(),
                               save_dir=save_dir2 + '_sequence_' + str(i) + '_.gif', name=save_dir2 + '_sequence_' + str(i) + '_.mp4')
            save_frames_as_gif(next_obs_rec_mu.reshape(nr_samples, 3, obs_dim_1, obs_dim_2).permute(0, 2, 3, 1).cpu().numpy(),
                               save_dir=save_dir3 + '_sequence_' + str(i) + '_.gif', name=save_dir3 + '_sequence_' + str(i) + '_.mp4')

def plot_traj_single_state_variables(z, std, save_dir, name, plot_uq=False, latent_var=True, name1=''):

    save_dir = save_dir + '/variables'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    timesteps = np.arange(0, z.shape[0], 1)
    for i in range(z.shape[1]):
        fig, ax = plt.subplots(1)
        fig.suptitle(name1)
        if latent_var:
            legend1 = str(i) + '-th latent state variable'
        else:
            legend1 = 'true state variable'
        if plot_uq:
            ax.fill_between(timesteps, z[:, i] - std[:, i] / 2, z[:, i] + std[:, i] / 2, facecolor='blue', alpha=0.5)
        p1 = ax.scatter(timesteps, z[:, i], c=timesteps, cmap='cool')
        ax.plot(timesteps, z[:, i])
        ax.legend([p1], [legend1])
        plt.xlabel('timestep')

        if plot_uq == False:
            plt.savefig(save_dir + '/' + name + str(i) + '_variable_timesteps.png')
        else:
            plt.savefig(save_dir + '/' + name + str(i) + '_UQ_variable_timesteps.png')

        plt.close()

def plot_reconstruction(obs, obs_rec, save_dir, mtype, nr_samples=80, obs_dim_1=84, obs_dim_2=84):

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_dir1 = save_dir + '/obs_' + str(mtype)
    save_dir2 = save_dir + '/obs_rec_' + str(mtype)

    obs = obs[:nr_samples, 3:6, :, :]
    obs_rec = obs_rec[:nr_samples, 3:6, :, :]

    save_image(tensor=obs.view(nr_samples, 3, obs_dim_1, obs_dim_2), fp=save_dir1 + '.png')
    save_image(tensor=obs_rec.view(nr_samples, 3, obs_dim_1, obs_dim_2), fp=save_dir2 + '.png')


def plot_results(model, test_loader, mtype, save_dir, PCA=False, obs_dim_1=84, obs_dim_2=84, num_samples_plot=100, latent_dim=50):

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    data = test_loader.sample_batch(batch_size=num_samples_plot)
    obs = torch.from_numpy(data['obs1']).permute(0, 3, 1, 2).cuda()
    act = torch.from_numpy(data['acts']).cuda()
    next_obs = torch.from_numpy(data['obs2']).permute(0, 3, 1, 2).cuda()
    s = torch.from_numpy(data['states'])

    if mtype == 'DKL':
        mu_x, _, mu, _, z, res, _, _, _, mu_fwd, _, res_fwd, z_next = model(obs, act, next_obs)
        plot_reconstruction(obs, mu_x, save_dir, mtype=mtype, obs_dim_1=obs_dim_1, obs_dim_2=obs_dim_2)

        plot_reconstruction_sequences(test_loader, model, save_dir, mtype, step=1, obs_dim_1=obs_dim_1, obs_dim_2=obs_dim_2, latent_dim=latent_dim)

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
        mu = TSNE(n_components=3, learning_rate='auto').fit_transform(mu)
        z_next_2d = TSNE(n_components=2, learning_rate='auto').fit_transform(z_next)
        z_next = TSNE(n_components=3, learning_rate='auto').fit_transform(z_next)
        mu_fwd_2d = TSNE(n_components=2, learning_rate='auto').fit_transform(mu_fwd)
        mu_fwd = TSNE(n_components=3, learning_rate='auto').fit_transform(mu_fwd)

    angle = np.arctan2(s[:, 1], s[:, 0])

    fig = plt.figure(dpi=300)
    ax = fig.add_subplot()
    p1 = ax.scatter(z_2d[:, 0], z_2d[:, 1], s=15, c=angle, cmap='hsv')
    ax.legend([p1], ['$z_{t}$ ~ p($z_{t}$|$x_{t}$)'])
    cbar = fig.colorbar(p1)
    cbar.set_label('angle', rotation=90)
    plt.savefig(save_dir + '/latent_states_2d.png')
    plt.close()

    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(projection='3d')
    p1 = ax.scatter(z[:, 0], z[:, 1], z[:, 2], s=15, c=angle, cmap='hsv')
    ax.legend([p1], ['$z_{t}$ ~ p($z_{t}$|$x_{t}$)'])
    cbar = fig.colorbar(p1)
    cbar.set_label('angle', rotation=90)
    plt.savefig(save_dir + '/latent_states.png')
    plt.close()

    fig = plt.figure(dpi=300)
    ax = fig.add_subplot()
    p1 = ax.scatter(z_next_2d[:, 0], z_next_2d[:, 1], s=15, c=angle, cmap='hsv')
    ax.legend([p1], ["$z_{t+1}$ ~ p($z_{t+1}$|$z_{t}$,$u_{t}$)"])
    cbar = fig.colorbar(p1)
    cbar.set_label('angle', rotation=90)
    plt.savefig(save_dir + '/latent_next_states_2d.png')
    plt.close()

    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(projection='3d')
    p1 = ax.scatter(z_next[:, 0], z_next[:, 1], z_next[:, 2], s=15, c=angle, cmap='hsv')
    ax.legend([p1], ["$z_{t+1}$ ~ p($z_{t+1}$|$z_{t}$,$u_{t}$)"])
    cbar = fig.colorbar(p1)
    cbar.set_label('angle', rotation=90)
    plt.savefig(save_dir + '/latent_next_states.png')
    plt.close()

    fig = plt.figure(dpi=300)
    ax = fig.add_subplot()
    p1 = ax.scatter(mu_2d[:, 0], mu_2d[:, 1], s=15, c=angle, cmap='hsv')
    ax.legend([p1], ['mean of p($z_{t}$|$x_{t}$)'])
    cbar = fig.colorbar(p1)
    cbar.set_label('angle', rotation=90)
    plt.savefig(save_dir + '/latent_states_2d_mean.png')
    plt.close()

    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(projection='3d')
    p1 = ax.scatter(mu[:, 0], mu[:, 1], mu[:, 2], s=15, c=angle, cmap='hsv')
    ax.legend([p1], ['mean of p($z_{t}$|$x_{t}$)'])
    cbar = fig.colorbar(p1)
    cbar.set_label('angle', rotation=90)
    plt.savefig(save_dir + '/latent_states_mean.png')
    plt.close()

    fig = plt.figure(dpi=300)
    ax = fig.add_subplot()
    p1 = ax.scatter(mu_fwd_2d[:, 0], mu_fwd_2d[:, 1], s=15, c=angle, cmap='hsv')
    ax.legend([p1], ['mean of p($z_{t+1}$|$z_{t}$,$u_{t}$)'])
    cbar = fig.colorbar(p1)
    cbar.set_label('angle', rotation=90)
    plt.savefig(save_dir + '/latent_next_states_2d_mean.png')
    plt.close()

    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(projection='3d')
    p1 = ax.scatter(mu_fwd[:, 0], mu_fwd[:, 1], mu_fwd[:, 2], s=15, c=angle, cmap='hsv')
    ax.legend([p1], ['mean of p($z_{t+1}$|$z_{t}$,$u_{t}$)'])
    cbar = fig.colorbar(p1)
    cbar.set_label('angle', rotation=90)
    plt.savefig(save_dir + '/latent_next_states_mean.png')
    plt.close()


