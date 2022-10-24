import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np

import torch
from torchvision.utils import save_image, make_grid
import gpytorch
import torch.distributions as td
import os
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.preprocessing import Normalizer

directory = os.path.dirname(os.path.abspath(__file__))

results_dir = directory + '/Figures'

if not os.path.exists(results_dir):
    os.makedirs(results_dir)


def compute_PCA(input, dim=2):
    pca = PCA(n_components=dim)
    return pca.fit_transform(input)

def compute_PCA_2(dim=1):
    pca = PCA(n_components=dim)
    return pca


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

def plot_rewards(mean, std, name='Evaluation Reward', folder='Figures/', exp='Pendulum', mtype='DKL'):

    closeAll()

    timesteps = np.array(range(len(mean)))
    mean = np.array(mean)
    std = np.array(std)

    #fig, ax = plt.subplots(1)
    plt.plot(timesteps, mean, linewidth=1, color='blue')
    plt.fill_between(timesteps, mean + std, mean - std, facecolor='blue', alpha=0.5)
    plt.grid()
    plt.legend(['Eval Rew'])

    plt.title(name, {'fontsize': 15})
    #ax.ylim((0.5, 100))
    plt.xlabel('Epoch')
    plt.ylabel('Eval Rew')
    plt.pause(0.1)

    saveMultipage(os.path.join(folder + exp + '/' + mtype + '/', "Evaluation_Reward.pdf"))

    closeAll()

def make_UQ_single_traj(model, likelihood, likelihood_fwd, test_loader, exp, mtype, noise_level=0.0,
                                           obs_dim_1=84, obs_dim_2=84, latent_dim=20):
    save_dir = results_dir + '/' + str(exp) + '/' + str(mtype) + '/Noise_level_' + str(
        noise_level) + '/UQ_trajectory/'

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    dir1 = '/z/'
    if not os.path.exists(save_dir+dir1):
        os.makedirs(save_dir+dir1)

    dir2 = '/next_z/'
    if not os.path.exists(save_dir+dir2):
        os.makedirs(save_dir+dir2)

    dir3 = '/z_pred/'
    if not os.path.exists(save_dir+dir3):
        os.makedirs(save_dir+dir3)

    dir4 = '/next_z_pred/'
    if not os.path.exists(save_dir+dir4):
        os.makedirs(save_dir+dir4)

    dir5 = '/next_z_pred_latent/'
    if not os.path.exists(save_dir+dir5):
        os.makedirs(save_dir+dir5)


    start_idx = [1]
    seq_len = 48#16
    for j in range(len(start_idx)):
        data = test_loader.sample_sequence(start_idx=start_idx[j], seq_len=seq_len)
        obs = torch.from_numpy(data['obs1']).permute(0, 3, 1, 2)
        act = torch.from_numpy(data['acts'])
        # obs2 = torch.from_numpy(data['obs2']).permute(0, 3, 1, 2)
        mu = torch.zeros(seq_len, latent_dim)
        mu_fwd = torch.zeros(seq_len, latent_dim)
        mu_pred = torch.zeros(seq_len, latent_dim)
        mu_fwd_pred = torch.zeros(seq_len, latent_dim)
        mu_fwd_pred_2 = torch.zeros(seq_len, latent_dim)
        lower = torch.zeros(seq_len, latent_dim)
        upper = torch.zeros(seq_len, latent_dim)
        lower_fwd = torch.zeros(seq_len, latent_dim)
        upper_fwd = torch.zeros(seq_len, latent_dim)
        lower_pred = torch.zeros(seq_len, latent_dim)
        upper_pred = torch.zeros(seq_len, latent_dim)
        lower_fwd_pred = torch.zeros(seq_len, latent_dim)
        upper_fwd_pred = torch.zeros(seq_len, latent_dim)
        lower_fwd_pred_2 = torch.zeros(seq_len, latent_dim)
        upper_fwd_pred_2 = torch.zeros(seq_len, latent_dim)

        if mtype == 'DKL':
            mu[0], mu_fwd[0], lower[0], upper[0], lower_fwd[0], upper_fwd[0], mu_x, _, z = model.predict_trajectory(
                x=obs[0].view(1, 6, obs_dim_1, obs_dim_2).cuda(),
                a=act[0].view(1, 1).cuda(),
                likelihood_fwd=likelihood_fwd, likelihood=likelihood)

            _, mu_fwd_pred_2[0], lower_fwd_pred_2[0], upper_fwd_pred_2[0], z = model.predict_latent_dynamics(z,
                                                          act[0].view(1, 1).cuda(),
                                                          likelihood_fwd)

            for i in range(seq_len - 1):
                mu[i + 1], mu_fwd[i + 1], lower[i + 1], upper[i + 1], lower_fwd[i + 1], upper_fwd[
                    i + 1], _, _, _ = model.predict_trajectory(
                    x=obs[i + 1].view(1, 6, obs_dim_1, obs_dim_2).cuda(), a=act[i + 1].view(1, 1).cuda(),
                    likelihood_fwd=likelihood_fwd, likelihood=likelihood)

                mu_pred[i + 1], mu_fwd_pred[i + 1], lower_pred[i + 1], upper_pred[i + 1], lower_fwd_pred[i + 1], \
                upper_fwd_pred[i + 1], mu_x, _, _ = model.predict_trajectory(
                    x=mu_x.view(1, 6, obs_dim_1, obs_dim_2).cuda(), a=act[i + 1].view(1, 1).cuda(),
                    likelihood_fwd=likelihood_fwd, likelihood=likelihood)

                _, mu_fwd_pred_2[i+1], lower_fwd_pred_2[i+1], upper_fwd_pred_2[i+1], z = model.predict_latent_dynamics(z,
                                                                                                        act[i + 1].view(1, 1).cuda(),
                                                                                                        likelihood_fwd)

        else:
            mu[0], mu_fwd[0], lower[0], upper[0], lower_fwd[0], upper_fwd[0], mu_x, _, z = model.predict_trajectory(
                x=obs[0].view(1, 6, obs_dim_1, obs_dim_2).cuda(),
                a=act[0].view(1, 1).cuda())

            _, mu_fwd_pred_2[0], lower_fwd_pred_2[0], upper_fwd_pred_2[0], z = model.predict_latent_dynamics(z,
                                                                                                             act[
                                                                                                                 0].view(
                                                                                                                 1,
                                                                                                                 1).cuda())

            for i in range(seq_len - 1):
                mu[i + 1], mu_fwd[i + 1], lower[i + 1], upper[i + 1], lower_fwd[i + 1], upper_fwd[
                    i + 1], _, _, _ = model.predict_trajectory(
                    x=obs[i + 1].view(1, 6, obs_dim_1, obs_dim_2).cuda(), a=act[i + 1].view(1, 1).cuda())

                mu_pred[i + 1], mu_fwd_pred[i + 1], lower_pred[i + 1], upper_pred[i + 1], lower_fwd_pred[i + 1], \
                upper_fwd_pred[i + 1], mu_x, _, _ = model.predict_trajectory(
                    x=mu_x.view(1, 6, obs_dim_1, obs_dim_2).cuda(), a=act[i + 1].view(1, 1).cuda())

                _, mu_fwd_pred_2[i + 1], lower_fwd_pred_2[i + 1], upper_fwd_pred_2[
                    i + 1], z = model.predict_latent_dynamics(z,
                                                              act[i + 1].view(1, 1).cuda())

        mu = mu.detach().cpu().numpy()
        mu_fwd = mu_fwd.detach().cpu().numpy()
        mu_pred = mu_pred.detach().cpu().numpy()
        mu_fwd_pred = mu_fwd_pred.detach().cpu().numpy()
        mu_fwd_pred_2 = mu_fwd_pred_2.detach().cpu().numpy()
        lower = lower.detach().cpu().numpy()
        upper = upper.detach().cpu().numpy()
        lower_fwd = lower_fwd.detach().cpu().numpy()
        upper_fwd = upper_fwd.detach().cpu().numpy()
        lower_pred = lower_pred.detach().cpu().numpy()
        upper_pred = upper_pred.detach().cpu().numpy()
        lower_fwd_pred = lower_fwd_pred.detach().cpu().numpy()
        upper_fwd_pred = upper_fwd_pred.detach().cpu().numpy()
        lower_fwd_pred_2 = lower_fwd_pred_2.detach().cpu().numpy()
        upper_fwd_pred_2 = upper_fwd_pred_2.detach().cpu().numpy()


        state = data['states']
        next_state = data['next_states']
        angle = np.arctan2(state[:, 1], state[:, 0])
        next_angle = np.arctan2(next_state[:, 1], next_state[:, 0])

        ang_idx = np.argsort(angle)

        ang_next_idx = np.argsort(next_angle)

        timestep = np.linspace(0, seq_len, angle.shape[0])

        true_frame = obs[:, 3:6, :, :]
        save_image(tensor=true_frame.view(seq_len, 3, obs_dim_1, obs_dim_2),
                   fp=save_dir + str(mtype) + '_' + str(exp) + '_sequence_' + str(start_idx[j]) + '_current_frame_' +
                   str(noise_level) + '.png')

        for i in range(3):
            fig = plt.figure(dpi=300)
            ax1 = fig.add_subplot()
            plt.plot(timestep[:], state[:, i], color='gray', linestyle='--')
            p1 = ax1.scatter(timestep[:], state[:, i], c=angle[:], cmap='hsv', zorder=2, s=5)
            #ax1.fill_between(timestep[:], lower[:, i], upper[:, i], color='gray', alpha=0.3)
            cbar = fig.colorbar(p1)
            cbar.set_label('angle', rotation=90)
            ax1.set_xlabel('timestep')
            ax1.set_ylabel('value')
            plt.savefig(save_dir + str(mtype) + '_' + str(exp) + '_traj_state_timesteps' + str(i) + '_' + str(
                noise_level) + '_init_idx_' + str(start_idx) + '.png')
            plt.close()

        #for i in range(latent_dim):
        #    fig = plt.figure(dpi=300)
        #    ax1 = fig.add_subplot()
        #    p1 = ax1.scatter(angle_sorted[:], mu_sorted[:, i], c=angle_sorted[:], cmap='hsv', zorder=2, s=5)
        #    ax1.fill_between(angle_sorted[:], lower_sorted[:, i], upper_sorted[:, i], color='gray', alpha=0.3)
        #    cbar = fig.colorbar(p1)
        #    cbar.set_label('angle', rotation=90)
        #    plt.savefig(save_dir + dir1 + str(mtype) + '_' + str(exp) + '_traj_conf_bound_z_dim' + str(i) + '_' + str(
        #        noise_level) + '_init_idx_' + str(start_idx) + '.png')
        #    plt.close()

        for i in range(latent_dim):
            fig = plt.figure(dpi=300)
            ax1 = fig.add_subplot()
            plt.plot(timestep[:], mu[:, i], color='gray', linestyle='--')
            p1 = ax1.scatter(timestep[:], mu[:, i], c=angle[:], cmap='hsv', zorder=2, s=5)
            ax1.fill_between(timestep[:], lower[:, i], upper[:, i], color='gray', alpha=0.3)
            cbar = fig.colorbar(p1)
            cbar.set_label('angle', rotation=90)
            ax1.set_xlabel('timestep')
            ax1.set_ylabel('value')
            plt.savefig(save_dir + dir1 + str(mtype) + '_' + str(exp) + '_traj_conf_bound_z_dim_timesteps' + str(i) + '_' + str(
                noise_level) + '_init_idx_' + str(start_idx) + '.png')
            plt.close()

        #for i in range(latent_dim):
        #    fig = plt.figure(dpi=300)
        #    ax1 = fig.add_subplot()
        #    p1 = ax1.scatter(next_angle_sorted[:], mu_fwd_sorted[:, i], c=next_angle_sorted[:], cmap='hsv', zorder=2, s=5)
        #    ax1.fill_between(next_angle_sorted[:], lower_fwd_sorted[:, i], upper_fwd_sorted[:, i], color='gray', alpha=0.3)
        #    cbar = fig.colorbar(p1)
        #    cbar.set_label('angle', rotation=90)
        #    plt.savefig(save_dir + dir2 + str(mtype) + '_' + str(exp) + '_traj_conf_bound_next_z_dim' + str(i) + '_' + str(
        #        noise_level) + '_init_idx_' + str(start_idx) + '.png')
        #    plt.close()

        for i in range(latent_dim):
            fig = plt.figure(dpi=300)
            ax1 = fig.add_subplot()
            plt.plot(timestep[:], mu_fwd[:, i], color='gray', linestyle='--')
            p1 = ax1.scatter(timestep[:], mu_fwd[:, i], c=next_angle[:], cmap='hsv', zorder=2, s=5)
            ax1.fill_between(timestep[:], lower_fwd[:, i], upper_fwd[:, i], color='gray', alpha=0.3)
            cbar = fig.colorbar(p1)
            cbar.set_label('angle', rotation=90)
            ax1.set_xlabel('timestep')
            ax1.set_ylabel('value')
            plt.savefig(save_dir + dir2 + str(mtype) + '_' + str(exp) + '_traj_conf_bound_next_z_dim_timesteps' + str(i) + '_' + str(
                noise_level) + '_init_idx_' + str(start_idx) + '.png')
            plt.close()

        #for i in range(latent_dim):
        #    fig = plt.figure(dpi=300)
        #    ax1 = fig.add_subplot()
        #    p1 = ax1.scatter(angle[:], mu_pred[:, i], c=angle[:], cmap='hsv', zorder=2, s=5)
        #    ax1.fill_between(angle[:], lower_pred[:, i], upper_pred[:, i], color='gray', alpha=0.3)
        #    cbar = fig.colorbar(p1)
        #    cbar.set_label('angle', rotation=90)
        #    plt.savefig(save_dir + dir3 + str(mtype) + '_' + str(exp) + '_traj_pred_conf_bound_z_dim' + str(i) + '_' + str(
        #        noise_level) + '_init_idx_' + str(start_idx) + '.png')
        #    plt.close()

        for i in range(latent_dim):
            fig = plt.figure(dpi=300)
            ax1 = fig.add_subplot()
            plt.plot(timestep[:], mu_pred[:, i], color='gray', linestyle='--')
            p1 = ax1.scatter(timestep[:], mu_pred[:, i], c=angle[:], cmap='hsv', zorder=2, s=5)
            ax1.fill_between(timestep[:], lower_pred[:, i], upper_pred[:, i], color='gray', alpha=0.3)
            cbar = fig.colorbar(p1)
            cbar.set_label('angle', rotation=90)
            ax1.set_xlabel('timestep')
            ax1.set_ylabel('value')
            plt.savefig(save_dir + dir3 + str(mtype) + '_' + str(exp) + '_traj_pred_conf_bound_z_dim_timesteps' + str(i) + '_' + str(
                noise_level) + '_init_idx_' + str(start_idx) + '.png')
            plt.close()

        #for i in range(latent_dim):
        #    fig = plt.figure(dpi=300)
        #    ax1 = fig.add_subplot()
        #    p1 = ax1.scatter(next_angle[:], mu_fwd_pred[:, i], c=next_angle[:], cmap='hsv', zorder=2, s=5)
        #    ax1.fill_between(next_angle[:], lower_fwd_pred[:, i], upper_fwd_pred[:, i], color='gray', alpha=0.3)
        #    cbar = fig.colorbar(p1)
        #    cbar.set_label('angle', rotation=90)
        #    plt.savefig(
        #        save_dir + dir4 + str(mtype) + '_' + str(exp) + '_traj_pred_conf_bound_next_z_dim' + str(i) + '_' + str(
        #            noise_level) + '_init_idx_' + str(start_idx) + '.png')
        #    plt.close()

        for i in range(latent_dim):
            fig = plt.figure(dpi=300)
            ax1 = fig.add_subplot()
            plt.plot(timestep[:], mu_fwd_pred[:, i], color='gray', linestyle='--')
            p1 = ax1.scatter(timestep[:], mu_fwd_pred[:, i], c=next_angle[:], cmap='hsv', zorder=2, s=5)
            ax1.fill_between(timestep[:], lower_fwd_pred[:, i], upper_fwd_pred[:, i], color='gray', alpha=0.3)
            cbar = fig.colorbar(p1)
            cbar.set_label('angle', rotation=90)
            ax1.set_xlabel('timestep')
            ax1.set_ylabel('value')
            plt.savefig(save_dir + dir4 + str(mtype) + '_' + str(exp) + '_traj_pred_conf_bound_next_z_dim_timesteps' + str(i) + '_' + str(
                noise_level) + '_init_idx_' + str(start_idx) + '.png')
            plt.close()

        #for i in range(latent_dim):
        #    fig = plt.figure(dpi=300)
        #    ax1 = fig.add_subplot()
        #    p1 = ax1.scatter(next_angle[:], mu_fwd_pred_2[:, i], c=next_angle[:], cmap='hsv', zorder=2, s=5)
        #    ax1.fill_between(next_angle[:], lower_fwd_pred_2[:, i], upper_fwd_pred_2[:, i], color='gray', alpha=0.3)
        #    cbar = fig.colorbar(p1)
        #    cbar.set_label('angle', rotation=90)
        #    plt.savefig(
        #        save_dir + dir5 + str(mtype) + '_' + str(exp) + '_traj_pred_conf_bound_next_z_dim' + str(i) + '_' + str(
        #            noise_level) + '_init_idx_' + str(start_idx) + '.png')
        #    plt.close()

        for i in range(latent_dim):
            fig = plt.figure(dpi=300)
            ax1 = fig.add_subplot()
            plt.plot(timestep[:], mu_fwd_pred_2[:, i], color='gray', linestyle='--')
            p1 = ax1.scatter(timestep[:], mu_fwd_pred_2[:, i], c=next_angle[:], cmap='hsv', zorder=2, s=5)
            ax1.fill_between(timestep[:], lower_fwd_pred_2[:, i], upper_fwd_pred_2[:, i], color='gray', alpha=0.3)
            cbar = fig.colorbar(p1)
            cbar.set_label('angle', rotation=90)
            ax1.set_xlabel('timestep')
            ax1.set_ylabel('value')
            plt.savefig(save_dir + dir5 + str(mtype) + '_' + str(exp) + '_traj_pred_conf_bound_next_z_dim_timesteps' + str(i) + '_' + str(
                noise_level) + '_init_idx_' + str(start_idx) + '.png')
            plt.close()




def make_reconstructions_predictions_plots(model, likelihood, likelihood_fwd, test_loader, exp, mtype, noise_level=0.0,
                                           obs_dim_1=84, obs_dim_2=84):

    save_dir = results_dir + '/' + str(exp) + '/' + str(mtype) + '/Noise_level_' + str(
                       noise_level) + '/reconstructions/'

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    start_idx = [1, 231, 115]
    seq_len = 24
    for j in range(len(start_idx)):
        data = test_loader.sample_sequence(start_idx=start_idx[j], seq_len=seq_len)
        obs = torch.from_numpy(data['obs1']).permute(0, 3, 1, 2)
        act = torch.from_numpy(data['acts'])
        obs2 = torch.from_numpy(data['obs2']).permute(0, 3, 1, 2)

        sample = torch.zeros(seq_len, 6, obs_dim_1,
                             obs_dim_2)  # torch.from_numpy(np.zeros((seq_len, 6, obs_dim_1, obs_dim_2)))

        sample_res_fwd = torch.zeros(seq_len, 6, obs_dim_1, obs_dim_2)

        # _, next_dist, _ = model.predict_dynamics(obs[0].view(1, 6, obs_dim_1, obs_dim_2).cuda(), act[0].view(1, 1).cuda())
        # next_dist_l = likelihood_fwd(next_dist)
        # sample[0] = model.decoder(next_dist_l.sample())

        if mtype == 'DKL':
            sample[0], sample_res_fwd[0] = model.predict_dynamics(x=obs[0].view(1, 6, obs_dim_1, obs_dim_2).cuda(),
                                                                  a=act[0].view(1, 1).cuda(),
                                                                  likelihood_fwd=likelihood_fwd, likelihood=likelihood)

            for i in range(seq_len - 1):
                # sample[i + 1], sample_res_fwd[i + 1] = model.predict_dynamics(
                #    x=sample[i].view(1, 6, obs_dim_1, obs_dim_2).cuda(), a=act[i + 1].view(1, 1).cuda(),
                #    likelihood_fwd=likelihood_fwd)
                sample[i + 1], sample_res_fwd[i + 1] = model.predict_dynamics(
                    x=obs[i + 1].view(1, 6, obs_dim_1, obs_dim_2).cuda(), a=act[i + 1].view(1, 1).cuda(),
                    likelihood_fwd=likelihood_fwd, likelihood=likelihood)

        else:
            sample[0], sample_res_fwd[0] = model.predict_dynamics(x=obs[0].view(1, 6, obs_dim_1, obs_dim_2).cuda(),
                                                                  a=act[0].view(1, 1).cuda())
            for i in range(seq_len - 1):
                #sample[i + 1], sample_res_fwd[i + 1] = model.predict_dynamics(
                #    x=sample[i].view(1, 6, obs_dim_1, obs_dim_2).cuda(), a=act[i + 1].view(1, 1).cuda(),
                #    likelihood_fwd=likelihood_fwd)
                sample[i + 1], sample_res_fwd[i + 1] = model.predict_dynamics(
                    x=obs[i+1].view(1, 6, obs_dim_1, obs_dim_2).cuda(), a=act[i + 1].view(1, 1).cuda())
            # model.predict_latent_dynamics(z.view(1, latent_dim), act[i+1].view(1, 1).cuda())
            # sample = torch.from_numpy(sample)

        true_frame = obs2[:, 3:6, :, :]
        current_frame = sample[:, 3:6, :, :]
        true_prev = obs2[:, :3, :, :]
        prev_frame = sample[:, :3, :, :]
        current_frame_res_fwd = sample_res_fwd[:, 3:6, :, :]
        prev_frame_res_fwd = sample_res_fwd[:, :3, :, :]
        # current_frame = make_grid(current_frame, nrow=seq_len)
        save_image(tensor=current_frame.view(seq_len, 3, obs_dim_1, obs_dim_2),
                   fp=save_dir + str(mtype) + '_' + str(exp) + '_sequence_' + str(start_idx[j]) + '_current_frame_' +
                   str(noise_level) + '.png')
        save_image(tensor=prev_frame.view(seq_len, 3, obs_dim_1, obs_dim_2),
                   fp=save_dir + str(mtype) + '_' + str(exp) + '_sequence_' + str(start_idx[j]) + '_prev_frame_' +
                   str(noise_level) + '.png')
        save_image(tensor=true_frame.view(seq_len, 3, obs_dim_1, obs_dim_2),
                   fp=save_dir + str(mtype) + '_' + str(exp) + '_sequence_' + str(start_idx[j])
                   + '_true_current_frame_GT_' + str(noise_level) + '.png')
        save_image(tensor=true_prev.view(seq_len, 3, obs_dim_1, obs_dim_2),
                   fp=save_dir + str(mtype) + '_' + str(exp) + '_sequence_' + str(start_idx[j]) + '_true_prev_frame_GT_'
                   + str(noise_level) + '.png')
        save_image(tensor=current_frame_res_fwd.view(seq_len, 3, obs_dim_1, obs_dim_2),
                   fp=save_dir + str(mtype) + '_' + str(exp) + '_sequence_' + str(start_idx[j])
                   + '_current_frame_res_fwd_' + str(noise_level) + '.png')
        save_image(tensor=prev_frame_res_fwd.view(seq_len, 3, obs_dim_1, obs_dim_2),
                   fp=save_dir + str(mtype) + '_' + str(exp) + '_sequence_' + str(start_idx[j]) + '_prev_frame_res_fwd_'
                   + str(noise_level) + '.png')
    return print("Reconstruction Predictions are printed")

def make_reconstructions_plots(model, likelihood, test_loader, exp, mtype, det_dec=False, noise_level=0.0,
                               obs_dim_1=84, obs_dim_2=84):

    save_dir = results_dir + '/' + str(exp) + '/' + str(mtype) + '/Noise_level_' + str(
        noise_level) + '/observation_sequence/'

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_dir_2 = results_dir + '/' + str(exp) + '/' + str(mtype) + '/Noise_level_' + str(
        noise_level) + '/reconstructions/'

    if not os.path.exists(save_dir_2):
        os.makedirs(save_dir_2)

    # idx = np.random.randint(1, counter - 1)
    # img = train_loader.sample_batch(1)
    img = test_loader.sample_sequence(start_idx=1, seq_len=11)
    for i in range(10):
        current_frame = img['obs1'][i].reshape(obs_dim_1, obs_dim_2, 6)
        current_frame = current_frame[:, :, 3:6]
        prev_frame = img['obs1'][0].reshape(obs_dim_1, obs_dim_2, 6)
        prev_frame = prev_frame[:, :, :3]

        save_image(torch.from_numpy(current_frame).permute(2, 0, 1),
                   save_dir + str(mtype) + '_' + str(exp) + '_current_frame_GT_' + str(noise_level) + '_' + str(i) + '.png')
        save_image(torch.from_numpy(prev_frame).permute(2, 0, 1),
                   save_dir + str(mtype) + '_' + str(exp) + '_prev_frame_GT_' + str(noise_level) + '_' + str(0) + '.png')

    num_samples = 8
    if mtype == 'DKL':
        data = test_loader.sample_batch(num_samples)
        input = torch.from_numpy(data['obs1']).permute(0, 3, 1, 2)
        input2 = torch.from_numpy(data['acts'])
        input3 = torch.from_numpy(data['obs2']).permute(0, 3, 1, 2)

        sample, _, mu, var, z, dist, _, _, _, _, _, _, _, _, _ = model(input.cuda(), input2.cuda(), input3.cuda(),
                                                                       likelihood)

        sample, _ = model.decoder(likelihood(dist).sample())

    else:
        data = test_loader.sample_batch(num_samples)
        input = torch.from_numpy(data['obs1']).permute(0, 3, 1, 2)
        input2 = torch.from_numpy(data['acts'])
        input3 = torch.from_numpy(data['obs2']).permute(0, 3, 1, 2)

        if det_dec:
            sample, mu, log_var, z, _, _, _, _, _, _ = model(input.cuda(), input2.cuda(), input3.cuda())
        else:
            mu_x, log_var_x, mu, log_var, z, _, _, _, _, _, _, _, _, _ = model(input.cuda(), input2.cuda(),
                                                                               input3.cuda())
            sample = mu_x

    true_frame = input[:, 3:6, :, :]
    current_frame = sample[:, 3:6, :, :]
    true_prev = input[:, :3, :, :]
    prev_frame = sample[:, :3, :, :]
    save_image(current_frame.view(num_samples, 3, obs_dim_1, obs_dim_2),
               save_dir_2 + str(mtype) + '_' + str(exp) + '_current_frame_' + str(noise_level) + '.png')
    save_image(prev_frame.view(num_samples, 3, obs_dim_1, obs_dim_2),
               save_dir_2 + str(mtype) + '_' + str(exp) + '_prev_frame_' + str(noise_level) + '.png')
    save_image(true_frame.view(num_samples, 3, obs_dim_1, obs_dim_2),
               save_dir_2 + str(mtype) + '_' + str(exp) + '_true_current_frame_GT_' + str(noise_level) + '.png')
    save_image(true_prev.view(num_samples, 3, obs_dim_1, obs_dim_2),
               save_dir_2 + str(mtype) + '_' + str(exp) + '_true_prev_frame_GT_' + str(noise_level) + '.png')
    return print("Reconstruction are printed")



def plot_results(model, likelihood, likelihood_fwd, test_loader, exp, mtype, latent_dim, det_dec=False, noise_level=0.0,
                 PCA=False, state_dim=1, obs_dim_1=84, obs_dim_2=84, num_samples_plot=5, batch_size=50):

    make_UQ_single_traj(model, likelihood, likelihood_fwd, test_loader, exp, mtype, noise_level, obs_dim_1, obs_dim_2,
                        latent_dim)

    make_reconstructions_predictions_plots(model, likelihood, likelihood_fwd, test_loader, exp, mtype,
                                           noise_level=noise_level, obs_dim_1=obs_dim_1, obs_dim_2=obs_dim_2)

    make_reconstructions_plots(model, likelihood, test_loader, exp, mtype, det_dec=False, noise_level=noise_level,
                               obs_dim_1=obs_dim_1, obs_dim_2=obs_dim_2)



    mu_list = []
    var_list = []
    var_y_list = []
    z_list = []
    state_list = []
    mu_y_list = []
    mu_fwd_list = []
    var_fwd_list = []
    z_fwd_lik_list = []
    z_y_list = []
    z_fwd_var_l_list = []
    lower_list = []
    upper_list = []

    lower_znext__list = []
    upper_znext_list = []

    z_y_samples_list = []
    angles_samples_list = []
    next_angles_samples_list = []
    p_y_samples_list = []
    z_fwd_l_samples_list = []
    z_fwd_y_samples_list = []
    next_state_list = []

    n_samp = num_samples_plot
    batch_size = batch_size
    num_data_sampled = n_samp*batch_size

    for i in range(n_samp):
        data = test_loader.sample_batch(batch_size=batch_size)

        if mtype == 'DKL':
            _, _, mu, var, z, dist, mu_target, var_target, res_target, mu_fwd, var_fwd, res_fwd, _, _, _ = model(torch.from_numpy(data['obs1']).permute(0, 3, 1, 2).cuda(),
                                                       torch.from_numpy(data['acts']).cuda(),
                                                       torch.from_numpy(data['obs2']).permute(0, 3, 1, 2).cuda(), likelihood)


            z_target = likelihood(res_target).sample()
            #z_target = likelihood(res_target).sample(sample_shape=torch.Size([16])).mean(0)

            z_fwd = res_fwd.sample()
            #z_fwd = res_fwd.sample(sample_shape=torch.Size([16])).mean(0)

            z_fwd_l = likelihood_fwd(res_fwd).sample()
            #z_fwd_l = likelihood_fwd(res_fwd).sample(sample_shape=torch.Size([16])).mean(0)

            fwd_l = likelihood_fwd(res_fwd)

            z_fwd_var_l = likelihood_fwd(res_fwd).variance

            z_y = likelihood(dist).sample()
            #z_y = likelihood(dist).sample(sample_shape=torch.Size([16])).mean(0)

            p_y = likelihood(dist)

            # Get upper and lower confidence bounds
            lower, upper = p_y.confidence_region()
            lower_znext, upper_znext = fwd_l.confidence_region()

            mu_y = p_y.mean
            var_y = p_y.variance

            #z_target_f = res_target.sample()
            #z_fwd_f = res_fwd.sample()
            #z_f = dist.sample()

            var_list.append(var.cpu().numpy())


            for i in range(n_samp):
                p_y_samples_list.append(p_y.sample().cpu().numpy())
            z_y_samples = np.array(p_y_samples_list).reshape(batch_size*n_samp, latent_dim)
            p_y_samples_list = []

            for i in range(n_samp):
                z_fwd_l_samples_list.append(fwd_l.sample().cpu().numpy())
            z_fwd_y_samples = np.array(z_fwd_l_samples_list).reshape(batch_size*n_samp, latent_dim)
            z_fwd_l_samples_list = []

        else:

            if det_dec:
                _, mu, std, z, _, _, _, _, _, _ = model(torch.from_numpy(data['obs1']).permute(0, 3, 1, 2).cuda(),
                                                       torch.from_numpy(data['acts']).cuda(),
                                                       torch.from_numpy(data['obs2']).permute(0, 3, 1, 2).cuda())
            else:
                _, _, mu, std, z, _, _, _, mu_fwd, std_fwd, z_fwd, _, _, _ = model(
                    torch.from_numpy(data['obs1']).permute(0, 3, 1, 2).cuda(),
                    torch.from_numpy(data['acts']).cuda(),
                    torch.from_numpy(data['obs2']).permute(0, 3, 1, 2).cuda())
                std = std.detach()
                mu = mu.detach()
                z = z.detach()
                mu_fwd = mu_fwd.detach()
                std_fwd = std_fwd.detach()
                z_fwd_l = z_fwd.detach()
                var_fwd = torch.square(std_fwd.detach())
                lower = mu - std
                upper = mu + std
                lower_znext = mu_fwd - std_fwd
                upper_znext = mu_fwd + std_fwd

                z_y = z
                z_fwd_var_l = var_fwd
                var_y = torch.square(std)

                for i in range(n_samp):
                    _, _, _, _, z_m, _, _, _, _, _, z_fwd_m, _, _, _ = model(
                        torch.from_numpy(data['obs1']).permute(0, 3, 1, 2).cuda(),
                        torch.from_numpy(data['acts']).cuda(),
                        torch.from_numpy(data['obs2']).permute(0, 3, 1, 2).cuda())

                    p_y_samples_list.append(z_m.detach().cpu().numpy())
                    z_fwd_l_samples_list.append(z_fwd_m.detach().cpu().numpy())

                z_y_samples = np.array(p_y_samples_list).reshape(batch_size * n_samp, latent_dim)
                p_y_samples_list = []
                z_fwd_y_samples = np.array(z_fwd_l_samples_list).reshape(batch_size * n_samp, latent_dim)
                z_fwd_l_samples_list = []

                var_list.append(torch.square(std).cpu().numpy())


        state_list.append(data['states'])
        next_state_list.append(data['next_states'])
        mu_list.append(mu.cpu().numpy())
        z_list.append(z.cpu().numpy())
        #mu_y_list.append(mu_y.cpu().numpy())
        mu_fwd_list.append(mu_fwd.cpu().numpy())
        var_fwd_list.append(var_fwd.cpu().numpy())
        z_fwd_lik_list.append(z_fwd_l.cpu().numpy())
        z_y_list.append(z_y.cpu().numpy())
        z_fwd_var_l_list.append(z_fwd_var_l.cpu().numpy())
        var_y_list.append(var_y.cpu().numpy())

        lower_list.append(lower.cpu().numpy())
        upper_list.append(upper.cpu().numpy())
        lower_znext__list.append(lower_znext.cpu().numpy())
        upper_znext_list.append(upper_znext.cpu().numpy())

        z_y_samples_list.append(z_y_samples)
        states = data['states']
        angle = np.arctan2(states[:, 1], states[:, 0])
        for i in range(n_samp):
            angles_samples_list.append(angle)

        z_fwd_y_samples_list.append(z_fwd_y_samples)
        next_states = data['next_states']
        next_angle = np.arctan2(next_states[:, 1], next_states[:, 0])
        for i in range(n_samp):
            next_angles_samples_list.append(next_angle)


    if PCA:
        mu_PCA = compute_PCA(normalize(np.array(mu_list).reshape(num_data_sampled, latent_dim)), 3)
        mu_PCA_2d = compute_PCA(normalize(np.array(mu_list).reshape(num_data_sampled, latent_dim)), 2)
        states = np.array(state_list).reshape(num_data_sampled, state_dim)
        next_states = np.array(next_state_list).reshape(num_data_sampled, state_dim)
        mu_fwd = compute_PCA(normalize(np.array(mu_fwd_list).reshape(num_data_sampled, latent_dim)), 3)
        mu_fwd_2D = compute_PCA(normalize(np.array(mu_fwd_list).reshape(num_data_sampled, latent_dim)), 2)

    else:
        mu_PCA = TSNE(n_components=3, learning_rate='auto').fit_transform(normalize(np.array(mu_list).reshape(num_data_sampled, latent_dim)))
        mu_PCA_2d = TSNE(n_components=2, learning_rate='auto').fit_transform(normalize(np.array(mu_list).reshape(num_data_sampled, latent_dim)))

        states = np.array(state_list).reshape(num_data_sampled, state_dim)
        next_states = np.array(next_state_list).reshape(num_data_sampled, state_dim)

        mu_fwd = TSNE(n_components=3, learning_rate='auto', init='pca').fit_transform(normalize(np.array(mu_fwd_list).reshape(num_data_sampled, latent_dim)))
        mu_fwd_2D = TSNE(n_components=2, learning_rate='auto', init='pca').fit_transform(
            normalize(np.array(mu_fwd_list).reshape(num_data_sampled, latent_dim)))

        #var_y = TSNE(n_components=3, learning_rate='auto', init='pca').fit_transform(normalize(np.array(var_y_list).reshape(num_data_sampled, latent_dim)))


    z_y_samples = TSNE(n_components=3, learning_rate='auto', init='pca').fit_transform(np.array(z_y_samples_list).reshape(num_data_sampled*n_samp, latent_dim))
    angles_samples = np.array(angles_samples_list).reshape(num_data_sampled*n_samp, )
    z_y_samples_2D = TSNE(n_components=2, learning_rate='auto', init='pca').fit_transform(np.array(z_y_samples_list).reshape(num_data_sampled * n_samp, latent_dim))

    z_fwd_y_samples = TSNE(n_components=3, learning_rate='auto', init='pca').fit_transform(np.array(z_fwd_y_samples_list).reshape(num_data_sampled * n_samp, latent_dim))
    next_angles_samples = np.array(next_angles_samples_list).reshape(num_data_sampled * n_samp, )
    z_fwd_y_samples_2D = TSNE(n_components=2, learning_rate='auto', init='pca').fit_transform(np.array(z_fwd_y_samples_list).reshape(num_data_sampled * n_samp, latent_dim))


    mu_z_norm = normalize(np.array(mu_list).reshape(num_data_sampled, latent_dim))
    mu_z = np.array(mu_list).reshape(num_data_sampled, latent_dim)
    #mu_y = np.array(mu_y_list).reshape(num_data_sampled, latent_dim)

    mu_znext_norm = normalize(np.array(mu_fwd_list).reshape(num_data_sampled, latent_dim))
    mu_znext = np.array(mu_fwd_list).reshape(num_data_sampled, latent_dim)
    var_y_10 = normalize(np.array(var_y_list).reshape(num_data_sampled, latent_dim))
    fwd_var_l_10 = normalize(np.array(var_fwd_list).reshape(num_data_sampled, latent_dim))

    angle = np.arctan2(states[:, 1], states[:, 0])
    next_angle = np.arctan2(next_states[:, 1], next_states[:, 0])


    save_dir = results_dir + '/' + str(exp) + '/' + str(mtype) + '/Noise_level_' + str(noise_level)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(projection='3d')
    p1 = ax.scatter(z_y_samples[:, 0], z_y_samples[:, 1], z_y_samples[:, 2], s=15, c=angles_samples, cmap='hsv')
    ax.legend([p1], ['$z_{t}$ ~ p($z_{t}$|$x_{t}$)'])
    cbar = fig.colorbar(p1)
    cbar.set_label('angle', rotation=90)
    plt.savefig(save_dir + '/' + str(mtype) + '_' + str(exp) + '_z_multi-samples_' + str(noise_level) + '.png')
    plt.close()

    fig = plt.figure(dpi=300)
    ax = fig.add_subplot()
    p1 = ax.scatter(z_y_samples_2D[:, 0], z_y_samples_2D[:, 1], s=15, c=angles_samples, cmap='hsv')
    ax.legend([p1], ['$z_{t}$ ~ p($z_{t}$|$x_{t}$)'])
    cbar = fig.colorbar(p1)
    cbar.set_label('angle', rotation=90)
    plt.savefig(save_dir + '/' + str(mtype) + '_' + str(exp) + '_z_multi-samples_2D_' + str(noise_level) + '.png')
    plt.close()

    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(projection='3d')
    p1 = ax.scatter(z_fwd_y_samples[:, 0], z_fwd_y_samples[:, 1], z_fwd_y_samples[:, 2], s=15, c=next_angles_samples, cmap='hsv')
    ax.legend([p1], ["$z_{t+1}$ ~ p($z_{t+1}$|$z_{t}$,$u_{t}$)"])
    cbar = fig.colorbar(p1)
    cbar.set_label('angle', rotation=90)
    plt.savefig(save_dir + '/' + str(mtype) + '_' + str(exp) + '_z_fwd_multi-samples_' + str(noise_level) + '.png')
    plt.close()

    fig = plt.figure(dpi=300)
    ax = fig.add_subplot()
    p1 = ax.scatter(z_fwd_y_samples_2D[:, 0], z_fwd_y_samples_2D[:, 1], s=15, c=next_angles_samples,
                    cmap='hsv')
    ax.legend([p1], ["$z_{t+1}$ ~ p($z_{t+1}$|$z_{t}$,$u_{t}$)"])
    cbar = fig.colorbar(p1)
    cbar.set_label('angle', rotation=90)
    plt.savefig(save_dir + '/' + str(mtype) + '_' + str(exp) + '_z_fwd_multi-samples_2D_' + str(noise_level) + '.png')
    plt.close()

    fig = plt.figure(dpi=300)
    pca = compute_PCA_2(latent_dim)
    pca.fit_transform(np.array(z_list).reshape(num_data_sampled, latent_dim))
    #
    # Determine explained variance using explained_variance_ration_ attribute
    #
    exp_var_pca = pca.explained_variance_ratio_
    #
    # Cumulative sum of eigenvalues; This will be used to create step plot
    # for visualizing the variance explained by each principal component.
    #
    cum_sum_eigenvalues = np.cumsum(exp_var_pca)

    plt.bar(range(0, len(exp_var_pca)), exp_var_pca, alpha=0.5, align='center', label='Individual explained variance')
    plt.step(range(0, len(cum_sum_eigenvalues)), cum_sum_eigenvalues, where='mid',
             label='Cumulative explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal component index')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(save_dir + '/' + str(mtype) + '_' + str(exp) + '_variance_explained_z_' + str(noise_level) + '.png')
    plt.close()

    fig = plt.figure(dpi=300)
    ax = fig.add_subplot()
    p1 = ax.scatter(mu_PCA_2d[:, 0], mu_PCA_2d[:, 1], s=15, c=angle, cmap='hsv')
    ax.legend([p1], ['mean of p($z_{t}$|$x_{t}$)'], loc='upper right')
    cbar = fig.colorbar(p1)
    cbar.set_label('angle', rotation=90)
    plt.savefig(save_dir + '/' + str(mtype) + '_' + str(exp) + '_mu_2d_' + str(noise_level) + '.png')
    plt.close()

    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(projection='3d')
    p1 = ax.scatter(mu_PCA[:, 0], mu_PCA[:, 1], mu_PCA[:, 2], s=15, c=angle, cmap='hsv')
    ax.legend([p1], ['mean of p($z_{t}$|$x_{t}$)'], loc='upper right')
    cbar = fig.colorbar(p1)
    cbar.set_label('angle', rotation=90)
    plt.savefig(save_dir + '/' + str(mtype) + '_' + str(exp) + '_mu_' + str(noise_level) + '.png')
    plt.close()

    fig = plt.figure(dpi=300)
    ax = fig.add_subplot()
    p1 = ax.scatter(mu_fwd_2D[:, 0], mu_fwd_2D[:, 1], s=15, c=next_angle, cmap='hsv')
    ax.legend([p1], ['mean of p($z_{t+1}$|$z_{t}$,$u_{t}$)'], loc='upper right')
    cbar = fig.colorbar(p1)
    cbar.set_label('angle', rotation=90)
    plt.savefig(save_dir + '/' + str(mtype) + '_' + str(exp) + '_mu_fwd_2d_' + str(noise_level) + '.png')
    plt.close()

    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(projection='3d')
    p1 = ax.scatter(mu_fwd[:, 0], mu_fwd[:, 1], mu_fwd[:, 2], s=15, c=next_angle, cmap='hsv')
    ax.legend([p1], ['mean of p($z_{t+1}$|$z_{t}$,$u_{t}$)'], loc='upper right')
    cbar = fig.colorbar(p1)
    cbar.set_label('angle', rotation=90)
    plt.savefig(save_dir + '/' + str(mtype) + '_' + str(exp) + '_mu_fwd_' + str(noise_level) + '.png')
    plt.close()

    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(projection='3d')
    p4 = ax.scatter(states[:, 0], states[:, 1], states[:, 2], s=15, c=np.arctan2(states[:, 1], states[:, 0]),
                    cmap='hsv')
    ax.legend([p4], ['true states - true rew'])
    cbar = fig.colorbar(p4)
    cbar.set_label('angle', rotation=90)
    plt.savefig(save_dir + '/' + str(mtype) + '_' + str(exp) + '_true_states_' + str(noise_level) + '.png')
    plt.close()


    ###################################

    ang_idx = np.argsort(angle)
    std_y_10 = np.sqrt(var_y_10[ang_idx])
    std_y_10_norm = normalize(std_y_10)
    mu_sorted = mu_z[ang_idx]
    angle_sorted = angle[ang_idx]

    mu_norm_sorted = mu_z_norm[ang_idx]

    lower = np.array(lower_list).reshape(num_data_sampled, latent_dim)
    upper = np.array(upper_list).reshape(num_data_sampled, latent_dim)
    lower_sorted = lower[ang_idx]
    upper_sorted = upper[ang_idx]

    idx = np.sort(np.random.choice(angle.shape[0], num_data_sampled, replace=False))

    save_dir_2 = results_dir + '/' + str(exp) + '/' + str(mtype) + '/Noise_level_' + str(noise_level) + '/encoder_embeddings/'

    if not os.path.exists(save_dir_2):
        os.makedirs(save_dir_2)

    for i in range(latent_dim):
        fig = plt.figure(dpi=300)
        ax1 = fig.add_subplot()
        p1 = ax1.scatter(angle_sorted[idx], mu_norm_sorted[idx, i], c=angle_sorted[idx], cmap='hsv', zorder=2, s=5)
        ax1.fill_between(angle_sorted[idx], mu_norm_sorted[idx, i] - std_y_10_norm[idx, i], mu_norm_sorted[idx, i] + std_y_10_norm[idx, i], color='gray', alpha=0.3)
        cbar = fig.colorbar(p1)
        cbar.set_label('angle', rotation=90)
        plt.savefig(save_dir_2 + str(mtype) + '_' + str(exp) + '_mu_var_z_dim_normalized' + str(i) + '_' + str(noise_level) + '.png')
        plt.close()

    for i in range(latent_dim):
        fig = plt.figure(dpi=300)
        ax1 = fig.add_subplot()
        p1 = ax1.scatter(angle_sorted[idx], mu_sorted[idx, i], c=angle_sorted[idx], cmap='hsv', zorder=2, s=5)
        ax1.fill_between(angle_sorted[idx], mu_sorted[idx, i] - std_y_10[idx, i], mu_sorted[idx, i] + std_y_10[idx, i], color='gray', alpha=0.3)
        cbar = fig.colorbar(p1)
        cbar.set_label('angle', rotation=90)
        plt.savefig(save_dir_2 + str(mtype) + '_' + str(exp) + '_mu_var_z_dim' + str(i) + '_' + str(noise_level) + '.png')
        plt.close()

    for i in range(latent_dim):
        fig = plt.figure(dpi=300)
        ax1 = fig.add_subplot()
        p1 = ax1.scatter(angle_sorted[idx], mu_sorted[idx, i], c=angle_sorted[idx], cmap='hsv', zorder=2, s=5)
        ax1.fill_between(angle_sorted[idx], lower_sorted[idx, i], upper_sorted[idx, i], color='gray', alpha=0.3)
        cbar = fig.colorbar(p1)
        cbar.set_label('angle', rotation=90)
        plt.savefig(save_dir_2 + str(mtype) + '_' + str(exp) + '_mu_conf_bound_z_dim' + str(i) + '_' + str(noise_level) + '.png')
        plt.close()

    for i in range(3):
        fig = plt.figure(dpi=300)
        ax1 = fig.add_subplot()
        p1 = ax1.scatter(angle, states[:, i], c=angle, cmap='hsv')
        cbar = fig.colorbar(p1)
        cbar.set_label('angle', rotation=90)
        plt.savefig(save_dir_2 + str(mtype) + '_' + str(exp) + '_state_dim' + str(i) + '_' + str(noise_level) + '.png')
        plt.close()

    #############################################################

    next_ang_idx = np.argsort(next_angle)
    std_y_10 = np.sqrt(fwd_var_l_10[next_ang_idx])
    mu_sorted = mu_znext[next_ang_idx]
    angle_sorted = next_angle[next_ang_idx]


    lower_znext = np.array(lower_znext__list).reshape(num_data_sampled, latent_dim)
    upper_znext = np.array(upper_znext_list).reshape(num_data_sampled, latent_dim)
    lower_znext_sorted = lower_znext[next_ang_idx]
    upper_znext_sorted = upper_znext[next_ang_idx]

    save_dir_3 = results_dir + '/' + str(exp) + '/' + str(mtype) + '/Noise_level_' + str(
        noise_level) + '/forward_model_embeddings/'

    if not os.path.exists(save_dir_3):
        os.makedirs(save_dir_3)


    for i in range(latent_dim):
        fig = plt.figure(dpi=300)
        ax1 = fig.add_subplot()
        p1 = ax1.scatter(angle_sorted[idx], mu_sorted[idx, i], c=angle_sorted[idx], cmap='hsv', zorder=2, s=5)
        ax1.fill_between(angle_sorted[idx], mu_sorted[idx, i] - std_y_10[idx, i], mu_sorted[idx, i] + std_y_10[idx, i], color='gray',
                         alpha=0.3)
        cbar = fig.colorbar(p1)
        cbar.set_label('angle', rotation=90)
        plt.savefig(save_dir_3 + str(mtype) + '_' + str(exp) + '_mu_var_z_dim' + str(i) + '_' + str(noise_level) + '.png')
        plt.close()

    for i in range(latent_dim):
        fig = plt.figure(dpi=300)
        ax1 = fig.add_subplot()
        p1 = ax1.scatter(angle_sorted[idx], mu_sorted[idx, i], c=angle_sorted[idx], cmap='hsv', zorder=2, s=5)
        ax1.fill_between(angle_sorted[idx], lower_znext_sorted[idx, i], upper_znext_sorted[idx, i], color='gray',
                         alpha=0.3)
        cbar = fig.colorbar(p1)
        cbar.set_label('angle', rotation=90)
        plt.savefig(save_dir_3 + str(mtype) + '_' + str(exp) + '_mu_conf_bound_z_dim' + str(i) + '_' + str(noise_level) + '.png')
        plt.close()

