import torch
from losses import loss_negloglikelihood, kl_divergence_balance, loss_bce

def train_DKL(epoch, batch_size, nr_data, train_loader, model, optimizers,
              variational_kl_term, variational_kl_term_fwd, k1=1, beta=1):

    model.train()
    model.AE_DKL.likelihood.train()
    model.fwd_model_DKL.likelihood.train()

    optimizer_var1 = optimizers[1]
    optimizer_var2 = optimizers[2]
    optimizer = optimizers[0]

    train_loss = 0
    train_loss_vae = 0
    train_loss_varKL_vae = 0
    train_loss_fwd = 0
    train_loss_varKL_fwd = 0

    for i in range(int(nr_data/batch_size)):
        data = train_loader.sample_batch(batch_size)

        obs = torch.from_numpy(data['obs1']).permute(0, 3, 1, 2).cuda()
        a = torch.from_numpy(data['acts']).cuda()
        next_obs = torch.from_numpy(data['obs2']).permute(0, 3, 1, 2).cuda()

        optimizer.zero_grad()
        optimizer_var1.zero_grad()
        optimizer_var2.zero_grad()

        mu_x, var_x, _, _, _, res, mu_target, var_target, res_target, _, _, res_fwd, _ = model(obs, a, next_obs)

        # compute vae loss + variational inference
        #loss_vae = loss_negloglikelihood(mu_x, obs, var_x, dim=3)
        loss_vae = loss_bce(mu_x, obs)
        loss_varKL_vae = variational_kl_term(beta=1)

        # compute forward model loss (KL divergence) + variational inference
        loss_fwd = -beta * kl_divergence_balance(model.AE_DKL.likelihood(res_target).mean,
                                                  model.AE_DKL.likelihood(res_target).variance,
                                                  model.fwd_model_DKL.likelihood(res_fwd).mean,
                                                  model.fwd_model_DKL.likelihood(res_fwd).variance, alpha=0.8,
                                                  dim=1)
        loss_varKL_fwd = variational_kl_term_fwd(beta=1)

        # loss = (-) loss_vae + (+) loss_fwd - (-) lossvarKL - (-) lossvarKL_fwd
        loss = loss_vae - loss_fwd #- loss_varKL_vae - loss_varKL_fwd

        loss_varKL_v = -loss_varKL_vae
        loss_varKL_f = -loss_varKL_fwd

        loss.backward(retain_graph=True)
        loss_varKL_v.backward()
        loss_varKL_f.backward()

        train_loss += loss.item()
        train_loss_vae += loss_vae.item()
        train_loss_varKL_vae += loss_varKL_v.item()
        train_loss_fwd += -loss_fwd.item()
        train_loss_varKL_fwd += loss_varKL_f.item()

        optimizer.step()
        optimizer_var1.step()
        optimizer_var2.step()

    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / nr_data))
    print('====> Epoch: {} Average VAE loss: {:.4f}'.format(epoch, train_loss_vae / nr_data))
    print('====> Epoch: {} Average variational loss: {:.4f}'.format(epoch, train_loss_varKL_vae / nr_data))
    print('====> Epoch: {} Average FWD loss: {:.4f}'.format(epoch, train_loss_fwd / nr_data))
    print('====> Epoch: {} Average FWD variational loss: {:.4f}'.format(epoch, train_loss_varKL_fwd / nr_data))

    # print('====> Epoch: %d - noise: %.3f - noise_fwd: %.3f' % (
    #     epoch,
    #     model.AE_DKL.likelihood.noise.item(),
    #     model.fwd_model_DKL.likelihood.noise.item(),
    # ))

    print('====> task noise AE', model.AE_DKL.likelihood.raw_task_noises)
    print('====> task noise FW', model.fwd_model_DKL.likelihood.raw_task_noises)


def train_StochasticVAE(epoch, batch_size, nr_data, train_loader, model, optimizer, beta=1):

    model.train()

    train_loss = 0
    train_loss_vae = 0
    train_loss_fwd = 0

    for i in range(int(nr_data/batch_size)):
        data = train_loader.sample_batch(batch_size)

        obs = torch.from_numpy(data['obs1']).permute(0, 3, 1, 2).cuda()
        a = torch.from_numpy(data['acts']).cuda()
        next_obs = torch.from_numpy(data['obs2']).permute(0, 3, 1, 2).cuda()

        optimizer.zero_grad()

        mu_x, std_x, _, _, _, mu_target, std_target, _, mu_next, std_next, _ = model(obs, a, next_obs)

        # compute vae loss and forward model loss (KL divergence)
        loss_vae = loss_negloglikelihood(mu_x, obs, torch.square(std_x), dim=3)
        loss_fwd = - beta * kl_divergence_balance(mu_target, torch.square(std_target), mu_next, torch.square(std_next),
                                           alpha=0.9, dim=1)

        loss_t = loss_vae - loss_fwd

        loss_t.backward()

        train_loss += loss_t.item()
        train_loss_vae += loss_vae.item()
        train_loss_fwd += loss_fwd.item()

        optimizer.step()

    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / nr_data))
    print('====> Epoch: {} Average VAE loss: {:.4f}'.format(epoch, train_loss_vae / nr_data))
    print('====> Epoch: {} Average FWD loss: {:.4f}'.format(epoch, train_loss_fwd / nr_data))
