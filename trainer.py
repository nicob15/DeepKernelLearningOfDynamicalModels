import torch
from losses import loss_negloglikelihood, kl_divergence_balance

def train_DKL(epoch, batch_size, nr_data, train_loader, model, likelihood, likelihood_fwd, optimizer,
              variational_kl_term, variational_kl_term_fwd, k1=1, beta=1):

    model.train()
    likelihood.train()
    likelihood_fwd.train()

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

        mu_x, var_x, _, _, _, res, mu_target, var_target, res_target, _, _, res_fwd = model(obs, a, next_obs)

        # compute vae loss + variational inference
        # 1- sample from the encoder likelihood to obtain z
        # 2- reconstruct the measurements (Gaussian distribution) by decoding z
        # 3- compute the negative log likelihood loss for training encoder and decoder
        # 4- compute the variational inference loss to update GP variational hyperparameters of the encoder

        z = k1 * likelihood(res).rsample(sample_shape=torch.Size([1])).mean(0) #change torch.Size to n > 1 if you want to sample multiple times from the likelihood and then take the mean
        mu_x, var_x = model.decoder(z)
        loss_vae = loss_negloglikelihood(mu_x, obs, var_x, dim=3)
        loss_varKL_vae = variational_kl_term(beta=1)

        # compute forward model loss (KL divergence) + variational inference
        # 1- compute KL divergence between the target next state distribution and the next state distribution
        # 2- compute the variational inference loss to update GP variational hyperparameters of the forward model
        loss_fwd = - beta * kl_divergence_balance(likelihood(res_target).mean, likelihood(res_target).variance,
                                           likelihood_fwd(res_fwd).mean, likelihood_fwd(res_fwd).variance, alpha=0.9,
                                           dim=1)
        loss_varKL_fwd = variational_kl_term_fwd(beta=1)

        # loss = (-) loss_vae + (+) loss_fwd - (-) lossvarKL - (-) lossvarKL_fwd
        loss = loss_vae - loss_fwd - loss_varKL_vae - loss_varKL_fwd

        loss.backward()

        train_loss += loss.item()
        train_loss_vae += loss_vae.item()
        train_loss_varKL_vae += -loss_varKL_vae.item()
        train_loss_fwd += -loss_fwd.item()
        train_loss_varKL_fwd += -loss_varKL_fwd.item()

        optimizer.step()

    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / nr_data))
    print('====> Epoch: {} Average VAE loss: {:.4f}'.format(epoch, train_loss_vae / nr_data))
    print('====> Epoch: {} Average variational loss: {:.4f}'.format(epoch, train_loss_varKL_vae / nr_data))
    print('====> Epoch: {} Average FWD loss: {:.4f}'.format(epoch, train_loss_fwd / nr_data))
    print('====> Epoch: {} Average FWD variational loss: {:.4f}'.format(epoch, train_loss_varKL_fwd / nr_data))

    # print GP hyperparameters
    #for param_name, param in model.gp_layer.named_parameters():
    #    print(param_name)
    #    print(param)

    #for param_name, param in model.fwd_model_DKL.gp_layer_2.named_parameters():
    #    print(param_name)
    #    print(param)

    print('====> Epoch: %d - noise: %.3f - noise_fwd: %.3f' % (
        epoch,
        likelihood.noise.item(),
        likelihood_fwd.noise.item(),
    ))


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

def train_ExactDKL(epochs, batch_size, nr_data, data, model, likelihood, likelihood_fwd, optimizer, k1=1, beta=1):

    model.train()
    likelihood.train()
    likelihood_fwd.train()

    train_loss = 0
    train_loss_vae = 0
    train_loss_fwd = 0

    for i in range(epochs):

        obs = torch.from_numpy(data['obs1']).permute(0, 3, 1, 2).cuda()
        a = torch.from_numpy(data['acts']).cuda()
        next_obs = torch.from_numpy(data['obs2']).permute(0, 3, 1, 2).cuda()

        optimizer.zero_grad()

        mu_x, var_x, _, _, _, res, mu_target, var_target, res_target, _, _, res_fwd = model(obs, a, next_obs)

        z = k1 * likelihood(res).rsample(sample_shape=torch.Size([1])).mean(0)
        mu_x, var_x = model.decoder(z)
        loss_vae = loss_negloglikelihood(mu_x, obs, var_x, dim=3)

        loss_fwd = - beta * kl_divergence_balance(likelihood(res_target).mean, likelihood(res_target).variance,
                                           likelihood_fwd(res_fwd).mean, likelihood_fwd(res_fwd).variance, alpha=0.9,
                                           dim=1)

        loss = loss_vae - loss_fwd

        loss.backward()

        train_loss += loss.item()
        train_loss_vae += loss_vae.item()
        train_loss_fwd += -loss_fwd.item()

        optimizer.step()

        print('====> Epoch: {} Average loss: {:.4f}'.format(i, train_loss / nr_data))
        print('====> Epoch: {} Average VAE loss: {:.4f}'.format(i, train_loss_vae / nr_data))
        print('====> Epoch: {} Average FWD loss: {:.4f}'.format(i, train_loss_fwd / nr_data))

        print('====> Epoch: %d - noise: %.3f - noise_fwd: %.3f' % (
              i,
              likelihood.noise.item(),
              likelihood_fwd.noise.item(),)
              )