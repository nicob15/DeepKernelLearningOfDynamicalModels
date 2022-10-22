import gpytorch
import torch
from Losses import loss_negloglikelihood, kl_divergence, kl_divergence_balance
from utils import logvar2var

def beta_scheduler(epoch, max_epoch, R=0.5, M=1):
    tau = ((epoch) % (max_epoch / M)) / (max_epoch / M)
    if tau <= R:
        beta = 2*(epoch % (max_epoch / M)) / (max_epoch / M)
    else:
        beta = 1
    return beta

def train_DKL(epoch, batch_size, nr_data, train_loader, model, likelihood, likelihood_fwd, optimizer,
              max_epoch, variational_kl_term, variational_kl_term_fwd, k1=1, k2=2, optimizer_var=0):
    mll = gpytorch.mlls.VariationalELBO(likelihood_fwd, model.fwd_model_DKL.gp_layer_2, num_data=nr_data,
                                        combine_terms=False)
    #mll2 = gpytorch.mlls.VariationalELBO(likelihood_rew, model.rew_model_DKL.gp_layer_3, num_data=nr_data,
    #                                     combine_terms=False)

    model.train()
    likelihood.train()
    likelihood_fwd.train()
    #likelihood_rew.train()
    train_loss = 0
    train_lossVAE = 0
    train_lossVAE_next = 0

    train_lossvarKL = 0
    train_loss_fwd = 0
    train_lossvarKL_fwd = 0
    train_loss_rew = 0
    train_lossvarKL_rew = 0
    for i in range(int(nr_data/batch_size)):
        data = train_loader.sample_batch(batch_size)

        obs = torch.from_numpy(data['obs1']).permute(0, 3, 1, 2).cuda()
        a = torch.from_numpy(data['acts']).cuda()
        next_obs = torch.from_numpy(data['obs2']).permute(0, 3, 1, 2).cuda()
        #r = torch.from_numpy(data['rews']).view(-1, 1).cuda()

        #variational_ngd_optimizer.zero_grad()
        optimizer.zero_grad()
        optimizer_var.zero_grad()
        mu_x, var_x, mu, var, z, dist, mu_target, var_target, res_target, mu_next, var_next, res_fwd, mu_r, var_r, res_rew = model(obs, a, next_obs, likelihood)

        var_f = var

        ###################### sample multiple times (e.g. 2) from the likelihood and take the mean #######
        #z = likelihood(dist).rsample(sample_shape=torch.Size([2])).mean(0)

        #z = likelihood(dist).rsample(sample_shape=torch.Size([8])).mean(0)
        z = likelihood(dist).rsample()
        var = likelihood(dist).variance

        mu_x, var_x = model.decoder(z)

        #z_target = res_target.rsample(sample_shape=torch.Size([8])).mean(0)
        z_target = res_target.rsample()

        z_next = likelihood_fwd(res_fwd).rsample()

        mu_x_next, var_x_next = model.decoder(z_next)

            #res_fwd, mu_fwd, var_fwd, z_fwd = model.fwd_model_DKL(z, a)
            #res_rew, mu_rew, var_rew, z_rew = model.rew_model_DKL(z, a)

            #res_fwd = likelihood_fwd(res_fwd)
            #res_rew = likelihood_rew(res_rew)

        loss_vae = k1 * loss_negloglikelihood(mu_x, obs, var_x, dim=3)
        #loss_vae_next = loss_negloglikelihood(mu_x_next, next_obs, var_x_next, dim=3)
        # + kl_divergence(mu, var_f, torch.zeros_like(mu), torch.ones_like(var), dim=1)
            #loss_fwd = loss_negloglikelihood(mu_next, z_target, var_next, dim=1)
            #loss_fwd = kl_divergence(mu_target, var_target, mu_next, var_next, dim=1)
            #loss_rew = loss_negloglikelihood(mu_r, torch.from_numpy(data['rews']).view(-1, 1).cuda(), var_r, dim=1) #+ torch.norm(var_r)

        c = 1
        lossvarKL = c*variational_kl_term(beta=1)

            # loss = (-) loss_vae + (+) loss_fwd + (-) loss_rew - (-) lossvarKL - (-) lossvarKL_fwd - (-) lossvarKL_rew
            # coefficients =  1 - 1 - 1 - 1 - 1 - 1 (first 3 from dreamer paper, while last 3 may need tuning)

        _, lossvarKL_fwd, log_prior_fwd = mll(res_fwd, z_target)

        lossvarKL_fwd = c * variational_kl_term_fwd(beta=1)
        #loss_fwd = - kl_divergence_balance(mu_target, var_target, likelihood_fwd(res_fwd).mean, likelihood_fwd(res_fwd).variance, alpha=0.9, dim=1)
        #loss_fwd = - kl_divergence_balance(mu_target, var_target, res_fwd.mean,
        #                                   res_fwd.variance, alpha=0.9, dim=1)
        loss_fwd = - kl_divergence_balance(likelihood(res_target).mean, likelihood(res_target).variance, likelihood_fwd(res_fwd).mean,
                                           likelihood_fwd(res_fwd).variance, alpha=0.9, dim=1)
            #loss_fwd = - kl_divergence(mu_target, var_target, likelihood_fwd(res_fwd).mean, likelihood_fwd(res_fwd).variance, dim=1)

        #loss_rew, lossvarKL_rew, log_prior_rew = mll2(res_rew, r)

            # loss = (-) loss_vae + (+) loss_fwd + (-) loss_rew - (-) lossvarKL - (-) lossvarKL_fwd - (-) lossvarKL_rew
            # coefficients =  1 - 1 - 1 - 1 - 1 - 1 (first 3 from dreamer paper, while last 3 may need tuning)
        beta = 1#beta_scheduler(epoch, max_epoch)

        loss_fwd = k2 * beta * loss_fwd
        lossvarKL_fwd = beta * lossvarKL_fwd
        log_prior_fwd = 0.0*beta * log_prior_fwd
        #loss_rew = 0.0*beta * loss_rew
        #lossvarKL_rew = 0.0*beta * lossvarKL_rew
        #log_prior_rew = 0.0*beta * log_prior_rew

        #loss_vae_next = 0.0*loss_vae_next
            # loss = loss_vae - lossvarKL + beta * loss_fwd - lossvarKL - beta * lossvarKL_fwd #- beta * lossvarKL_rew + beta * loss_rew #- log_prior
        loss = loss_vae - lossvarKL - loss_fwd - lossvarKL_fwd - log_prior_fwd #+ loss_vae_next #- loss_rew + lossvarKL_rew - log_prior_rew


        loss.backward()

        train_loss += loss.item()

        train_lossVAE += loss_vae.item()
        #train_lossVAE_next += loss_vae_next.item()

        train_lossvarKL += -lossvarKL.item()
        train_loss_fwd += -loss_fwd.item()
        train_lossvarKL_fwd += -lossvarKL_fwd.item()
        #train_loss_rew += -loss_rew.item()
        #train_lossvarKL_rew += +lossvarKL_rew.item()


            #variational_ngd_optimizer.step()
        optimizer.step()
            #optimizer.step(loss)
        optimizer_var.step()

    for param_name, param in model.gp_layer.named_parameters():
        print(param_name)
        print(param)

    for param_name, param in model.fwd_model_DKL.gp_layer_2.named_parameters():
        print(param_name)
        print(param)

        #for param_name, param in model.rew_model_DKL.gp_layer_3.named_parameters():
        #    print(param_name)
        #    print(param)

    print('Iter %d - noise: %.3f - noise_fwd: %.3f' % (
        epoch,
        likelihood.noise.item(),
        likelihood_fwd.noise.item(),
        #likelihood_rew.noise.item()
    ))

    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / nr_data))
    print('====> Epoch: {} Average VAE loss: {:.4f}'.format(epoch, train_lossVAE / nr_data))
    #print('====> Epoch: {} Average VAE loss: {:.4f}'.format(epoch, train_lossVAE_next / nr_data))
    print('====> Epoch: {} Average variational loss: {:.4f}'.format(epoch, train_lossvarKL / nr_data))
    print('====> Epoch: {} Average FWD loss: {:.4f}'.format(epoch, train_loss_fwd / nr_data))
    print('====> Epoch: {} Average FWD variational loss: {:.4f}'.format(epoch, train_lossvarKL_fwd / nr_data))
        #print('====> Epoch: {} Average REW loss: {:.4f}'.format(epoch, train_loss_rew / nr_data))
        #print('====> Epoch: {} Average REW variational loss: {:.4f}'.format(epoch, train_lossvarKL_rew))

def train_GP(epoch, batch_size, nr_data, train_loader, model, likelihood, likelihood_fwd, likelihood_rew, optimizer_gp,
              max_epoch, variational_kl_term, variational_kl_term_fwd, variational_kl_term_rew):
    mll = gpytorch.mlls.VariationalELBO(likelihood_fwd, model.fwd_model_DKL.gp_layer_2, num_data=nr_data,
                                        combine_terms=False)
    mll2 = gpytorch.mlls.VariationalELBO(likelihood_rew, model.rew_model_DKL.gp_layer_3, num_data=nr_data,
                                         combine_terms=False)

    model.train()
    likelihood.train()
    likelihood_fwd.train()
    likelihood_rew.train()
    train_loss = 0
    train_lossVAE = 0
    train_lossvarKL = 0
    train_loss_fwd = 0
    train_lossvarKL_fwd = 0
    train_loss_rew = 0
    train_lossvarKL_rew = 0
    for i in range(int(nr_data/batch_size)):
        data = train_loader.sample_batch(batch_size)

        obs = torch.from_numpy(data['obs1']).permute(0, 3, 1, 2).cuda()
        a = torch.from_numpy(data['acts']).cuda()
        next_obs = torch.from_numpy(data['obs2']).permute(0, 3, 1, 2).cuda()
        r = torch.from_numpy(data['rews']).view(-1, 1).cuda()

        optimizer_gp.zero_grad()
        mu_x, var_x, mu, var, z, dist, mu_target, var_target, res_target, mu_next, var_next, res_fwd, mu_r, var_r, res_rew = model(obs, a, next_obs, likelihood)

        var_f = var
        z = likelihood(dist).rsample()
        var = likelihood(dist).variance

        mu_x, var_x = model.decoder(z)

        z_target = res_target.rsample()

        loss_vae = loss_negloglikelihood(mu_x, obs, var_x, dim=3)

        c = 1
        lossvarKL = c*variational_kl_term(beta=1)


        _, lossvarKL_fwd, log_prior_fwd = mll(res_fwd, z_target)

        loss_fwd = - kl_divergence_balance(mu_target, var_target, likelihood_fwd(res_fwd).mean, likelihood_fwd(res_fwd).variance, dim=1)
        #loss_fwd = - kl_divergence_balance(likelihood(res_target).mean, likelihood(res_target).variance,
        #                                   likelihood_fwd(res_fwd).mean,
        #                                   likelihood_fwd(res_fwd).variance, dim=1)

        loss_rew, lossvarKL_rew, log_prior_rew = mll2(res_rew, r)

        beta = 1

        loss_fwd = beta * loss_fwd
        lossvarKL_fwd = beta * lossvarKL_fwd
        log_prior_fwd = beta * log_prior_fwd
        loss_rew = 0.0*beta * loss_rew
        lossvarKL_rew = 0.0*beta * lossvarKL_rew
        log_prior_rew = 0.0*beta * log_prior_rew

        loss = loss_vae - lossvarKL - loss_fwd + lossvarKL_fwd - log_prior_fwd - loss_rew + lossvarKL_rew - log_prior_rew

        loss.backward()

        train_loss += loss.item()

        train_lossVAE += loss_vae.item()
        train_lossvarKL += -lossvarKL.item()
        train_loss_fwd += -loss_fwd.item()
        train_lossvarKL_fwd += lossvarKL_fwd.item()
        train_loss_rew += -loss_rew.item()
        train_lossvarKL_rew += +lossvarKL_rew.item()

        optimizer_gp.step()

    for param_name, param in model.gp_layer.named_parameters():
        print(param_name)
        print(param)

    for param_name, param in model.fwd_model_DKL.gp_layer_2.named_parameters():
        print(param_name)
        print(param)

    print('Iter %d - noise: %.3f - noise_fwd: %.3f' % (
        epoch,
        likelihood.noise.item(),
        likelihood_fwd.noise.item()
    ))

    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / nr_data))
    print('====> Epoch: {} Average VAE loss: {:.4f}'.format(epoch, train_lossVAE / nr_data))
    print('====> Epoch: {} Average variational loss: {:.4f}'.format(epoch, train_lossvarKL / nr_data))
    print('====> Epoch: {} Average FWD loss: {:.4f}'.format(epoch, train_loss_fwd / nr_data))
    print('====> Epoch: {} Average FWD variational loss: {:.4f}'.format(epoch, train_lossvarKL_fwd / nr_data))

def train_NN(epoch, batch_size, nr_data, train_loader, model, likelihood, likelihood_fwd, likelihood_rew, optimizer_nn,
              max_epoch, variational_kl_term, variational_kl_term_fwd, variational_kl_term_rew):
    mll = gpytorch.mlls.VariationalELBO(likelihood_fwd, model.fwd_model_DKL.gp_layer_2, num_data=nr_data,
                                        combine_terms=False)
    mll2 = gpytorch.mlls.VariationalELBO(likelihood_rew, model.rew_model_DKL.gp_layer_3, num_data=nr_data,
                                         combine_terms=False)

    model.train()
    likelihood.train()
    likelihood_fwd.train()
    likelihood_rew.train()
    train_loss = 0
    train_lossVAE = 0
    train_lossvarKL = 0
    train_loss_fwd = 0
    train_lossvarKL_fwd = 0
    train_loss_rew = 0
    train_lossvarKL_rew = 0
    for i in range(int(nr_data/batch_size)):
        data = train_loader.sample_batch(batch_size)

        obs = torch.from_numpy(data['obs1']).permute(0, 3, 1, 2).cuda()
        a = torch.from_numpy(data['acts']).cuda()
        next_obs = torch.from_numpy(data['obs2']).permute(0, 3, 1, 2).cuda()
        r = torch.from_numpy(data['rews']).view(-1, 1).cuda()

        optimizer_nn.zero_grad()
        mu_x, var_x, mu, var, z, dist, mu_target, var_target, res_target, mu_next, var_next, res_fwd, mu_r, var_r, res_rew = model(obs, a, next_obs)

        var_f = var
        z = likelihood(dist).rsample()
        var = likelihood(dist).variance

        mu_x, var_x = model.decoder(z)

        z_target = res_target.rsample()

        loss_vae = loss_negloglikelihood(mu_x, obs, var_x, dim=3)

        c = 1
        lossvarKL = c*variational_kl_term(beta=1)


        _, lossvarKL_fwd, log_prior_fwd = mll(res_fwd, z_target)

        loss_fwd = - kl_divergence_balance(mu_target, var_target, likelihood_fwd(res_fwd).mean, likelihood_fwd(res_fwd).variance, dim=1)

        loss_rew, lossvarKL_rew, log_prior_rew = mll2(res_rew, r)

        beta = 1

        loss_fwd = beta * loss_fwd
        lossvarKL_fwd = beta * lossvarKL_fwd
        log_prior_fwd = beta * log_prior_fwd

        loss = loss_vae - lossvarKL - loss_fwd + lossvarKL_fwd - log_prior_fwd - loss_rew + lossvarKL_rew - log_prior_rew

        loss.backward()

        train_loss += loss.item()

        train_lossVAE += loss_vae.item()
        train_lossvarKL += -lossvarKL.item()
        train_loss_fwd += -loss_fwd.item()
        train_lossvarKL_fwd += lossvarKL_fwd.item()
        train_loss_rew += -loss_rew.item()
        train_lossvarKL_rew += +lossvarKL_rew.item()

        optimizer_nn.step()

    for param_name, param in model.gp_layer.named_parameters():
        print(param_name)
        print(param)

    for param_name, param in model.fwd_model_DKL.gp_layer_2.named_parameters():
        print(param_name)
        print(param)

    print('Iter %d - noise: %.3f - noise_fwd: %.3f' % (
        epoch,
        likelihood.noise.item(),
        likelihood_fwd.noise.item()
    ))

    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / nr_data))
    print('====> Epoch: {} Average VAE loss: {:.4f}'.format(epoch, train_lossVAE / nr_data))
    print('====> Epoch: {} Average variational loss: {:.4f}'.format(epoch, train_lossvarKL / nr_data))
    print('====> Epoch: {} Average FWD loss: {:.4f}'.format(epoch, train_loss_fwd / nr_data))
    print('====> Epoch: {} Average FWD variational loss: {:.4f}'.format(epoch, train_lossvarKL_fwd / nr_data))

def train_StochasticVAE(epoch, batch_size, nr_data, train_loader, model, optimizer, max_epoch):
    model.train()
    train_loss = 0
    train_loss_vae = 0
    train_loss_fwd = 0
    #train_loss_rew = 0
    for i in range(int(nr_data/batch_size)):
        data = train_loader.sample_batch(batch_size)
        optimizer.zero_grad()

        mu_x, std_x, mu, std, z, mu_target, std_target, _, mu_next, std_next, _, mu_r, std_r, _ = model(
                                                                   torch.from_numpy(data['obs1']).permute(0, 3, 1, 2).cuda(),
                                                                   torch.from_numpy(data['acts']).cuda(),
                                                                   torch.from_numpy(data['obs2']).permute(0, 3, 1, 2).cuda()
                                                                   )

        loss_vae = loss_negloglikelihood(mu_x, torch.from_numpy(data['obs1']).permute(0, 3, 1, 2).cuda(), torch.square(std_x), dim=3)
        loss_fwd = kl_divergence(mu_target, torch.square(std_target), mu_next, torch.square(std_next), dim=1)
        loss_rew = 0.0*loss_negloglikelihood(mu_r, torch.from_numpy(data['rews']).view(-1, 1).cuda(), torch.square(std_r), dim=1)

        beta = 1#beta_scheduler(epoch, max_epoch)

        loss_t = loss_vae + beta * loss_fwd #+ loss_rew

        loss_t.backward()
        train_loss += loss_t.item()
        train_loss_vae += loss_vae.item()
        train_loss_fwd += loss_fwd.item()
        #train_loss_rew += loss_rew.item()
        optimizer.step()

    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / nr_data))
    print('====> Epoch: {} Average VAE loss: {:.4f}'.format(epoch, train_loss_vae / nr_data))
    print('====> Epoch: {} Average FWD loss: {:.4f}'.format(epoch, train_loss_fwd / nr_data))
    #print('====> Epoch: {} Average REW loss: {:.4f}'.format(epoch, train_loss_rew / nr_data))



###### OLD #######
def train_VAE(epoch, batch_size, nr_data, train_loader, model, optimizer, loss_function, loss_forward_model,
              loss_reward_model):
    model.train()
    train_loss = 0
    train_loss_vae = 0
    train_loss_fwd = 0
    train_loss_rew = 0
    for i in range(int(nr_data/batch_size)):
        data = train_loader.sample_batch(batch_size)
        optimizer.zero_grad()

        recon_batch, mu, log_var, z, mu_target, log_var_target, mu_next, log_var_next, mu_r, log_var_r = model(
                                                                   torch.from_numpy(data['obs1']).permute(0, 3, 1, 2).cuda(),
                                                                   torch.from_numpy(data['acts']).cuda(),
                                                                   torch.from_numpy(data['obs2']).permute(0, 3, 1, 2).cuda()
                                                                   )

        loss_vae = loss_function(recon_batch, torch.from_numpy(data['obs1']).permute(0, 3, 1, 2).cuda(), mu, log_var, beta=0)
        loss_fwd = loss_forward_model(mu_target, log_var_target, mu_next, log_var_next, beta=100, batch_size=batch_size)
        loss_rew = 100*loss_negloglikelihood(mu_r, torch.from_numpy(data['rews']).view(-1, 1).cuda(), logvar2var(log_var_r))

        loss_t = loss_vae + loss_fwd + loss_rew

        loss_t.backward()
        train_loss += loss_t.item()
        train_loss_vae += loss_vae.item()
        train_loss_fwd += loss_fwd.item()
        train_loss_rew += loss_rew.item()
        optimizer.step()

    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / nr_data))
    print('====> Epoch: {} Average VAE loss: {:.4f}'.format(epoch, train_loss_vae / nr_data))
    print('====> Epoch: {} Average FWD loss: {:.4f}'.format(epoch, train_loss_fwd / nr_data))
    print('====> Epoch: {} Average REW loss: {:.4f}'.format(epoch, train_loss_rew / nr_data))