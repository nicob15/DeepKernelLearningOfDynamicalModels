import torch.nn.functional as F
import torch
import torch.distributions as td

# return reconstruction error + KL divergence losses
def loss_function_VAE(recon_x, x, mu, log_var, beta=1):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE + beta*KLD

# return reconstruction error + KL divergence losses
def loss_function_DKL(recon_x, x, mu, var, beta=1):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + torch.log(var) - mu.pow(2) - var)
    return BCE + beta*KLD

def loss_forward_model(mu1, logvar1, mu2, logvar2, beta=1, batch_size=256):

    #wasserstain 2

    var1 = torch.exp(logvar1)
    var2 = torch.exp(logvar2)
    var1_m = torch.eye(var1.shape[1])
    var1_m = var1_m.unsqueeze(0)
    var1_m = var1_m.repeat(batch_size, 1, 1)
    var1_sqrt = var1_m.cuda() * var1.unsqueeze(1)
    var1_m = torch.square(var1_m.cuda() * var1.unsqueeze(1))
    var2_m = torch.eye(var2.shape[1])
    var2_m = var2_m.unsqueeze(0)
    var2_m = var2_m.repeat(batch_size, 1, 1)
    var2_m = torch.square(var2_m.cuda() * var2.unsqueeze(1))

    term1 = torch.square(torch.linalg.vector_norm((mu1 - mu2), ord=2, keepdim=True, dim=1))
    term2 = (var1_m + var2_m - 2 * (
        (torch.sqrt(var1_sqrt * var2_m * var1_sqrt + 1e-8)))).diagonal(offset=0, dim1=-1, dim2=-2).sum(-1).unsqueeze(1)

    Wass = torch.sum(term1 + term2)

    return beta*Wass

def kl_divergence(mu_1, var_1, mu_2, var_2, beta=1.0, dim=1):
    p = td.Independent(td.Normal(mu_1, torch.sqrt(var_1)), dim)
    q = td.Independent(td.Normal(mu_2, torch.sqrt(var_2)), dim)
    div = td.kl_divergence(p, q)
    div = torch.max(div, div.new_full(div.size(), 3))
    return beta*torch.mean(div)

def kl_divergence_balance(mu_1, var_1, mu_2, var_2, alpha=0.8, beta=1.0, dim=1):
    p = td.Independent(td.Normal(mu_1, torch.sqrt(var_1)), dim)
    p_stop_grad = td.Independent(td.Normal(mu_1.detach(), torch.sqrt(var_1.detach())), dim)
    q = td.Independent(td.Normal(mu_2, torch.sqrt(var_2)), dim)
    q_stop_grad = td.Independent(td.Normal(mu_2.detach(), torch.sqrt(var_2.detach())), dim)
    div = alpha * td.kl_divergence(p_stop_grad, q) + (1 - alpha) * td.kl_divergence(p, q_stop_grad)
    div = torch.max(div, div.new_full(div.size(), 3))
    return beta*torch.mean(div)

lossNLL = torch.nn.GaussianNLLLoss()

def loss_negloglikelihood(mu, target, var, dim):
    #print('max mean', torch.max(mu))
    #print('min mean', torch.min(mu))
    #print('max var', torch.max(var))
    #print('min var', torch.min(var))
    #print('max std', torch.max(torch.sqrt(var)))
    #print('min std', torch.min(torch.sqrt(var)))
    normal_dist = torch.distributions.Independent(torch.distributions.Normal(mu, var), dim)

    return -torch.mean(normal_dist.log_prob(target))

    #return torch.mean(neg_log_likelihood)#lossNLL(mu, target, var)#

def loss_reward_model():
    return torch.nn.GaussianNLLLoss(reduction='sum')

def loss_forward_model_DKL(mu1, var1, mu2, var2, beta=1, batch_size=256):

    #wasserstain 2

    var1_m = torch.eye(var1.shape[1])
    var1_m = var1_m.unsqueeze(0)
    var1_m = var1_m.repeat(batch_size, 1, 1)
    var1_sqrt = var1_m.cuda() * var1.unsqueeze(1)
    var1_m = torch.square(var1_m.cuda() * var1.unsqueeze(1))
    var2_m = torch.eye(var2.shape[1])
    var2_m = var2_m.unsqueeze(0)
    var2_m = var2_m.repeat(batch_size, 1, 1)
    var2_m = torch.square(var2_m.cuda() * var2.unsqueeze(1))

    term1 = torch.square(torch.linalg.vector_norm((mu1 - mu2), ord=2, keepdim=True, dim=1))
    term2 = (var1_m + var2_m - 2 * (
        (torch.sqrt(var1_sqrt * var2_m * var1_sqrt + 1e-8)))).diagonal(offset=0, dim1=-1, dim2=-2).sum(-1).unsqueeze(1)

    Wass = torch.sum(term1 + term2)

    return beta*Wass