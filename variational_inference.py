from abc import ABC
import torch
from gpytorch.mlls.marginal_log_likelihood import MarginalLogLikelihood

class _ApproximateMarginalLogLikelihood(MarginalLogLikelihood, ABC):

    def __init__(self, likelihood, model, num_data, beta=1.0):
        super().__init__(likelihood, model)
        self.num_data = num_data
        self.beta = beta

    def forward(self, **kwargs):
        # Get KL term
        kl_divergence = self.model.variational_strategy.kl_divergence().div(self.num_data / self.beta)

        # Log prior term
        log_prior = torch.zeros_like(kl_divergence)
        for name, module, prior, closure, _ in self.named_priors():
            log_prior.add_(prior.log_prob(closure(module)).sum().div(self.num_data))

        return - kl_divergence + log_prior

class VariationalKL(_ApproximateMarginalLogLikelihood):

    def forward(self, **kwargs):
        return super().forward(**kwargs)