import torch

def bayesian_update(prior: Tensor, likelihood: Tensor) -> Tensor:
    posterior = prior * likelihood
    posterior /= posterior.sum()
    return posterior

prior = torch.tensor([0.2, 0.3, 0.5])
likelihood = torch.tensor([0.6, 0.1, 0.3])
posterior = bayesian_update(prior, likelihood)
print(f'Posterior: {posterior}')