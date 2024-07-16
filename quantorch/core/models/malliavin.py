import torch
import torch.autograd as autograd

class MalliavinGreeks:
    def __init__(self, model, spot, strike, rate, volatility, maturity, num_paths=10000, num_steps=252):
        self.model = model
        self.S0 = spot
        self.K = strike
        self.r = rate
        self.sigma = volatility
        self.T = maturity
        self.num_paths = num_paths
        self.num_steps = num_steps
        self.dt = self.T / self.num_steps

    def generate_paths(self):
        dW = torch.randn(self.num_paths, self.num_steps) * torch.sqrt(self.dt)
        W = torch.cumsum(dW, dim=1)
        t = torch.linspace(0, self.T, self.num_steps + 1)
        S = self.S0 * torch.exp((self.r - 0.5 * self.sigma**2) * t + self.sigma * W)
        return S

    def payoff(self, S):
        return torch.max(S[:, -1] - self.K, torch.tensor(0.))

    def delta(self):
        S = self.generate_paths()
        payoff = self.payoff(S)
        
        H = torch.sum(payoff * (torch.log(S[:, -1] / self.S0) - (self.r - 0.5 * self.sigma**2) * self.T) 
                      / (self.sigma**2 * self.T * self.S0))
        
        return H / self.num_paths

    def gamma(self):
        S = self.generate_paths()
        payoff = self.payoff(S)
        
        log_return = torch.log(S[:, -1] / self.S0)
        drift = (self.r - 0.5 * self.sigma**2) * self.T
        
        H = torch.sum(payoff * ((log_return - drift)**2 - self.sigma**2 * self.T) 
                      / (self.sigma**4 * self.T**2 * self.S0**2))
        
        return H / self.num_paths

    def vega(self):
        S = self.generate_paths()
        payoff = self.payoff(S)
        
        log_return = torch.log(S[:, -1] / self.S0)
        drift = (self.r - 0.5 * self.sigma**2) * self.T
        
        H = torch.sum(payoff * (self.sigma * self.T - (log_return - drift) / self.sigma) 
                      / (self.sigma**2 * self.T))
        
        return H / self.num_paths

    def calculate_all_greeks(self):
        return {
            'delta': self.delta().item(),
            'gamma': self.gamma().item(),
            'vega': self.vega().item()
        }