from collections import OrderedDict
from numbers import Number
import torch
class Data(torch.Tensor):
    @staticmethod
    def __new__(cls, *args, **kwargs):
        return super().__new__(cls, *args, **kwargs)
    
    def __init__(self, x):
        super().__init__()
    
    @classmethod
    def zeros(cls, *sizes, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False):
        return cls(torch.zeros(*sizes, out=out, dtype=dtype, layout=layout, device=device))
    
    @classmethod
    def ones(cls, *sizes, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False):
        return cls(torch.ones(*sizes, out=out, dtype=dtype, layout=layout, device=device))
    
    def __repr__(self):
        return f"{self.__class__.__name__}{repr(self.data)[6:]}"

class Parameter(Data):
    
    def __init__(self, x):
        self.requires_grad = True



def fails_constraints(*conditions):

    for each in conditions:
        if not torch.all(each):
            return True
    else:
        return False


def num_to_tensor(x):
    return Data([x]) if isinstance(x, Number) else x

class Distribution:
    def logp(self):
        raise NotImplementedError
        
    def __call__(self):
        return self.logp().squeeze()

class Normal(Distribution):
    def __init__(self, x, mu=0., sig=1.):
        self.x = x
        self.mu = num_to_tensor(mu)
        self.sig = num_to_tensor(sig)

    def logp(self):
        if fails_constraints(self.sig >= 0):
            return -inf
        
        return torch.sum(-torch.log(self.sig) - (self.x - self.mu)**2/(2*self.sig**2))
    
    def __repr__(self):
        return f"Normal(mu={self.mu}, sigma={self.sig})"
    
class Uniform(Distribution):
    def __init__(self, x, low=0., high=1.):
        self.x = x
        self.low = num_to_tensor(low)
        self.high = num_to_tensor(high)

    def logp(self):
        if fails_constraints(self.x >= self.low, self.x <= self.high):
            return -inf
    
        size = 1
        for dim_size in self.x.shape:
            size *= dim_size

        return -size * torch.log(self.high - self.low)
    
    def __repr__(self):
        return f"Uniform(low={self.low}, high={self.high})"
    
class Model(Distribution):
    
    def __init__(self):
        self.parameters = {}
    
    def iter_params(self):
        for name, param  in self.parameters.items():
            yield name, param
    
    def update(self, param_dict):
       
        for name, param in param_dict.items():
            self.parameters[name][:] = param
        
    def __setattr__(self, name, value):
        if isinstance(value, Parameter):
            self.parameters[name] = value
        
        super().__setattr__(name, value)

def dist_sum(*distributions):
    return sum(x.logp() for x in distributions)

class BEST(Model):
    def __init__(self, group1, group2):
        super().__init__()
        
        self.group1 = group1
        self.group2 = group2
        
        self.mu = Parameter([0., 0.])
        self.sigma = Parameter([1., 1.])
        
    def logp(self):
        llh1 = Normal(self.group1, mu=self.mu[0], sig=self.sigma[0])
        llh2 = Normal(self.group2, mu=self.mu[1], sig=self.sigma[1])
        
        prior_mu = Normal(self.mu, mu=0., sig=1.)
        prior_sig = Uniform(self.sigma, 0., 100.)
        
        return dist_sum(llh1, llh2, prior_mu, prior_sig)
    
from torch import optim
def find_MAP(model, iters=2000):
    optimizer = optim.Adam((param for _, param in model.iter_params()), lr=0.003)
    for step in range(iters):
        optimizer.zero_grad()

        logp = -model()
        logp.backward()
        optimizer.step()


class Chain:
    def __init__(self, samples, **kwargs):
        self.fields = OrderedDict()
        
        prev_size = 0
        for name, size in kwargs.items():
            self.fields.update({name:(prev_size, prev_size + size)})
            prev_size = size
            
        total_size = sum(size for size in kwargs.values())
        self.data = np.zeros((samples, total_size))
        
    def __getitem__(self, name):
        return self.data[name]
        
    def __setitem__(self, name, value):
        self.data[name] = value
        
    def __getattr__(self, name):
        if name in self.fields:
            field = self.fields[name]
            return self.data[:, field[0]:field[1]]
    
    def __len__(self):
        return len(self.data)
        
    def __repr__(self):
        return f"Chain({[field for field in self.fields]})"

class Sampler:
    def __init__(self, model):
        self.model = model
    
    def step(self):
       
        raise NotImplementedError
    
    def sample(self, num, burn=1, thin=1):
        param_shapes = {name: param.shape[0] for name, param in self.model.iter_params()}
        self.chain = Chain(num, **param_shapes)
        
        for ii in range(num):
            sample = self.step()
            self.chain[ii] = sample.data.numpy()
        
        return self.chain
    
class Metropolis(Sampler):
    def __init__(self, model, scale=1, tune_interval=100):
        self.model = model
        self.scale = scale
        self.proposal_dist = torch.distributions.normal.Normal(0, self.scale)
        
        self.tune_interval = tune_interval
        self._steps_until_tune = tune_interval
        self._accepted = 0
        self._sampled = 0
    
    def step(self):
        
        model = self.model
        logp = model()
        state = {name: param.clone() for name, param in model.iter_params()}
        
        
        for name in state:
            self.proposal_dist.loc = state[name]
            new_state = self.proposal_dist.sample(state[name].shape)
            model.update({name: new_state})
        
        new_logp = model()
        
        
        if not accept(logp, new_logp):
            model.update(state)
            
        else:
            self._accepted += 1
            
        
        self._sampled += 1
        self._steps_until_tune -= 1
        if self._steps_until_tune == 0:
            self.scale = tune(self.scale, self.acceptance)
            self.proposal_dist.scale = torch.tensor(self.scale)
            self._steps_until_tune = self.tune_interval
        
        return torch.cat([param.view(-1) for _, param in model.iter_params()])
    
    @property
    def acceptance(self):
        return self._accepted/self._sampled
    
    def __repr__(self):
        return 'Metropolis-Hastings sampler'

def accept(old_logp, new_logp):
    diff_logp = new_logp - old_logp
    if torch.isfinite(diff_logp) and torch.log(torch.rand(1)) < diff_logp:
        return True
    else:
        return False
    
def tune(scale, acceptance):

 
    if acceptance < 0.001:
        scale *= 0.1
    elif acceptance < 0.05:
        scale *= 0.5
    elif acceptance < 0.2:
        scale *= 0.9
    elif acceptance > 0.95:
        scale *= 10.0
    elif acceptance > 0.75:
        scale *= 2.0
    elif acceptance > 0.5:
        scale *= 1.1

    return scale

class Linear(Model):
    def __init__(self, features, outcomes):
        super().__init__()
        
        self.features = features
        self.outcomes = outcomes
        
        self.beta = Parameter.zeros(3, 1)
        
        self.sigma = Parameter([1.])
        
    def logp(self):
        predicted = torch.mm(self.features, self.beta)
        
        llh = Normal(self.outcomes, mu=predicted, sig=self.sigma)
        
        beta_prior = Uniform(self.beta, -100, 100)
        sig_prior = Uniform(self.sigma, 0., 100.)
        
        return dist_sum(llh, sig_prior, beta_prior)