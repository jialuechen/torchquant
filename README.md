<div align=center>
<img src="assets/quantorch-high-resolution-color-logo.png" width="45%" loc>
</div>

<div align=center>

[![python version](https://img.shields.io/badge/python-3.8+-brightgreen.svg)](https://github.com/jialuechen)
![PyPI](https://img.shields.io/pypi/v/0.0.1)
![PyPI - License](https://img.shields.io/pypi/l/quantorch)

# PyTorch-based Python Library for Quant Finance
</div>

## Introduction
QuanTorch is developed to empower quantitative research by providing modern machine learning computation and acceleration service. QuanTorch provides high-performance components leveraging the hardware acceleration support and automatic differentiation of PyTorch. QuanTorch supports foundational mathematical methods, mid-level methods, and specific pricing models, which is also an experimental light-weight alternative to QuantLib.

## Motivation
PyTorch provides two high-level features: 

* Tensor computing with strong acceleration via GPU
* Automatic Differentiation System

Quantorch makes use of these modern features on PyTorch library to build advanced stochastic models, high-performance pricing_models, PDE solvers and numerical methods.

## Installation
```
pip install --upgrade quantorch
```

## Example
* Refined Black-Scholes-Merton Framework
```
from quantorch.core.optionPricer import OptionPricer
from torch import Tensor

optionType='european',optionDirection='put',\
spot=Tensor([100,95]),strike=Tensor([120,80]),\
expiry=Tensor([1.0,0.75]),volatility=Tensor([0.1,0.3]),\
rate=Tensor([0.01,0.05]),dividend=Tensor([0.01,0.02]),\

pricingModel='BSM'

# here we use GPU accleration as an example

if torch.cuda.is_available():
   device=torch.device("cuda")
   spot=spot.to(device)
   strike=strike.to(device)
   expiry=expiry.to(device)
   volatility=volatility.to(device)
   rate=rate.to(device)
   
OptionPricer.price(
    optionType,
    optionDirection,
    spot,
    strike,
    expiry,
    volatility,
    rate,
    dividend,
    pricingModel,
    device='GPU'
    )
```
* Binomial Tree Option Pircing Model
* Root-Finding Algorithms
* Random Walk
```
import torch
from quantorch.models.rw import utils
from quantorch.models.rw import rw
import networkx as nx

g = nx.Graph()

g.add_edge("A","B")
g.add_edge("A","C")
g.add_edge("B","C")
g.add_edge("B","D")
g.add_edge("D","C")

row, col = utils.to_csr(g)
nodes = utils.nodes_tensor(g)

# using GPU
device="cuda"
row = row.to(device)
col = col.to(device)
nodes = nodes.to(device)

walks = rw.walk(row=row,col=col,target_nodes=nodes,p=1.0,q=1.0,walk_length=6,seed=10)
```

* Monte Carlo Simulation
* Risk Management (e.g., Greeks Calculation, Hedging)
* Bayesian Inference
* ... (More promising applications in quantitative finance)




## License

MIT License
