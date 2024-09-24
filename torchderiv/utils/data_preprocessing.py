import torch
from torch import Tensor

def normalize(data: Tensor) -> Tensor:
    mean = torch.mean(data)
    std = torch.std(data)
    return (data - mean) / std

def denormalize(data: Tensor, mean: float, std: float) -> Tensor:
    return data * std + mean

def split_data(data: Tensor, train_ratio: float) -> (Tensor, Tensor):
    train_size = int(len(data) * train_ratio)
    train_data = data[:train_size]
    test_data = data[train_size:]
    return train_data, test_data