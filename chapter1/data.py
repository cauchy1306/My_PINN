import torch


def f(x):
    """目标函数：低频 + 高频叠加"""
    return torch.sin(2 * torch.pi * x) + 0.1 * torch.sin(50 * torch.pi * x)


def sample_data(n_samples=200, x_range=(0.0, 1.0)):
    x = torch.linspace(x_range[0], x_range[1], n_samples).unsqueeze(1)
    y = f(x)
    return x, y
