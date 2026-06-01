import torch


def f(x):
    """目标函数：低频 + 高频叠加"""
    return torch.sin(2 * torch.pi * x) + 0.1 * torch.sin(50 * torch.pi * x)


def sample_data(n_samples=200, x_range=(0.0, 1.0)):
    x = torch.linspace(x_range[0], x_range[1], n_samples).unsqueeze(1)
    y = f(x)
    return x, y


def fourier_feature_map(x, n_features=40, sigma=10.0):
    """傅里叶特征映射: γ(v) = [cos(2π B v), sin(2π B v)], B ~ N(0, σ²)"""
    generator = torch.Generator(device=x.device)
    generator.manual_seed(42)
    B = torch.randn(n_features, generator=generator, device=x.device) * sigma
    x_proj = 2 * torch.pi * x @ B.unsqueeze(0)
    return torch.cat([torch.cos(x_proj), torch.sin(x_proj)], dim=-1)
