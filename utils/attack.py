import torch
from utils.utils import atleast_kdim


def get_init_with_noise(model, X, y):
    init = X.clone()
    p = model(X).argmax(1)

    while any(p == y):
        noise_scale = 0.5  # todo 限制最大噪声强度
        init = torch.where(
            atleast_kdim(p == y, len(X.shape)), 
            (X + noise_scale*torch.randn_like(X)).clip(0, 1), 
            init)
        p = model(init).argmax(1)
    return init
