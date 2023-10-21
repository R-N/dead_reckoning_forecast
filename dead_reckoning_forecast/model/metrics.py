import torch

def mape(pred, y, weights=True, eps=1e-9, dim=-1):
    if weights is None:
        weights = torch.ones(y.size(dim))
    if weights is True:
        weights = torch.Tensor([1/i for i in range(1, y.size(dim)+1)])
    weights = weights / sum(weights)
    extra_shape = y.shape[(dim+1):]
    extra_dim = (1,)*len(extra_shape)
    weights = weights.view(-1, *extra_dim)

    ape = torch.abs(y-pred)/torch.clamp(torch.abs(y), min=eps)
    value = torch.sum(weights * ape, dim=dim)
    return value
