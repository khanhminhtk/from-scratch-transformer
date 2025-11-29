import torch

class Softmax:
    def __init__(self, dim: int = -1):
        self.dim = dim
        self._cache_out = None

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X_max, _ = X.max(dim=self.dim, keepdim=True)
        X_stable = X - X_max
        exp = torch.exp(X_stable)
        out = exp / exp.sum(dim=self.dim, keepdim=True)
        self._cache_out = out
        return out

    def backward(self, grad_out: torch.Tensor) -> torch.Tensor:
        a = self._cache_out
        dot = (grad_out * a).sum(dim=self.dim, keepdim=True)
        grad_x = a * (grad_out - dot)
        return grad_x

# X = torch.normal( mean=0, std= 1, size=(3, 32, 124)).to("cuda")

# softmax = Softmax()

# y = softmax.forward(X=X)
# print(y)

# print(y.shape)