import torch 

class Relu:
    def forward(self, X: torch.Tensor):
        self._cache_X = X
        return torch.maximum(torch.zeros_like(X), X)
    
    def backward(self, grad_out: torch.Tensor):
        X = self._cache_X
        mask = (X > 0).float()
        return grad_out * mask
