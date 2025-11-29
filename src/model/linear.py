import torch
from src.model.optimizer.adamw import AdamW

class Linear:
    def __init__(self, *,in_feature: int, out_feature: int, device: str, mean: float, std: float):
        self.weights = self._init_weights(in_feature=in_feature, out_feature=out_feature, device=device, mean=mean, std=std)
        self.bias = self._init_bias(out_feature=out_feature, device=device, mean=mean, std=std)

        self.grad_weights = torch.zeros_like(self.weights)
        self.grad_bias = torch.zeros_like(self.bias)

    def _init_weights(self, in_feature, out_feature, device, mean, std):
        return torch.normal(
            mean=mean,
            std=std,
            size=(in_feature, out_feature)
        ).to(device=device)
    
    def _init_bias(self, out_feature, device, mean, std):
        return torch.normal(
            mean=mean,
            std=std,
            size=(out_feature,)
        ).to(device=device)
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        self._cache_X = X
        return X @ self.weights + self.bias
    
    def zero_grad(self):
        self.grad_bias.zero_()
        self.grad_weights.zero_()

    def parameters(self):
        return [
            {"param": self.weights, "grad": self.grad_weights},
            {"param": self.bias, "grad": self.grad_bias}
        ]
    
    def backward(self, grad_out: torch.Tensor) -> torch.Tensor:
        X = self._cache_X 
        X_flat = X.reshape(-1, X.size(-1))  # (N, in_feature)
        grad_out_flat = grad_out.reshape(-1, grad_out.size(-1))  # (N, out_feature)
        self.grad_weights.copy_(X_flat.t() @ grad_out_flat)
        self.grad_bias.copy_(grad_out_flat.sum(dim=0))
        grad_input = grad_out @ self.weights.t()

        return grad_input
    


# X = torch.normal( mean=0, std= 1, size=(3, 32, 124)).to("cuda")

# linear = Linear(in_feature=124, out_feature=512, device="cuda", mean=0, std= 1)

# y = linear.forward(X=X)
# print(y)

# print(y.shape)
        
