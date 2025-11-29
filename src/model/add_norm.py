import torch

class LayerNorm:
    def __init__(self, e_dim, eps=1e-5, device="cpu"):
        self.eps = eps
        self.gamma = torch.ones(1, 1, e_dim, device=device)
        self.beta = torch.zeros(1, 1, e_dim, device=device)

        self.grad_gamma = torch.zeros_like(self.gamma)
        self.grad_beta = torch.zeros_like(self.beta)
    def forward(self, X: torch.Tensor):
        #X (B, sql, e_dim)

        mean = torch.mean(X, dim=-1, keepdim=True) #(B, sql, 1)
        std = torch.std(X, dim=-1, keepdim=True) #(B, sql, 1)
        std_hat =  std_hat = torch.sqrt(std**2 + self.eps)  
        x_hat = (X - mean)/std_hat#(B, sql, e_dim) 
        out = self.gamma * x_hat + self.beta
        self._cache_X = X
        self._cache_mean = mean
        self._cache_std = std
        self._cache_std_hat = std_hat
        self._cache_x_hat = x_hat
        return out
    

    def backward(self, grad_out: torch.Tensor) -> torch.Tensor:
        X = self._cache_X
        x_hat = self._cache_x_hat
        std_hat = self._cache_std_hat
        self.grad_gamma.copy_((grad_out * x_hat).sum(dim=(0, 1), keepdim=True))
        self.grad_beta.copy_(grad_out.sum(dim=(0, 1), keepdim=True))

        grad_x_hat = grad_out * self.gamma 
        mean_grad_x_hat = grad_x_hat.mean(dim=-1, keepdim=True)             # (B, seq, 1)
        mean_grad_x_hat_x_hat = (grad_x_hat * x_hat).mean(dim=-1, keepdim=True)  # (B, seq, 1)

        grad_X = (grad_x_hat
                  - mean_grad_x_hat
                  - x_hat * mean_grad_x_hat_x_hat) / std_hat

        return grad_X
    
    def zero_grad(self):
        self.grad_gamma.zero_()
        self.grad_beta.zero_()

    def parameters(self):
        return [
            {"param": self.gamma, "grad": self.grad_gamma},
            {"param": self.beta, "grad": self.grad_beta},
        ]
    

class AddNorm:
    def __init__(self, e_dim, eps=1e-5, device="cpu"):
        self.layernorm = LayerNorm(e_dim, eps=eps, device=device)

    def forward(self, X: torch.Tensor, sublayer: torch.Tensor):
        self._cache_X = X
        self._cache_sublayer = sublayer

        Y = X + sublayer                   # (B, seq, e_dim)
        out = self.layernorm.forward(Y)    # (B, seq, e_dim)
        return out

    def backward(self, grad_out: torch.Tensor):
        grad_Y = self.layernorm.backward(grad_out)  # (B, seq, e_dim)
        grad_X = grad_Y
        grad_sublayer = grad_Y

        return grad_X, grad_sublayer

    def parameters(self):
        return self.layernorm.parameters()

    def zero_grad(self):
        self.layernorm.zero_grad()



