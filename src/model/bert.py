import torch

from src.model.encoder import BlockEncoder

class Bert:
    def __init__(
        self,
        n_encoder: int,
        n_head: int,
        e_dim: int,
        std: float,
        mean: float,
        device: str,
        hidden_dim_ffn: int
    ):
        self.encoders = [
            BlockEncoder(e_dim=e_dim, n_head=n_head, mean=mean, std=std, device=device, hidden_dim_ffn=hidden_dim_ffn)
            for i in range(n_encoder)
        ]

    def forward(self, X: torch.Tensor):
        out: torch.Tensor
        for i in range(len(self.encoders)):
            if i == 0:
                out = self.encoders[i].forward(X)
            else:
                out = self.encoders[i].forward(out)

        return out
    
    def backward(self, grad_out: torch.Tensor):
        grad = grad_out
        for encoder in reversed(self.encoders):
            grad = encoder.backward(grad)
        return grad
    
    def parameters(self):
        params = []
        for encoder in self.encoders:
            params += encoder.parameters()
        return params
    
    def zero_grad(self):
        for encoder in self.encoders:
            if hasattr(encoder, "zero_grad"):
                encoder.zero_grad()



# X = torch.normal(mean=0, std=1, size=(3, 32, 124)).to("cuda")
# a = Bert(n_encoder = 12, n_head=8, e_dim=124, mean=0, std=1, device="cuda", hidden_dim_ffn=512)
# y = a.forward(X)
# print(y)