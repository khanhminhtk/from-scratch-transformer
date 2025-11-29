import torch

from src.model.mask_attention import MaskSelfAttention
from src.model.linear import Linear

class MaskMultiHeadAttention:
    def __init__(self, n_head, e_dim, mean, std, device):
        self.n_head = n_head
        self.maskselsattentions = [
            MaskSelfAttention(
                e_dim=e_dim,
                mean=mean,
                std=std,
                device=device
            )
            for _ in range(n_head)
        ]
        self.scaler = Linear(
            in_feature=e_dim*n_head,
            out_feature=e_dim,
            mean=mean,
            std=std,
            device=device
        )

    def forward(self, X: torch.Tensor, mask: torch.Tensor):
        output = [maskselsattention.forward(X=X, mask=mask) for maskselsattention in self.maskselsattentions]
        concatenated = torch.cat(output, dim=-1)
        return self.scaler.forward(X=concatenated)
    
    def backward(self, grad_out: torch.Tensor):
        grad_concat = self.scaler.backward(grad_out=grad_out)
        grad_heads = torch.chunk(grad_concat, self.n_head, dim=-1)
        grad_X_total = None
        for head, g in zip(self.maskselsattentions, grad_heads):
            grad_X_head = head.backward(g)  # (B, L, e_dim)
            if grad_X_total is None:
                grad_X_total = grad_X_head
            else:
                grad_X_total = grad_X_total + grad_X_head
        return grad_X_total
    
    def parameters(self):
        params = []
        for head in self.maskselsattentions:
            params += head.parameters()
        params += self.scaler.parameters()
        return params

    def zero_grad(self):
        for head in self.maskselsattentions:
            if hasattr(head, "zero_grad"):
                head.zero_grad()
        self.scaler.zero_grad()