import torch

from src.model.add_norm import AddNorm
from src.model.multi_head_attention import MultiHeadAttention
from src.model.feed_forward_networks import FeedForwardNetworks


class BlockEncoder:
    def __init__(self, e_dim, n_head, mean, std, device, hidden_dim_ffn):
        self.multi_head_attention = MultiHeadAttention(
            n_head=n_head,
            e_dim=e_dim,
            mean=mean,
            std=std,
            device=device
        )
        self.addnorm_multi_head = AddNorm(e_dim=e_dim, device=device)
        self.feed_forward_networks = FeedForwardNetworks(
            e_dim=e_dim,
            mean=mean,
            std=std,
            device=device,
            hidden_dim=hidden_dim_ffn
        )
        self.addnorm_ffn = AddNorm(e_dim=e_dim, device=device)

    def forward(self, X: torch.Tensor):
        self._cache_X = X
        out = self.multi_head_attention.forward(X)
        out_addnorm_multi_head = self.addnorm_multi_head.forward(X=X, sublayer=out)
        out = self.feed_forward_networks.forward(X=out_addnorm_multi_head)
        out = self.addnorm_ffn.forward(X=out_addnorm_multi_head, sublayer=out)
        return out
    
    def backward(self, grad_out: torch.Tensor):
        grad_out_add1_from_add2, grad_out_ffn = self.addnorm_ffn.backward(grad_out)
        grad_out_add1_from_ffn = self.feed_forward_networks.backward(grad_out_ffn)
        grad_out_add1_total = grad_out_add1_from_add2 + grad_out_add1_from_ffn
        grad_X_from_add1, grad_out_mha = self.addnorm_multi_head.backward(grad_out_add1_total)
        grad_X_from_mha = self.multi_head_attention.backward(grad_out_mha)
        grad_X_total = grad_X_from_add1 + grad_X_from_mha

        return grad_X_total

    def parameters(self):
        return (
            self.multi_head_attention.parameters()
            + self.addnorm_multi_head.parameters()
            + self.feed_forward_networks.parameters()
            + self.addnorm_ffn.parameters()
        )
    
    def zero_grad(self):
        for m in [
            self.multi_head_attention,
            self.addnorm_multi_head,
            self.feed_forward_networks,
            self.addnorm_ffn,
        ]:
            if hasattr(m, "zero_grad"):
                m.zero_grad()

    
# X = torch.normal(mean=0, std=1, size=(3, 32, 124)).to("cuda")
# a = BlockEncoder(n_head=8, e_dim=124, mean=0, std=1, device="cuda", hidden_dim_ffn=512)
# y = a.forward(X)
# print(y)