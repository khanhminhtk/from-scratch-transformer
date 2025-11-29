import torch

from src.model.self_attention import SelfAttention
from src.model.linear import Linear

class MultiHeadAttention:
    def __init__(self, n_head, e_dim, mean, std, device):
        self.n_head = n_head
        self.multi_heads = [
            SelfAttention(e_dim, mean, std, device)
            for _ in range(n_head)
        ]
        self.scale = Linear(
            in_feature=e_dim*n_head, 
            out_feature=e_dim,
            device=device,
            mean=mean,
            std=std
        )

    def forward(self, X: torch.Tensor, encoder_output_K=None, encoder_output_V=None):
        if encoder_output_V is None and encoder_output_K is None:
            self._self_attn = True
            head_output = [head.forward(X) for head in self.multi_heads]
        else:
            self._self_attn = False
            head_output = [
                head.forward(X, encoder_output_K, encoder_output_V)
                for head in self.multi_heads
            ]
        concatenated = torch.cat(head_output, dim=-1)  # (B, L, e_dim * H)
        out = self.scale.forward(concatenated)
        return out

    def backward(self, grad_out: torch.Tensor):
        grad_concat = self.scale.backward(grad_out)  # (B, L, e_dim * H)
        grad_heads = torch.chunk(grad_concat, self.n_head, dim=-1)

        if self._self_attn:
            grad_X_total = None
            for head, g in zip(self.multi_heads, grad_heads):
                grad_X_head = head.backward(g)  # (B, L, e_dim)
                if grad_X_total is None:
                    grad_X_total = grad_X_head
                else:
                    grad_X_total = grad_X_total + grad_X_head
            return grad_X_total
        else:
            grad_X_dec_total = None
            grad_enc_K_total = None
            grad_enc_V_total = None

            for head, g in zip(self.multi_heads, grad_heads):
                grad_X_dec, grad_enc_K, grad_enc_V = head.backward(g)

                if grad_X_dec_total is None:
                    grad_X_dec_total = grad_X_dec
                    grad_enc_K_total = grad_enc_K
                    grad_enc_V_total = grad_enc_V
                else:
                    grad_X_dec_total = grad_X_dec_total + grad_X_dec
                    grad_enc_K_total = grad_enc_K_total + grad_enc_K
                    grad_enc_V_total = grad_enc_V_total + grad_enc_V

            return grad_X_dec_total, grad_enc_K_total, grad_enc_V_total
        
    def parameters(self):
        params = []
        for head in self.multi_heads:
            params += head.parameters()
        params += self.scale.parameters()
        return params

    def zero_grad(self):
        for head in self.multi_heads:
            if hasattr(head, "zero_grad"):
                head.zero_grad()
        self.scale.zero_grad()

# X = torch.normal(mean=0, std=1, size=(3, 32, 124)).to("cuda")
# a = MultiHeadAttention(n_head=8, e_dim=124, mean=0, std=1, device="cuda")
# y = a.forward(X)
# print(y)

# print(y.shape)
