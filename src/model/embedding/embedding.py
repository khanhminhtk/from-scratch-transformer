import torch

class Embedding:
    def __init__(self, vocab_size: int, e_dim: int, mean: float, std: float, device: str):
        self.weight = torch.normal(
            mean=mean,
            std=std,
            size=(vocab_size, e_dim),
            device=device
        )
        self.grad_weight = torch.zeros_like(self.weight)
        self._cache_token_ids = None

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        self._cache_token_ids = token_ids #  (batch, seq_len)
        out = self.weight[token_ids]  # indexing: (B, L, D)
        return out

    __call__ = forward

    def backward(self, grad_out: torch.Tensor):
        self.grad_weight.zero_()
        token_ids_flat = self._cache_token_ids.view(-1) #(B*L)
        grad_flat = grad_out.view(-1, self.weight.size(1)) #(B*L, e_dim)
        self.grad_weight.index_add_(0, token_ids_flat, grad_flat)
        return None

    def zero_grad(self):
        self.grad_weight.zero_()

def get_sinusoidal_positional_encoding(max_seq_len: int, e_dim: int, device: str):
    pos = torch.arange(0, max_seq_len, device=device, dtype=torch.float32).unsqueeze(1) # pos: (L, 1)
    i = torch.arange(0, e_dim, device=device, dtype=torch.float32).unsqueeze(0)    # i: (1, D)
    div_term = torch.pow(10000.0, (2 * (i // 2)) / e_dim)  # (1, D)
    pe = pos / div_term  # (L, D)
    pe[:, 0::2] = torch.sin(pe[:, 0::2]) 
    pe[:, 1::2] = torch.cos(pe[:, 1::2])
    return pe.unsqueeze(0) #(1, L, D)


class PositionEncoding:
    def __init__(self, max_seq_len: int, e_dim: int, device: str):
        self.pe = get_sinusoidal_positional_encoding(max_seq_len, e_dim, device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]

    __call__ = forward

    def backward(self, grad_out: torch.Tensor) -> torch.Tensor:
        return grad_out