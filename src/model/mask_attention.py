import torch
import math

from src.model.self_attention import SelfAttention

class MaskSelfAttention(SelfAttention):
    def __init__(self, e_dim, mean, std, device):
        super().__init__(e_dim, mean, std, device)

    def forward(self, X: torch.Tensor, mask: torch.Tensor):
        Q = self.q.forward(X)
        K = self.k.forward(X) 
        V = self.v.forward(X)

        d_k = K.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

        scores = scores + mask
        attn = self.softmax.forward(scores)
        out = torch.matmul(attn, V)
        self.cache_k = K
        self.cache_v = V
        self.cache_q = Q
        self.cache_attn = attn
        self.cache_scores = scores
        self.cache_out = out
        self.cache_x = X
        self.cache_mask = mask
        return out
    
    def backward(self, grad_out):
        K = self.cache_k
        V = self.cache_v
        Q = self.cache_q
        X = self.cache_x
        attn = self.cache_attn
        d_k = K.size(-1)
        grad_attn = torch.matmul(grad_out, V.transpose(-2, -1))
        grad_V = torch.matmul(attn.transpose(-2, -1), grad_out)
        grad_scores = self.softmax.backward(grad_attn) / math.sqrt(d_k)
        grad_Q = torch.matmul(grad_scores, K)
        grad_K = torch.matmul(grad_scores.transpose(-2, -1), Q)
        grad_X_Q = self.q.backward(grad_out=grad_Q)

        grad_X_K = self.k.backward(grad_out=grad_K)
        grad_X_V = self.v.backward(grad_out=grad_V)
        grad_X = grad_X_K + grad_X_Q+ grad_X_V
        return grad_X
    
    def parameters(self):
        return (
            self.v.parameters()+
            self.q.parameters()+
            self.k.parameters()
        )
    
    def zero_grad(self):
        self.v.zero_grad()
        self.q.zero_grad()
        self.k.zero_grad()