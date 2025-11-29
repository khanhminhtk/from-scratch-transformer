import math

import torch

from src.model.linear import Linear
from src.model.activate_function.softmax import Softmax

class SelfAttention:
    def __init__(self, e_dim, mean, std, device):
        self.v = Linear(in_feature=e_dim, out_feature=e_dim, mean=mean, std=std, device=device)
        self.k = Linear(in_feature=e_dim, out_feature=e_dim, mean=mean, std=std, device=device)
        self.q = Linear(in_feature=e_dim, out_feature=e_dim, mean=mean, std=std, device=device)
        self.device = device
        self.softmax = Softmax()

    def forward(self, X: torch.Tensor, encoder_output_K = None, encoder_output_V = None):
        #X (B, L, e_dim)
        Q=self.q.forward(X) #(B, L, e_dim)
        if encoder_output_K is None and encoder_output_V is None:
            K=self.k.forward(X)
            V=self.v.forward(X)
            self._attn = True
        else:
            K = encoder_output_K
            V = encoder_output_V
            self._attn = False

        d_k = K.size(-1)
        #K.transpose(-2, -1) (B, e_dim, L)
        # Q @ K^T (B, L, L)
        # softmax((Q @ K^T)/sqrt(d_k)) @ V (B, L, e_dim)
        # scores: (B, L, L)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        attn = self.softmax.forward(scores)  # (B, L, L)
        out = torch.matmul(attn, V)         # (B, L, e_dim)
        
        self.cache_k = K
        self.cache_v = V
        self.cache_q = Q
        self.cache_attn = attn
        self.cache_scores = scores
        self.cache_out = out
        self.cache_x = X
        return out
    
    def backward(self, grad_out: torch.Tensor):
        K = self.cache_k
        V = self.cache_v
        Q = self.cache_q
        X = self.cache_x
        attn = self.cache_attn
        d_k = K.size(-1)

        #shape out (B, sql, e_dim)
        #shape attn (B, sql, sql)
        #shape Q, K, L (B, sql, e_dim)
        #f = attn * V
        #dL/dattn = grad_out @ V^T  shape(B, sql, sql)
        #dL/dV =  attn^T @ grad_out shape(B, sql, e_dim)
        #attn = softmax(Q@K^T/sqrt(d_k)) 
        #dL/dQ = grad_scores @ K shape(B, sql, e_dim)
        #dL/dK = grad_scores^T @ Q shape(B, sql, e_dim)

        grad_attn = torch.matmul(grad_out, V.transpose(-2, -1))
        grad_V = torch.matmul(attn.transpose(-2, -1), grad_out)
        grad_scores = self.softmax.backward(grad_attn) / math.sqrt(d_k)
        grad_Q = torch.matmul(grad_scores, K)
        grad_K = torch.matmul(grad_scores.transpose(-2, -1), Q)
        grad_X_Q = self.q.backward(grad_out=grad_Q)

        if self._attn:
            grad_X_K = self.k.backward(grad_out=grad_K)
            grad_X_V = self.v.backward(grad_out=grad_V)
            grad_X = grad_X_K + grad_X_Q + grad_X_V
            return grad_X
        else:
            # Cross-attention: K, V từ encoder, Q từ decoder
            # grad_X_Q là gradient cho decoder input
            grad_enc_K = grad_K
            grad_enc_V = grad_V
            return grad_X_Q, grad_enc_K, grad_enc_V

    def parameters(self):
        return(
            self.q.parameters()+
            self.k.parameters()+
            self.v.parameters()
        )

    def zero_grad(self):
        self.q.zero_grad()
        self.k.zero_grad()
        self.v.zero_grad()
