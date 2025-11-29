import torch

from src.model.linear import Linear
from src.model.activate_function.relu import Relu

class FeedForwardNetworks:
    def __init__(self, e_dim, mean, std, device, hidden_dim):
        self.layer_1 = Linear(
            in_feature=e_dim,
            out_feature=hidden_dim,
            mean=mean,
            std=std, 
            device=device
        )
        self.relu = Relu()
        self.layer_2 = Linear(
            in_feature=hidden_dim,
            out_feature=e_dim,
            mean=mean,
            std=std,
            device=device
        )
    
    def forward(self, X: torch.Tensor):
        out = self.layer_1.forward(X)
        out = self.relu.forward(out)
        out = self.layer_2.forward(out)
        return out
    
    def backward(self, grad_out: torch.Tensor):
        grad_out_layer_2 = self.layer_2.backward(grad_out=grad_out)
        grad_out_relu = self.relu.backward(grad_out_layer_2)
        grad_out_layer_1 = self.layer_1.backward(grad_out=grad_out_relu)
        return grad_out_layer_1
    
    def parameters(self):
        return (
            self.layer_1.parameters() + self.layer_2.parameters()
        )

    def zero_grad(self):
        self.layer_1.zero_grad()
        self.layer_2.zero_grad()