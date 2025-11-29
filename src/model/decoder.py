import torch

from src.model.mask_multi_head_attention import MaskMultiHeadAttention
from src.model.multi_head_attention import MultiHeadAttention
from src.model.add_norm import AddNorm
from src.model.feed_forward_networks import FeedForwardNetworks

class BlockDecoder:
    def __init__(self, n_head, e_dim, mean, std, device, hidden_dim_ffn):
        self.maskmuiltiheadattention = MaskMultiHeadAttention(
            n_head=n_head,
            e_dim=e_dim,
            mean=mean,
            std=std,
            device=device
        )

        self.addnorm_mask = AddNorm(e_dim=e_dim, device=device)
        self.multiheadattention = MultiHeadAttention(
            n_head=n_head,
            e_dim=e_dim,
            mean=mean,
            std=std,
            device=device
        )
        self.addnorm_cross_encoder = AddNorm(e_dim=e_dim, device=device)
        self.ffn = FeedForwardNetworks(
            e_dim=e_dim,
            mean=mean,
            std=std,
            device=device,
            hidden_dim=hidden_dim_ffn
        )
        self.addnorm_ffn = AddNorm(e_dim=e_dim, device=device)

    def forward(self, X: torch.Tensor, mask: torch.Tensor, encoder_output_K: torch.Tensor, encoder_output_V: torch.Tensor):
        output = self.maskmuiltiheadattention.forward(X=X, mask=mask)
        output_addnorm_mask = self.addnorm_mask.forward(X=X, sublayer=output)
        output = self.multiheadattention.forward(X=output_addnorm_mask, encoder_output_K=encoder_output_K, encoder_output_V=encoder_output_V)
        output_addnorm_cross_encoder = self.addnorm_cross_encoder.forward(X=output_addnorm_mask, sublayer=output)
        output = self.ffn.forward(X=output_addnorm_cross_encoder)
        output = self.addnorm_ffn.forward(X=output_addnorm_cross_encoder, sublayer=output)
        return output
    
    def backward(self, grad_out: torch.Tensor):
        grad_cross_encoder_from_addffn, grad_ffn_out = self.addnorm_ffn.backward(grad_out)
        grad_cross_encoder_from_ffn = self.ffn.backward(grad_ffn_out)
        grad_cross_encoder_total = grad_cross_encoder_from_addffn + grad_cross_encoder_from_ffn
        grad_addnorm_mask_from_add2, grad_cross_out = self.addnorm_cross_encoder.backward(grad_cross_encoder_total)
        grad_addnorm_mask_from_mha, grad_enc_K, grad_enc_V = self.multiheadattention.backward(grad_cross_out)
        grad_addnorm_mask_total = grad_addnorm_mask_from_add2 + grad_addnorm_mask_from_mha
        grad_X_from_add1, grad_mask_out = self.addnorm_mask.backward(grad_addnorm_mask_total)
        grad_X_from_mask = self.maskmuiltiheadattention.backward(grad_mask_out)
        grad_X_dec = grad_X_from_add1 + grad_X_from_mask

        return grad_X_dec, grad_enc_K, grad_enc_V

    def parameters(self):
        return (
            self.maskmuiltiheadattention.parameters()
            + self.addnorm_mask.parameters()
            + self.multiheadattention.parameters()
            + self.addnorm_cross_encoder.parameters()
            + self.ffn.parameters()
            + self.addnorm_ffn.parameters()
        )

    def zero_grad(self):
        for m in [
            self.maskmuiltiheadattention,
            self.addnorm_mask,
            self.multiheadattention,
            self.addnorm_cross_encoder,
            self.ffn,
            self.addnorm_ffn,
        ]:
            if hasattr(m, "zero_grad"):
                m.zero_grad()
