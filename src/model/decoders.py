import torch

from src.model.decoder import BlockDecoder

class Decoders:
    def __init__(self, n_decoder, n_head, e_dim, mean, std, device, hidden_dim_ffn):
        self.decoders = [
            BlockDecoder(
                n_head=n_head,
                e_dim=e_dim,
                mean=mean,
                std=std,
                device=device,
                hidden_dim_ffn=hidden_dim_ffn
            )
            for _ in range(n_decoder)
        ]

    def forward(self, X: torch.Tensor, mask: torch.Tensor, encoder_output_K: torch.Tensor, encoder_output_V: torch.Tensor):
        output: torch.Tensor
        for i in range(len(self.decoders)):
            if i == 0:
                input = X
            else:
                input = output
            output = self.decoders[i].forward(
                X=input,
                mask=mask,
                encoder_output_K=encoder_output_K,
                encoder_output_V=encoder_output_V
            )
        return output
    
    def backward(self, grad_out: torch.Tensor):
        grad_dec = grad_out

        grad_enc_K_total = None
        grad_enc_V_total = None


        for decoder in reversed(self.decoders):
            grad_dec, grad_enc_K, grad_enc_V = decoder.backward(grad_dec)

            if grad_enc_K_total is None:
                grad_enc_K_total = grad_enc_K
                grad_enc_V_total = grad_enc_V
            else:
                grad_enc_K_total = grad_enc_K_total + grad_enc_K
                grad_enc_V_total = grad_enc_V_total + grad_enc_V

        grad_X_dec_in = grad_dec
        return grad_X_dec_in, grad_enc_K_total, grad_enc_V_total
    
    def parameters(self):
        params = []
        for decoder in self.decoders:
            params += decoder.parameters()
        return params

    def zero_grad(self):
        for decoder in self.decoders:
            if hasattr(decoder, "zero_grad"):
                decoder.zero_grad()
