import torch

from src.model.bert import Bert
from src.model.activate_function.softmax import Softmax
from src.model.linear import Linear
from src.model.decoders import Decoders

class Transformer:
    def __init__(self, n_transformer, n_head, e_dim, mean, std, device, hidden_dim_ffn, vocab_size):
        self.bert = Bert(
            n_encoder=n_transformer,
            n_head=n_head,
            e_dim=e_dim,
            std=std,
            mean=mean,
            device=device,
            hidden_dim_ffn=hidden_dim_ffn
        )
        self.linear_k = Linear(
            in_feature=e_dim,
            out_feature=e_dim,
            mean=mean,
            std=std,
            device=device
        )
        self.linear_v = Linear(
            in_feature=e_dim,
            out_feature=e_dim,
            mean=mean,
            std=std,
            device=device
        )
        self.decoders = Decoders(
            n_decoder=n_transformer,
            n_head=n_head,
            e_dim=e_dim,
            std=std,
            mean=mean,
            device=device,
            hidden_dim_ffn=hidden_dim_ffn
        )
        self.softmax = Softmax()
        self.linear_after_decoder = Linear(
            in_feature=e_dim,
            out_feature=vocab_size,
            mean=mean,
            std=std,
            device=device
        )

    def forward(
        self,
        embedding_encoder,
        embedding_decoder,
        mask
    ):

        output_encoder = self.bert.forward(
            X=embedding_encoder
        )
        encoder_output_V = self.linear_v.forward(X=output_encoder)
        encoder_output_K = self.linear_k.forward(X=output_encoder)
        decoder_output = self.decoders.forward(
            X=embedding_decoder,
            mask=mask,
            encoder_output_K=encoder_output_K,
            encoder_output_V=encoder_output_V
        )
        logits = self.linear_after_decoder.forward(decoder_output)
        output = self.softmax.forward(X=logits)
        self._cache_embedding_encoder = embedding_encoder
        self._cache_embedding_decoder = embedding_decoder
        self._cache_output_encoder = output_encoder
        self._cache_encoder_output_K = encoder_output_K
        self._cache_encoder_output_V = encoder_output_V
        return output
    
    def backward(self, grad_out: torch.Tensor):
        grad_logits = self.softmax.backward(grad_out)  # (B, L_dec, vocab_size)
        grad_decoder_output = self.linear_after_decoder.backward(grad_logits)  # (B, L_dec, e_dim)
        grad_dec_in, grad_enc_K, grad_enc_V = self.decoders.backward(grad_decoder_output)
        grad_output_encoder_from_K = self.linear_k.backward(grad_enc_K)  # (B, L_enc, e_dim)
        grad_output_encoder_from_V = self.linear_v.backward(grad_enc_V)  # (B, L_enc, e_dim)
        grad_output_encoder_total = grad_output_encoder_from_K + grad_output_encoder_from_V
        grad_embedding_encoder = self.bert.backward(grad_output_encoder_total)
        grad_embedding_decoder = grad_dec_in

        return grad_embedding_encoder, grad_embedding_decoder

    def parameters(self):
        return (
            self.bert.parameters()
            + self.linear_k.parameters()
            + self.linear_v.parameters()
            + self.decoders.parameters()
            + self.linear_after_decoder.parameters()
        )

    def zero_grad(self):
        self.bert.zero_grad()
        self.linear_k.zero_grad()
        self.linear_v.zero_grad()
        self.decoders.zero_grad()
        self.linear_after_decoder.zero_grad()