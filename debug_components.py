import torch
import math

def test_linear():
    print("="*50)
    print("Testing Linear...")
    from src.model.linear import Linear
    
    linear = Linear(in_feature=32, out_feature=64, device="cuda", mean=0, std=0.01)
    x = torch.randn(2, 5, 32, device="cuda")
    
    y = linear.forward(x)
    print(f"  Forward output: mean={y.mean().item():.4f}, std={y.std().item():.4f}")
    print(f"  Has NaN: {torch.isnan(y).any().item()}")
    
    grad_out = torch.randn_like(y)
    grad_in = linear.backward(grad_out)
    print(f"  Backward grad: mean={grad_in.mean().item():.4f}, std={grad_in.std().item():.4f}")
    print(f"  Has NaN: {torch.isnan(grad_in).any().item()}")
    print(f"  grad_weights has NaN: {torch.isnan(linear.grad_weights).any().item()}")
    print("  ✅ Linear OK" if not torch.isnan(grad_in).any() else "  ❌ Linear FAILED")


def test_softmax():
    print("="*50)
    print("Testing Softmax...")
    from src.model.activate_function.softmax import Softmax
    
    softmax = Softmax()
    x = torch.randn(2, 5, 10, device="cuda")
    
    y = softmax.forward(x)
    print(f"  Forward output: mean={y.mean().item():.4f}, sum per row={y.sum(dim=-1).mean().item():.4f}")
    print(f"  Has NaN: {torch.isnan(y).any().item()}")
    
    grad_out = torch.randn_like(y)
    grad_in = softmax.backward(grad_out)
    print(f"  Backward grad: mean={grad_in.mean().item():.4f}, std={grad_in.std().item():.4f}")
    print(f"  Has NaN: {torch.isnan(grad_in).any().item()}")
    print("  ✅ Softmax OK" if not torch.isnan(grad_in).any() else "  ❌ Softmax FAILED")


def test_layernorm():
    print("="*50)
    print("Testing LayerNorm...")
    from src.model.add_norm import LayerNorm
    
    ln = LayerNorm(e_dim=32, device="cuda")
    x = torch.randn(2, 5, 32, device="cuda")
    
    y = ln.forward(x)
    print(f"  Forward output: mean={y.mean().item():.4f}, std={y.std().item():.4f}")
    print(f"  Has NaN: {torch.isnan(y).any().item()}")
    
    grad_out = torch.randn_like(y)
    grad_in = ln.backward(grad_out)
    print(f"  Backward grad: mean={grad_in.mean().item():.4f}, std={grad_in.std().item():.4f}")
    print(f"  Has NaN: {torch.isnan(grad_in).any().item()}")
    print("  ✅ LayerNorm OK" if not torch.isnan(grad_in).any() else "  ❌ LayerNorm FAILED")


def test_self_attention():
    print("="*50)
    print("Testing SelfAttention...")
    from src.model.self_attention import SelfAttention
    
    attn = SelfAttention(e_dim=32, mean=0, std=0.01, device="cuda")
    x = torch.randn(2, 5, 32, device="cuda")
    
    y = attn.forward(x)
    print(f"  Forward output: mean={y.mean().item():.4f}, std={y.std().item():.4f}")
    print(f"  Has NaN: {torch.isnan(y).any().item()}")
    
    grad_out = torch.randn_like(y)
    grad_in = attn.backward(grad_out)
    print(f"  Backward grad: mean={grad_in.mean().item():.4f}, std={grad_in.std().item():.4f}")
    print(f"  Has NaN: {torch.isnan(grad_in).any().item()}")
    print("  ✅ SelfAttention OK" if not torch.isnan(grad_in).any() else "  ❌ SelfAttention FAILED")


def test_self_attention_cross():
    print("="*50)
    print("Testing SelfAttention (Cross-Attention mode)...")
    from src.model.self_attention import SelfAttention
    
    attn = SelfAttention(e_dim=32, mean=0, std=0.01, device="cuda")
    x_dec = torch.randn(2, 5, 32, device="cuda")  # decoder input
    enc_K = torch.randn(2, 7, 32, device="cuda")  # encoder K
    enc_V = torch.randn(2, 7, 32, device="cuda")  # encoder V
    
    y = attn.forward(x_dec, encoder_output_K=enc_K, encoder_output_V=enc_V)
    print(f"  Forward output: mean={y.mean().item():.4f}, std={y.std().item():.4f}")
    print(f"  Has NaN: {torch.isnan(y).any().item()}")
    
    grad_out = torch.randn_like(y)
    grad_X_dec, grad_enc_K, grad_enc_V = attn.backward(grad_out)
    print(f"  grad_X_dec: mean={grad_X_dec.mean().item():.4f}, NaN={torch.isnan(grad_X_dec).any().item()}")
    print(f"  grad_enc_K: mean={grad_enc_K.mean().item():.4f}, NaN={torch.isnan(grad_enc_K).any().item()}")
    print(f"  grad_enc_V: mean={grad_enc_V.mean().item():.4f}, NaN={torch.isnan(grad_enc_V).any().item()}")
    
    has_nan = torch.isnan(grad_X_dec).any() or torch.isnan(grad_enc_K).any() or torch.isnan(grad_enc_V).any()
    print("  ✅ Cross-Attention OK" if not has_nan else "  ❌ Cross-Attention FAILED")


def test_ffn():
    print("="*50)
    print("Testing FeedForwardNetworks...")
    from src.model.feed_forward_networks import FeedForwardNetworks
    
    ffn = FeedForwardNetworks(e_dim=32, mean=0, std=0.01, device="cuda", hidden_dim=64)
    x = torch.randn(2, 5, 32, device="cuda")
    
    y = ffn.forward(x)
    print(f"  Forward output: mean={y.mean().item():.4f}, std={y.std().item():.4f}")
    print(f"  Has NaN: {torch.isnan(y).any().item()}")
    
    grad_out = torch.randn_like(y)
    grad_in = ffn.backward(grad_out)
    print(f"  Backward grad: mean={grad_in.mean().item():.4f}, std={grad_in.std().item():.4f}")
    print(f"  Has NaN: {torch.isnan(grad_in).any().item()}")
    print("  ✅ FFN OK" if not torch.isnan(grad_in).any() else "  ❌ FFN FAILED")


def test_encoder_block():
    print("="*50)
    print("Testing BlockEncoder...")
    from src.model.encoder import BlockEncoder
    
    encoder = BlockEncoder(e_dim=32, n_head=4, mean=0, std=0.01, device="cuda", hidden_dim_ffn=64)
    x = torch.randn(2, 5, 32, device="cuda")
    
    y = encoder.forward(x)
    print(f"  Forward output: mean={y.mean().item():.4f}, std={y.std().item():.4f}")
    print(f"  Has NaN: {torch.isnan(y).any().item()}")
    
    grad_out = torch.randn_like(y)
    grad_in = encoder.backward(grad_out)
    print(f"  Backward grad: mean={grad_in.mean().item():.4f}, std={grad_in.std().item():.4f}")
    print(f"  Has NaN: {torch.isnan(grad_in).any().item()}")
    print("  ✅ BlockEncoder OK" if not torch.isnan(grad_in).any() else "  ❌ BlockEncoder FAILED")


def test_decoder_block():
    print("="*50)
    print("Testing BlockDecoder...")
    from src.model.decoder import BlockDecoder
    from src.model.mask_utils import create_causal_mask
    
    decoder = BlockDecoder(n_head=4, e_dim=32, mean=0, std=0.01, device="cuda", hidden_dim_ffn=64)
    x = torch.randn(2, 5, 32, device="cuda")
    enc_K = torch.randn(2, 7, 32, device="cuda")
    enc_V = torch.randn(2, 7, 32, device="cuda")
    mask = create_causal_mask(5, "cuda")
    
    y = decoder.forward(x, mask=mask, encoder_output_K=enc_K, encoder_output_V=enc_V)
    print(f"  Forward output: mean={y.mean().item():.4f}, std={y.std().item():.4f}")
    print(f"  Has NaN: {torch.isnan(y).any().item()}")
    
    grad_out = torch.randn_like(y)
    grad_X, grad_K, grad_V = decoder.backward(grad_out)
    print(f"  grad_X: mean={grad_X.mean().item():.4f}, NaN={torch.isnan(grad_X).any().item()}")
    print(f"  grad_K: mean={grad_K.mean().item():.4f}, NaN={torch.isnan(grad_K).any().item()}")
    print(f"  grad_V: mean={grad_V.mean().item():.4f}, NaN={torch.isnan(grad_V).any().item()}")
    
    has_nan = torch.isnan(grad_X).any() or torch.isnan(grad_K).any() or torch.isnan(grad_V).any()
    print("  ✅ BlockDecoder OK" if not has_nan else "  ❌ BlockDecoder FAILED")


def test_mask_attention():
    print("="*50)
    print("Testing MaskSelfAttention...")
    from src.model.mask_attention import MaskSelfAttention
    from src.model.mask_utils import create_causal_mask
    
    attn = MaskSelfAttention(e_dim=32, mean=0, std=0.01, device="cuda")
    x = torch.randn(2, 5, 32, device="cuda")
    mask = create_causal_mask(5, "cuda")
    
    y = attn.forward(x, mask=mask)
    print(f"  Forward output: mean={y.mean().item():.4f}, std={y.std().item():.4f}")
    print(f"  Has NaN: {torch.isnan(y).any().item()}")
    
    grad_out = torch.randn_like(y)
    grad_in = attn.backward(grad_out)
    print(f"  Backward grad: mean={grad_in.mean().item():.4f}, std={grad_in.std().item():.4f}")
    print(f"  Has NaN: {torch.isnan(grad_in).any().item()}")
    print("  ✅ MaskSelfAttention OK" if not torch.isnan(grad_in).any() else "  ❌ MaskSelfAttention FAILED")


if __name__ == "__main__":
    print("\n🔍 DEBUGGING TRANSFORMER COMPONENTS\n")
    
    test_linear()
    test_softmax()
    test_layernorm()
    test_self_attention()
    test_self_attention_cross()
    test_mask_attention()
    test_ffn()
    test_encoder_block()
    test_decoder_block()
    
    print("\n" + "="*50)
    print("DEBUG COMPLETE!")
