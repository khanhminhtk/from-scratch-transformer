import torch
from src.model.transformers import Transformer
from src.model.embedding.embedding import Embedding, PositionEncoding
from src.model.mask_utils import create_causal_mask, create_padding_mask

# Test Transformer với Embedding
def test_full_transformer():
    print("🧪 Testing Full Transformer Architecture...")
    
    # Parameters
    vocab_size = 1000
    seq_len = 32
    batch_size = 2
    e_dim = 124
    n_head = 8
    n_transformer = 6
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Initialize components
    print(f"📱 Device: {device}")
    
    # Token embeddings
    src_embedding = Embedding(vocab_size, e_dim, mean=0, std=0.1, device=device)
    tgt_embedding = Embedding(vocab_size, e_dim, mean=0, std=0.1, device=device)
    
    # Positional encoding
    pos_encoding = PositionEncoding(max_seq_len=100, e_dim=e_dim, device=device)
    
    # Transformer model
    transformer = Transformer(
        n_transformer=n_transformer,
        n_head=n_head,
        e_dim=e_dim,
        mean=0,
        std=0.1,
        device=device,
        hidden_dim_ffn=512,
        vocab_size=vocab_size
    )
    
    # 2. Create sample data
    print("📝 Creating sample data...")
    
    # Source tokens (encoder input)
    src_tokens = torch.randint(1, vocab_size-1, (batch_size, seq_len)).to(device)
    
    # Target tokens (decoder input) - shifted right for training
    tgt_tokens = torch.randint(1, vocab_size-1, (batch_size, seq_len)).to(device)
    
    print(f"Source tokens shape: {src_tokens.shape}")
    print(f"Target tokens shape: {tgt_tokens.shape}")
    
    # 3. Create embeddings
    print("🔤 Creating embeddings...")
    
    # Source embeddings (encoder)
    src_embeddings = src_embedding(src_tokens)  # (B, L, D)
    src_embeddings = pos_encoding(src_embeddings)
    
    # Target embeddings (decoder)
    tgt_embeddings = tgt_embedding(tgt_tokens)  # (B, L, D)
    tgt_embeddings = pos_encoding(tgt_embeddings)
    
    print(f"Source embeddings shape: {src_embeddings.shape}")
    print(f"Target embeddings shape: {tgt_embeddings.shape}")
    
    # 4. Create masks
    print("🎭 Creating masks...")
    
    # Causal mask for decoder (không nhìn future tokens)
    causal_mask = create_causal_mask(seq_len, device)
    
    # Padding mask (if needed)
    pad_token_id = 0
    # Create some padding in target
    tgt_tokens_with_pad = tgt_tokens.clone()
    tgt_tokens_with_pad[:, -5:] = pad_token_id  # Last 5 tokens are padding
    
    padding_mask = create_padding_mask(tgt_tokens_with_pad, pad_token_id)
    
    # Combine masks
    combined_mask = causal_mask + padding_mask
    
    print(f"Causal mask shape: {causal_mask.shape}")
    print(f"Padding mask shape: {padding_mask.shape}")
    
    # 5. Forward pass
    print("⚡ Running forward pass...")
    
    try:
        output = transformer.forward(
            embedding_encoder=src_embeddings,
            embedding_decoder=tgt_embeddings,
            mask=combined_mask
        )
        
        print(f"✅ Success! Output shape: {output.shape}")
        print(f"Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
        print(f"Output sum: {output.sum().item():.4f}")
        
        # Check if probabilities sum to 1 (after softmax)
        prob_sums = output.sum(dim=-1)
        print(f"Probability sums (should be ~1.0): {prob_sums.mean().item():.4f} ± {prob_sums.std().item():.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_full_transformer()