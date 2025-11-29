"""
Script để test xem Transformer có học được không bằng cách overfit trên dataset nhỏ.
Nếu loss giảm xuống gần 0 -> model hoạt động đúng!
"""

import torch
import torch.nn.functional as F

from src.model.transformers import Transformer
from src.model.embedding.embedding import Embedding, PositionEncoding
from src.model.mask_utils import create_causal_mask
from src.model.optimizer.adamw import AdamW


def create_tiny_dataset(device):
    """
    Tạo dataset nhỏ để test overfit.
    Task: Copy input sang output (shifted by 1)
    
    Ví dụ:
        Input:  [BOS, 5, 3, 7, 2]
        Output: [BOS, 5, 3, 7, 2, EOS]
        Target: [5, 3, 7, 2, EOS]  (shifted)
    """
    # Vocab: 0=PAD, 1=BOS, 2=EOS, 3-9=actual tokens
    # Chỉ 4 samples để dễ overfit
    
    encoder_inputs = torch.tensor([
        [1, 3, 4, 5, 2],  # BOS, 3, 4, 5, EOS
        [1, 6, 7, 2, 0],  # BOS, 6, 7, EOS, PAD  
        [1, 3, 5, 7, 2],  # BOS, 3, 5, 7, EOS
        [1, 4, 6, 8, 2],  # BOS, 4, 6, 8, EOS
    ], device=device)
    
    # Decoder input (teacher forcing): BOS + target[:-1]
    decoder_inputs = torch.tensor([
        [1, 3, 4, 5, 2],  # BOS, 3, 4, 5, EOS
        [1, 6, 7, 2, 0],  # BOS, 6, 7, EOS, PAD
        [1, 3, 5, 7, 2],  # BOS, 3, 5, 7, EOS
        [1, 4, 6, 8, 2],  # BOS, 4, 6, 8, EOS
    ], device=device)
    
    # Target: what decoder should predict (shifted by 1)
    targets = torch.tensor([
        [3, 4, 5, 2, 0],  # 3, 4, 5, EOS, PAD
        [6, 7, 2, 0, 0],  # 6, 7, EOS, PAD, PAD
        [3, 5, 7, 2, 0],  # 3, 5, 7, EOS, PAD
        [4, 6, 8, 2, 0],  # 4, 6, 8, EOS, PAD
    ], device=device)
    
    return encoder_inputs, decoder_inputs, targets


def cross_entropy_loss_with_grad(logits, targets, ignore_index=0):
    """
    Cross entropy loss với backward.
    
    Args:
        logits: (B, L, vocab_size) - output từ model (sau softmax)
        targets: (B, L) - target token ids
        ignore_index: index để ignore (PAD token)
    
    Returns:
        loss: scalar
        grad: (B, L, vocab_size) - gradient w.r.t logits
    """
    B, L, V = logits.shape
    
    # Flatten
    logits_flat = logits.reshape(-1, V)  # (B*L, V)
    targets_flat = targets.reshape(-1)    # (B*L,)
    
    # Mask để ignore padding
    mask = (targets_flat != ignore_index).float()  # (B*L,)
    
    # Log của softmax output (logits đã qua softmax)
    log_probs = torch.log(logits_flat + 1e-10)  # (B*L, V)
    
    # Lấy log prob của target
    target_log_probs = log_probs[torch.arange(B*L, device=logits.device), targets_flat]  # (B*L,)
    
    # Apply mask và tính mean loss
    masked_loss = -target_log_probs * mask
    loss = masked_loss.sum() / (mask.sum() + 1e-10)
    
    # Gradient của cross entropy: softmax - one_hot(target)
    one_hot = torch.zeros_like(logits_flat)
    one_hot[torch.arange(B*L, device=logits.device), targets_flat] = 1.0
    
    grad_flat = (logits_flat - one_hot) * mask.unsqueeze(-1)
    grad_flat = grad_flat / (mask.sum() + 1e-10)
    
    grad = grad_flat.reshape(B, L, V)
    
    return loss, grad


class TransformerWithEmbedding:
    """Wrapper để kết hợp Embedding + Transformer"""
    
    def __init__(self, vocab_size, e_dim, n_transformer, n_head, hidden_dim_ffn, 
                 max_seq_len, device, mean=0.0, std=0.02):
        
        self.encoder_embedding = Embedding(
            vocab_size=vocab_size, e_dim=e_dim, mean=mean, std=std, device=device
        )
        self.decoder_embedding = Embedding(
            vocab_size=vocab_size, e_dim=e_dim, mean=mean, std=std, device=device
        )
        self.position_encoding = PositionEncoding(
            max_seq_len=max_seq_len, e_dim=e_dim, device=device
        )
        self.transformer = Transformer(
            n_transformer=n_transformer,
            n_head=n_head,
            e_dim=e_dim,
            mean=mean,
            std=std,
            device=device,
            hidden_dim_ffn=hidden_dim_ffn,
            vocab_size=vocab_size
        )
        self.device = device
        
    def forward(self, encoder_input, decoder_input, mask):
        # Embedding + Position Encoding
        enc_emb = self.encoder_embedding.forward(encoder_input)
        enc_emb = self.position_encoding.forward(enc_emb)
        
        dec_emb = self.decoder_embedding.forward(decoder_input)
        dec_emb = self.position_encoding.forward(dec_emb)
        
        # Transformer
        output = self.transformer.forward(
            embedding_encoder=enc_emb,
            embedding_decoder=dec_emb,
            mask=mask
        )
        return output
    
    def backward(self, grad_out):
        grad_enc_emb, grad_dec_emb = self.transformer.backward(grad_out)
        
        # Position encoding backward (identity)
        grad_enc_emb = self.position_encoding.backward(grad_enc_emb)
        grad_dec_emb = self.position_encoding.backward(grad_dec_emb)
        
        # Embedding backward
        self.encoder_embedding.backward(grad_enc_emb)
        self.decoder_embedding.backward(grad_dec_emb)
    
    def zero_grad(self):
        self.encoder_embedding.zero_grad()
        self.decoder_embedding.zero_grad()
        self.transformer.zero_grad()
    
    def parameters(self):
        params = []
        params.append({"param": self.encoder_embedding.weight, 
                       "grad": self.encoder_embedding.grad_weight})
        params.append({"param": self.decoder_embedding.weight, 
                       "grad": self.decoder_embedding.grad_weight})
        params += self.transformer.parameters()
        return params


def train():
    # Hyperparameters (nhỏ để dễ overfit)
    VOCAB_SIZE = 10
    E_DIM = 32
    N_TRANSFORMER = 2
    N_HEAD = 4
    HIDDEN_DIM_FFN = 64
    MAX_SEQ_LEN = 32
    LEARNING_RATE = 5e-3  # Tăng learning rate
    NUM_EPOCHS = 100  # Tăng epochs
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = TransformerWithEmbedding(
        vocab_size=VOCAB_SIZE,
        e_dim=E_DIM,
        n_transformer=N_TRANSFORMER,
        n_head=N_HEAD,
        hidden_dim_ffn=HIDDEN_DIM_FFN,
        max_seq_len=MAX_SEQ_LEN,
        device=device,
        std=0.01
    )
    
    # Optimizer
    optimizer = AdamW(params=model.parameters(), lr=LEARNING_RATE)
    
    # Dataset
    encoder_inputs, decoder_inputs, targets = create_tiny_dataset(device)
    seq_len = decoder_inputs.shape[1]
    
    # Causal mask cho decoder
    causal_mask = create_causal_mask(seq_len, device)
    
    print(f"\n{'='*50}")
    print("Bắt đầu training (overfit test)...")
    print(f"Dataset size: {encoder_inputs.shape[0]} samples")
    print(f"Sequence length: {seq_len}")
    print(f"{'='*50}\n")
    
    losses = []
    
    for epoch in range(NUM_EPOCHS):
        # Zero gradients
        model.zero_grad()
        
        # Forward pass
        output = model.forward(encoder_inputs, decoder_inputs, causal_mask)
        
        # Check for NaN in output
        if torch.isnan(output).any():
            print(f"Epoch {epoch}: NaN detected in output!")
            break
        
        # Compute loss
        loss, grad = cross_entropy_loss_with_grad(output, targets, ignore_index=0)
        
        # Check for NaN in loss/grad
        if torch.isnan(loss) or torch.isnan(grad).any():
            print(f"Epoch {epoch}: NaN detected in loss or grad!")
            break
        
        # Backward pass
        model.backward(grad)
        
        # Gradient clipping
        for param_dict in model.parameters():
            g = param_dict["grad"]
            if g is not None and torch.isnan(g).any():
                print(f"Epoch {epoch}: NaN detected in gradients!")
                break
            if g is not None:
                torch.clamp_(g, -1.0, 1.0)
        
        # Update weights
        optimizer.step()
        
        losses.append(loss.item())
        
        if epoch % 10 == 0:
            # Tính accuracy
            predictions = output.argmax(dim=-1)  # (B, L)
            mask = (targets != 0)  # Ignore padding
            correct = ((predictions == targets) & mask).sum().item()
            total = mask.sum().item()
            accuracy = correct / total if total > 0 else 0
            
            print(f"Epoch {epoch:4d} | Loss: {loss.item():.6f} | Accuracy: {accuracy:.2%}")
    
    print(f"\n{'='*50}")
    print("Training hoàn thành!")
    print(f"{'='*50}")
    
    # Kiểm tra kết quả cuối cùng
    print("\n📊 Kết quả cuối cùng:")
    print(f"  - Loss đầu: {losses[0]:.6f}")
    print(f"  - Loss cuối: {losses[-1]:.6f}")
    print(f"  - Giảm: {(losses[0] - losses[-1]) / losses[0] * 100:.1f}%")
    
    # Test prediction
    print("\n🔍 Test predictions:")
    output = model.forward(encoder_inputs, decoder_inputs, causal_mask)
    predictions = output.argmax(dim=-1)
    
    for i in range(len(encoder_inputs)):
        enc = encoder_inputs[i].tolist()
        target = targets[i].tolist()
        pred = predictions[i].tolist()
        
        # Filter out padding
        target_filtered = [t for t in target if t != 0]
        pred_filtered = pred[:len(target_filtered)]
        
        match = "✅" if target_filtered == pred_filtered else "❌"
        print(f"  Sample {i+1}: Target={target_filtered}, Pred={pred_filtered} {match}")
    
    # Đánh giá
    if losses[-1] < 0.1:
        print("\n🎉 PASS: Model học được! Loss giảm xuống < 0.1")
    elif losses[-1] < losses[0] * 0.5:
        print("\n⚠️  PARTIAL: Model đang học (loss giảm >50%) nhưng chưa converge hoàn toàn")
    else:
        print("\n❌ FAIL: Model không học được. Kiểm tra lại implementation!")
    
    return losses


if __name__ == "__main__":
    losses = train()
