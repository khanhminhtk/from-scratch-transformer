import torch

def create_causal_mask(seq_len: int, device: str) -> torch.Tensor:
    """
    Tạo causal mask (lower triangular) cho autoregressive modeling.
    
    Args:
        seq_len: Độ dài sequence
        device: Device để đặt tensor
        
    Returns:
        Mask tensor shape (seq_len, seq_len) với -inf ở upper triangle
        
    Example:
        >>> mask = create_causal_mask(3, "cuda")
        >>> # [[0,   -∞,  -∞],
        >>> #  [0,    0,  -∞], 
        >>> #  [0,    0,   0]]
    """
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
    mask = mask.masked_fill(mask == 1, float('-inf'))
    return mask.to(device)


def create_padding_mask(tokens: torch.Tensor, pad_token_id: int) -> torch.Tensor:
    """
    Tạo padding mask để ignore padded positions.
    
    Args:
        tokens: Token tensor shape (batch_size, seq_len)
        pad_token_id: ID của padding token
        
    Returns:
        Mask tensor shape (batch_size, seq_len, seq_len) với -inf ở padding positions
        
    Example:
        >>> tokens = torch.tensor([[1, 2, 0, 0]])  # 0 là padding
        >>> mask = create_padding_mask(tokens, pad_token_id=0)
    """
    batch_size, seq_len = tokens.shape
    
    # Tìm padding positions: (batch_size, seq_len)
    pad_positions = (tokens == pad_token_id)
    
    # Expand để tạo attention mask: (batch_size, seq_len, seq_len)
    mask = pad_positions.unsqueeze(1).expand(-1, seq_len, -1)
    
    # Convert True -> -inf, False -> 0
    return mask.masked_fill(mask, float('-inf'))


def create_look_ahead_mask(seq_len: int, device: str) -> torch.Tensor:
    """
    Alias cho create_causal_mask - tên rõ ý nghĩa hơn.
    """
    return create_causal_mask(seq_len, device)


def combine_masks(*masks: torch.Tensor) -> torch.Tensor:
    """
    Kết hợp nhiều masks bằng cách cộng chúng lại.
    
    Args:
        *masks: Variable number of mask tensors
        
    Returns:
        Combined mask tensor
        
    Example:
        >>> causal_mask = create_causal_mask(seq_len, device)
        >>> pad_mask = create_padding_mask(tokens, pad_id)
        >>> combined = combine_masks(causal_mask, pad_mask)
    """
    if not masks:
        raise ValueError("At least one mask must be provided")
    
    result = masks[0]
    for mask in masks[1:]:
        result = result + mask
    
    return result