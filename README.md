# From-Scratch Transformer (PyTorch Tensor Only)

Dự án này hiện thực một mô hình **Transformer encoder–decoder** hoàn chỉnh *từ số 0* chỉ dùng `torch.Tensor` thuần:

- **Không dùng** `torch.nn.Module`
- **Không dùng** `torch.autograd`
- Tự viết:
  - Linear, ReLU, Softmax, LayerNorm, Multi-Head Attention, Add&Norm
  - Encoder / Decoder block, Encoder / Decoder stack, full Transformer
  - Optimizer **AdamW**
  - Training loop + overfitting test nhỏ để verify gradient

Mục tiêu: *hiểu rõ Transformer ở mức toán & code*, trước khi dùng `torch.nn` cho các project AI Engineer thực chiến.

---

## 🔧 Tính năng chính

- **Core layers tự viết**:
  - `linear.py` – Linear layer + backward tay
  - `activate_function/softmax.py` – Softmax ổn định số + backward
  - `activate_function/relu.py` – ReLU + backward
  - `self_attention.py` – Self-Attention (self/cross) + backward
  - `multi_head_attention.py` – Multi-Head Attention + backward
  - `mask_attention.py`, `mask_multi_head_attention.py`, `mask_utils.py` – masked self-attention (causal / padding)
  - `add_norm.py` – LayerNorm + Add&Norm (residual + layer norm)
  - `feed_forward_networks.py` – FFN 2-layer
  - `embedding/embedding.py` – simple embedding + positional encoding (sinusoidal)

- **Kiến trúc Transformer**:
  - `encoder.py` – `BlockEncoder`
    - Multi-Head Self-Attention → Add&Norm → FFN → Add&Norm
  - `bert.py` – encoder stack (n_encoder × BlockEncoder)
  - `decoder.py` – `BlockDecoder`
    - Masked self-attention + Add&Norm  
    - Cross-attention với encoder (K, V) + Add&Norm  
    - FFN + Add&Norm
  - `decoders.py` – decoder stack (n_decoder × BlockDecoder)
  - `transformers.py` – full Transformer encoder–decoder:
    - encoder (Bert)
    - linear K/V cho encoder output
    - decoder stack
    - linear vocab + softmax output

- **Optimizer tự code**:
  - `optimizer/adamw.py` – bản AdamW riêng:
    - moment `m`, `v`
    - bias correction `m_hat`, `v_hat`
    - decoupled weight decay: `param *= (1 - lr * weight_decay)`
  - Base class `Optimizer` với API giống `torch.optim.Optimizer`:
    - `zero_grad()`
    - `step()`

---

## 📁 Cấu trúc thư mục

```text
.
├── src
│   └── model
│       ├── activate_function
│       │   ├── relu.py
│       │   └── softmax.py
│       ├── embedding
│       │   └── embedding.py
│       ├── optimizer
│       │   └── adamw.py
│       ├── add_norm.py
│       ├── bert.py
│       ├── decoder.py
│       ├── decoders.py
│       ├── encoder.py
│       ├── feed_forward_networks.py
│       ├── linear.py
│       ├── mask_attention.py
│       ├── mask_multi_head_attention.py
│       ├── mask_utils.py
│       ├── multi_head_attention.py
│       ├── self_attention.py
│       └── transformers.py
├── debug_components.py    # Script test từng component (attention, LN, FFN, ...)
├── test_transformer.py    # Script test tổng thể Transformer
├── train_overfit.py       # Script train overfit dataset toy
├── README.md
└── .gitignore

```

## 📊 Kết quả (Overfit toy dataset)

Dự án đi kèm script `train_overfit.py` để kiểm tra xem toàn bộ mô hình (forward + backward + optimizer) có hoạt động đúng hay không bằng cách **overfit 4 sample nhỏ**.

Ví dụ log chạy thực tế:

```text
Using device: cuda

==================================================
Bắt đầu training (overfit test)...
Dataset size: 4 samples
Sequence length: 5
==================================================

Epoch    0 | Loss: 2.293168 | Accuracy: 13.33%
Epoch   10 | Loss: 1.525540 | Accuracy: 46.67%
Epoch   20 | Loss: 0.873742 | Accuracy: 73.33%
Epoch   30 | Loss: 0.497384 | Accuracy: 80.00%
Epoch   40 | Loss: 0.254271 | Accuracy: 100.00%
Epoch   50 | Loss: 0.112127 | Accuracy: 100.00%
Epoch   60 | Loss: 0.075076 | Accuracy: 100.00%
Epoch   70 | Loss: 0.023652 | Accuracy: 100.00%
Epoch   80 | Loss: 0.015757 | Accuracy: 100.00%
Epoch   90 | Loss: 0.012851 | Accuracy: 100.00%
Epoch  100 | Loss: 0.011614 | Accuracy: 100.00%
...
Epoch  490 | Loss: 0.006171 | Accuracy: 100.00%

==================================================
Training hoàn thành!
==================================================

📊 Kết quả cuối cùng:
  - Loss đầu: 2.293168
  - Loss cuối: 0.006138
  - Giảm: 99.7%

🔍 Test predictions:
  Sample 1: Target=[3, 4, 5, 2], Pred=[3, 4, 5, 2] ✅
  Sample 2: Target=[6, 7, 2], Pred=[6, 7, 2] ✅
  Sample 3: Target=[3, 5, 7, 2], Pred=[3, 5, 7, 2] ✅
  Sample 4: Target=[4, 6, 8, 2], Pred=[4, 6, 8, 2] ✅

🎉 PASS: Model học được! Loss giảm xuống < 0.1

```

## 📚 Tài liệu & bài báo liên quan

Dự án này dựa trên các ý tưởng kinh điển trong deep learning & Transformer:

- **Attention is All You Need**  
  *Ashish Vaswani, Noam Shazeer, Niki Parmar, et al., NeurIPS 2017*  
  Bài báo giới thiệu kiến trúc Transformer, scaled dot-product attention, multi-head attention, positional encoding.  
  PDF / arXiv: https://arxiv.org/abs/1706.03762
  
- **BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding**  
  *Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova, 2018*  
  Sử dụng encoder stack kiểu Transformer (giống phần `Bert` trong repo) cho các bài toán NLP.  
  ArXiv: https://arxiv.org/abs/1810.04805

- **Adam: A Method for Stochastic Optimization**  
  *Diederik P. Kingma, Jimmy Ba, 2014*  
  Trình bày optimizer Adam – nền tảng cho phần cập nhật moment `m`, `v` trong `AdamW`.  
  ArXiv: https://arxiv.org/abs/1412.6980

- **Decoupled Weight Decay Regularization (AdamW)**  
  *Ilya Loshchilov, Frank Hutter, ICLR 2019*  
  Phân biệt rõ Adam + L2 regularization và **AdamW** với weight decay tách rời – chính là kiểu update được hiện thực trong `optimizer/adamw.py`.  
  ArXiv: https://arxiv.org/abs/1711.05101

