import torch
import torch.nn as nn
from torchtyping import TensorType
class SingleHeadAttention(nn.Module):

    def __init__(self, embedding_dim: int, attention_dim: int):
        super().__init__()
        torch.manual_seed(0)
        # Create three linear projections (Key, Query, Value) with bias=False
        # Instantiation order matters for reproducible weights: key, query, value
        self.Wk = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.Wq = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.embedding_dim = embedding_dim
        self.attention_dim = attention_dim
        pass

    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
        # 1. Project input through K, Q, V linear layers
        Q = self.Wq(embedded)
        K = self.Wk(embedded)
        V = self.Wv(embedded)
        seq_len = embedded.shape[1]
        # 2. Compute attention scores: (Q @ K^T) / sqrt(attention_dim)
        atten = (Q @ K.transpose(-2, -1)) / self.attention_dim**(1/2)

        # 3. Apply causal mask: use torch.tril(torch.ones(...)) to build lower-triangular matrix,
        #    then masked_fill positions where mask == 0 with float('-inf')
        mask = torch.tril(torch.ones(seq_len, seq_len))
        # mask[mask == 0] = float('-inf')
        atten[:, mask == 0] = float('-inf')
        
        # 4. Apply softmax(dim=2) to masked scores
        scores = torch.softmax(atten, dim=2)
        # 5. Return (scores @ V) rounded to 4 decimal places
        
        return (scores @ V).round(decimals=4)