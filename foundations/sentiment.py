import torch
import torch.nn as nn
from torchtyping import TensorType

class Solution(nn.Module):
    def __init__(self, vocabulary_size: int):
        super().__init__()
        torch.manual_seed(0)
        # Layers: Embedding(vocabulary_size, 16) -> Linear(16, 1) -> Sigmoid
        self.embedding_layer = nn.Embedding(vocabulary_size, 16)
        self.linear_layer = nn.Linear(16, 1)
        self.sigmoid_layer = nn.Sigmoid()
                                   
        pass

    def forward(self, x: TensorType[int]) -> TensorType[float]:
        # Hint: The embedding layer outputs a B, T, embed_dim tensor
        # but you should average it into a B, embed_dim tensor before using the Linear layer
        embedded_vector = self.embedding_layer(x)
        embedded_vector = embedded_vector.mean(axis=1)
        embedded_vector = self.linear_layer(embedded_vector)
        # Return a B, 1 tensor and round to 4 decimal places
        return self.sigmoid_layer(embedded_vector).round(decimals=4)