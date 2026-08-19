import torch
import torch.nn as nn
from torchtyping import TensorType
from typing import List


class Solution:
    def get_dataset(self, positive: List[str], negative: List[str]) -> TensorType[float]:
        # 1. Build vocabulary: collect all unique words, sort them, assign integer IDs starting at 1
        # 2. Encode each sentence by replacing words with their IDs
        # 3. Combine positive + negative into one list of tensors
        # 4. Pad shorter sequences with 0s using nn.utils.rnn.pad_sequence(tensors, batch_first=True)

        # merge the pos and neg lists
        sentences = positive + negative

        unique_words = {}
        for sentence in sentences:
            for word in sentence.split():
                unique_words[word] = []

        unique_words = sorted(unique_words)

        vocab = {}
        for id, word in enumerate(unique_words, start=1):
            vocab[word] = id

        encoded_sentences = []

        for sentence in sentences:
            encoded_sentences.append(torch.tensor([vocab[word] for word in sentence.split()], dtype=torch.long))

        return nn.utils.rnn.pad_sequence(encoded_sentences, batch_first=True, padding_value=0)
        

