import torch
import torch.nn as nn
from typing import List

class Solution:

    def detect_dead_neurons(self, model: nn.Module, x: torch.Tensor) -> List[float]:
        # Forward pass through the model.
        # After each ReLU layer, compute the fraction of neurons that are dead.
        # A neuron is dead if it outputs 0 for ALL samples in the batch.
        # Return a list of dead fractions (one per ReLU layer), rounded to 4 decimals.
        current = x
        stats = []
        with torch.no_grad():
            for layer in model:
                current = layer(current)

                if isinstance(layer, nn.ReLU):
                    stats.append(round((current == 0).all(dim=0).float().mean().item(), 4))

        return stats

    def suggest_fix(self, dead_fractions: List[float]) -> str:
        # Given dead fractions per ReLU layer, suggest a fix.
        # Check in this order:
        # 1. 'use_leaky_relu' if any layer has dead fraction > 0.5
        # 2. 'reinitialize' if the first layer has dead fraction > 0.3
        # 3. 'reduce_learning_rate' if dead fraction strictly increases
        #    with depth AND the last layer's fraction > 0.1
        # 4. 'healthy' if max dead fraction < 0.1
        # 5. 'healthy' otherwise
        max_dead = 0
        for id, dead_frac in enumerate(dead_fractions):
            if dead_frac > 0.5:
                return 'use_leaky_relu'

            if id == 0:
                if dead_frac > 0.3:
                    return 'reinitialize'
        
        
        prev_dead = 0
        strict_inc = True
        for id, dead_frac in enumerate(dead_fractions):
            curr_dead = dead_frac
            if prev_dead >= curr_dead:
                strict_inc = False

            prev_dead = curr_dead
            max_dead = max(max_dead, curr_dead)
        
        if strict_inc and curr_dead > 0.1: 
            return 'reduce_learning_rate'
            
        
        return 'healthy'