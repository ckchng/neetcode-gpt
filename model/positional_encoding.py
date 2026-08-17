import numpy as np
from numpy.typing import NDArray


class Solution:
    def get_positional_encoding(self, seq_len: int, d_model: int) -> NDArray[np.float64]:
        # PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
        # PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))
        #
        # Hint: Use np.arange() to create position and dimension index vectors,
        # then compute all values at once with broadcasting (no loops needed).
        # Assign sine to even columns (PE[:, 0::2]) and cosine to odd columns (PE[:, 1::2]).
        # Round to 5 decimal places.
        
        # first create the pos and i 
        pos = np.arange(seq_len, dtype=np.float64)[:, None]
        dimensions = np.arange(d_model)

        exponents = 2 * (dimensions // 2) / d_model
        angles = pos / 10000**(exponents)

        PE = np.empty((seq_len, d_model), dtype=np.float64)
        PE[:, 0::2] = np.sin(angles[:, 0::2])
        PE[:, 1::2] = np.cos(angles[:, 1::2])

        return np.round(PE, 5)
    