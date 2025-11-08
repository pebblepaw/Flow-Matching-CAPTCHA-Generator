import string
import numpy as np

# Random seed for reproducibility, 42 is the answer to the universe
SEED = np.random.randint(42) 

# Character mapping
IDX_TO_CHAR = string.digits + string.ascii_uppercase
CHAR_TO_IDX = {char: idx for idx, char in enumerate(IDX_TO_CHAR)}
