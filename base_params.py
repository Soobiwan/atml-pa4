import torch
import random
import numpy as np


torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

NUM_CLIENTS = 10      # Use 10 clients
TOTAL_TRAIN_SAMPLES = 50000 # CIFAR-10 has 50k training images
SAMPLES_PER_CLIENT = TOTAL_TRAIN_SAMPLES // NUM_CLIENTS # 5000 each
LEARNING_RATE = 0.01
NUM_ROUNDS = 20       
LOCAL_BATCH_SIZE = 1024
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

#task 2
# # Experiment 1 Params (Vary K)
# K_VALUES = [1, 5, 10, 20]      #
# FRAC_K_EXP = 1.0              # Full participation


# # Experiment 2 Params (Vary Fraction)
# FRAC_VALUES = [1.0, 0.5, 0.2]  #
# K_FRAC_EXP = 5                # Fixed K=5

#task 3
K_VALUE = 5
K_FRAC_EXP = 1.0
ALPHAS = [100.0, 50.0, 10.0, 1.0, 0.5, 0.1]
