import torch


# Hyperparameters and Configuration
class Config:
    # Training Configurations
    BATCH_SIZE = 64  # B, nb of independent sequences processed in parallel
    BLOCK_SIZE = 256  # T, maximum context length for predictions
    EPOCHS = 5000  # number of epochs
    EVAL_INTERVAL = 500  # interval at which we evaluate the loss
    EVAL_ITERS = 200  # nb of batches on which we evaluate losses
    LEARNING_RATE = 3e-4

    # Model Architecture
    N_EMBED = 384  # C, embedding dimension, every head will be of dimension n_embed // n_head
    N_HEAD = 6  # number of self-attention heads running in parallel
    N_LAYER = 6  # number of transformer blocks in the model
    DROPOUT = 0.2  # every forward-backward pass, 20% of all intermediate calculations are disabled and dropped to 0

    # Device Configuration
    DEVICE = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    # Data Paths
    INPUT_DATA = "data/input.txt"

    @classmethod
    def get_device(cls):
        return cls.DEVICE
