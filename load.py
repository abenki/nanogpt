import torch
from preprocessing import TextDataset


def load_model(model_path, model_class):
    """
    Load a model and its associated metadata from a .pt file.

    Args:
        model_path (str): Path to the .pt file.
        model_class (class): The class of the model to be instantiated.

    Returns:
        model (torch.nn.Module): The loaded model.
        metadata (dict): A dictionary containing the additional metadata.
    """
    # Load the checkpoint
    checkpoint = torch.load(model_path, weights_only=True)

    # Reconstruct the model
    model = model_class(vocab_size=TextDataset().vocab_size)  # Pass the saved config as kwargs
    model.load_state_dict(checkpoint["model_state_dict"])

    # Return the model and metadata
    metadata = {
        "vocab_size": checkpoint["vocab_size"],
        "str_to_int": checkpoint["str_to_int"],
        "int_to_str": checkpoint["int_to_str"],
        "training_losses": checkpoint["training_losses"],
    }
    return model, metadata
