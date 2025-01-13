from load import load_model
from model import GPT  # Import your model class

# Path to the saved model
MODEL_PATH = "checkpoints/run_20241228_212938/model.pt"  # Replace with the correct path

# Load the model
model, metadata = load_model(MODEL_PATH, GPT)

# Ensure the model is in evaluation mode
model.eval()

# Count the total number of parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {total_params}")

# Access additional metadata if needed
print("Metadata:")
[print(f"{key}: {value}") for key, value in metadata.items()]
