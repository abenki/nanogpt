import torch
from config import Config
from preprocessing import TextDataset
from model import GPT

def load_model(model_path=None):
    """Load a pre-trained model."""
    dataset = TextDataset()
    model = GPT(vocab_size=dataset.vocab_size).to(Config.DEVICE)

    if model_path:
        model.load_state_dict(torch.load(model_path))

    return model, dataset

def generate_text(model, dataset, prompt=None, max_tokens=500):
    """Generate text from a trained model."""
    model.eval()

    # If no prompt is provided, start with zero tensor
    if prompt is None:
        context = torch.zeros((1, 1), dtype=torch.long, device=Config.DEVICE)
    else:
        # Encode the prompt and convert to tensor
        context = torch.tensor(dataset.encode(prompt), dtype=torch.long, device=Config.DEVICE).unsqueeze(0)

    # Generate text
    with torch.no_grad():
        generated_tokens = model.generate(context, max_new_tokens=max_tokens)

    # Decode and return generated text
    return dataset.decode(generated_tokens[0].tolist())

def main():
    model, dataset = load_model()

    # Optional: Generate with a specific prompt
    prompt = "Once upon a time"
    generated_text = generate_text(model, dataset, prompt)
    print(f"Generated Text:\n{generated_text}")

if __name__ == "__main__":
    main()
