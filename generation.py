import torch
import typer
from config import Config
from load import load_model
from model import GPT
from preprocessing import TextDataset

app = typer.Typer()


def generate_text(model, max_tokens=500):
    """Generate text from a trained model."""
    model.eval()

    context = torch.zeros((1, 1), dtype=torch.long, device=Config.DEVICE)

    # Generate text
    with torch.no_grad():
        generated_tokens = model.generate(context, max_new_tokens=max_tokens)

    # Decode and return generated text
    dataset = TextDataset()
    return dataset.decode(generated_tokens[0].tolist())


@app.command()
def main(
    checkpoint_path: str = typer.Option(None, help="Path to the pre-trained model file"),
    max_tokens: int = typer.Option(500, help="Maximum number of tokens to generate"),
):
    model, metadata = load_model(checkpoint_path, GPT)
    model.to(Config.DEVICE)

    generated_text = generate_text(model, max_tokens)
    print(f"Generated Text:\n{generated_text}")


if __name__ == "__main__":
    app()
