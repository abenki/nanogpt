import os
from datetime import datetime
import torch
import torch.optim as optim
from config import Config
from model import GPT
from preprocessing import TextDataset


def get_batch(data, batch_size=Config.BATCH_SIZE, block_size=Config.BLOCK_SIZE):
    """Generate a small batch of inputs x and targets y."""
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    x, y = x.to(Config.DEVICE), y.to(Config.DEVICE)
    return x, y


@torch.no_grad()
def estimate_loss(model, train_data, val_data):
    """Estimate loss on train and val sets over several batches."""
    out = {}
    model.eval()
    for split, data in [("train", train_data), ("val", val_data)]:
        losses = torch.zeros(Config.EVAL_ITERS)
        for k in range(Config.EVAL_ITERS):
            X, Y = get_batch(data)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


def save_model(model, dataset, losses, save_dir="checkpoints"):
    """Save model, vocabulary, and training metrics."""
    # Create checkpoint directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Generate timestamp for unique model naming
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create checkpoint directory for this run
    run_dir = os.path.join(save_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    # Save model state
    model_path = os.path.join(run_dir, "model.pt")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "vocab_size": dataset.vocab_size,
            "str_to_int": dataset.str_to_int,
            "int_to_str": dataset.int_to_str,
            "training_losses": losses,
            "config": {
                "n_embed": Config.N_EMBED,
                "n_head": Config.N_HEAD,
                "n_layer": Config.N_LAYER,
                "dropout": Config.DROPOUT,
                "block_size": Config.BLOCK_SIZE,
            },
        },
        model_path,
    )

    print(f"Model saved to {model_path}")
    return model_path


def train_model():
    print("Training on device: ", Config.DEVICE)
    # Load and prepare dataset
    dataset = TextDataset()
    train_data, val_data = dataset.get_train_val_split()

    # Initialize model
    model = GPT(vocab_size=dataset.vocab_size).to(Config.DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)

    # Track losses for saving
    training_losses = {"train": [], "val": []}

    # Training loop
    for iter in range(Config.EPOCHS):
        # Periodic evaluation
        if iter % Config.EVAL_INTERVAL == 0:
            losses = estimate_loss(model, train_data, val_data)
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            # Store losses
            training_losses["train"].append(losses["train"])
            training_losses["val"].append(losses["val"])

        # Training step
        xb, yb = get_batch(train_data)
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    model_path = save_model(model, dataset, training_losses)

    return model, dataset, model_path


if __name__ == "__main__":
    train_model()
