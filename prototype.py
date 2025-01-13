import torch
import torch.nn as nn
from torch.nn import functional as F

# fix seed for reproducibility
torch.manual_seed(1543)

# hyperparameters
batch_size = 8  # B, nb of independent sequences processed in parallel
block_size = 8  # T, maximum context length for predictions
max_iters = 100  # number of epochs
eval_interval = 10  # interval at which we evaluate the loss
learning_rate = 1e-3
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(device)
eval_iters = 10  # nb of batches on which we evaluate losses
n_embed = 32  # C, embedding dimension, every head will be of dimension n_embed // n_head
n_head = 4  # number of self-attention heads running in parallel
n_layer = 2  # number of transformer blocks in the model
dropout = 0.2  # every forward-backward pass, 20% of all intermediate calculations are disabled and dropped to 0

# Load dataset
with open("data/input.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Tokenizer
chars = sorted(list(set(text)))
vocab_size = len(chars)

str_to_int = {ch: i for i, ch in enumerate(chars)}
int_to_str = {i: ch for i, ch in enumerate(chars)}


def encode(string):
    """
    Take a string and outputs a list of integers.
    """
    return [str_to_int[c] for c in string]


def decode(int_list):
    """
    Take a list of integers and outputs a string.
    """
    return "".join([int_to_str[i] for i in int_list])


# Train / val splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))  # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]


# data loader
def get_batch(split):
    # generate a small batch of inputs x and targets y
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


# evaluate the loss over several batches (average)
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class Head(nn.Module):
    """
    Applying 1 head of self-attention. Self-attention is the mecanism that allows tokens to communicate
    with each other.
    """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape  # here C is head_size
        k = self.key(x)  # (B,T,C)
        q = self.query(x)  # (B,T,C)

        # compute attention scores ie affinities between tokens
        weights = q @ k.transpose(-2, -1) * C**-0.5  # (B,T,C) @ (B,C,T) -> (B,T,T)
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # (B,T,T)
        weights = F.softmax(weights, dim=-1)  # (B,T,T)
        weights = self.dropout(weights)

        # weighted aggregation of the values
        v = self.value(x)  # (B,T,C)
        out = weights @ v  # (B,T,T) @ (B, T, C) -> (B,T,C)
        return out


class MultiHeadAttention(nn.Module):
    """multiple heads of self-attention running in parallel"""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.projection = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)  # concatenation over the channel dimension
        out = self.dropout(self.projection(out))
        return out


class FeedForward(nn.Module):
    """
    Simple linear layer followed by a non-linearity. This FF layer allows the tokens to "think" individually about
    the data they gathered during self-attention phase
    """

    def __init__(self, n_embed):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),  # projection layer
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.network(x)


class Block(nn.Module):
    """Transformer block: communication followed by computation"""

    def __init__(self, n_embed, n_head):
        # n_embed: embedding dimension, n_head: number of heads we want to have
        super().__init__()
        head_size = n_embed // n_head
        self.self_attention = MultiHeadAttention(n_head, head_size)  # communication
        self.feed_forward = FeedForward(n_embed)  # computation
        self.layer_norm1 = nn.LayerNorm(n_embed)
        self.layer_norm2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.self_attention(self.layer_norm1(x))
        x = x + self.feed_forward(self.layer_norm2(x))
        return x


class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(*[Block(n_embed, n_head=n_head) for _ in range(n_layer)])
        self.layer_norm_final = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        # idx and targets: (B,T) tensors of integers
        B, T = idx.shape

        # logits : score for the next character in the sequence
        tok_emb = self.token_embedding_table(idx)  # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # (T,C)
        x = tok_emb + pos_emb  # (B,T,C)
        x = self.blocks(x)  # (B,T,C)
        logits = self.lm_head(x)  # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        """The goal of generate is to take idx (B,T) and output (B,T+max_new_tokens)"""
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


# model instantiation
model = GPT()
model = model.to(device)

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# training loop
for iter in range(max_iters):  # increase number of steps for good results...
    # every once in a while evalutate loss on train and val
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch("train")

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the trained model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))
