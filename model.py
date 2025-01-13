import torch
import torch.nn as nn
from torch.nn import functional as F
from config import Config


class Head(nn.Module):
    """
    Single head of self-attention. Self-attention is the mecanism that allows tokens to communicate
    with each other.
    """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(Config.N_EMBED, head_size, bias=False)
        self.query = nn.Linear(Config.N_EMBED, head_size, bias=False)
        self.value = nn.Linear(Config.N_EMBED, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(Config.BLOCK_SIZE, Config.BLOCK_SIZE)))
        self.dropout = nn.Dropout(Config.DROPOUT)

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
    """Multiple heads of self-attention running in parallel."""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.projection = nn.Linear(Config.N_EMBED, Config.N_EMBED)
        self.dropout = nn.Dropout(Config.DROPOUT)

    def forward(self, x):
        out = torch.cat(
            [h(x) for h in self.heads], dim=-1
        )  # concatenation over the channel dimension
        out = self.dropout(self.projection(out))
        return out


class FeedForward(nn.Module):
    """
    Simple linear layer followed by a non-linearity. This FF layer allows the tokens to "think"
    individually about the data they gathered during self-attention phase.
    """

    def __init__(self, n_embed):
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),  # projection layer
            nn.Dropout(Config.DROPOUT),
        )

    def forward(self, x):
        return self.ff(x)


class Block(nn.Module):
    """Transformer block: communication followed by computation."""

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
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, Config.N_EMBED)
        self.position_embedding_table = nn.Embedding(Config.BLOCK_SIZE, Config.N_EMBED)
        self.blocks = nn.Sequential(
            *[Block(Config.N_EMBED, n_head=Config.N_HEAD) for _ in range(Config.N_LAYER)]
        )
        self.layer_norm_final = nn.LayerNorm(Config.N_EMBED)
        self.lm_head = nn.Linear(Config.N_EMBED, vocab_size)

    def forward(self, idx, targets=None):
        # idx and targets: (B,T) tensors of integers
        B, T = idx.shape

        # logits : score for the next character in the sequence
        tok_emb = self.token_embedding_table(idx)  # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=Config.DEVICE))  # (T,C)
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
        """Take idx of shape (B,T) and output (B,T+max_new_tokens)"""
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -Config.BLOCK_SIZE:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last timestep
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx
