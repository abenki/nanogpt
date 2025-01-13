import torch
from config import Config


class TextDataset:
    def __init__(self, file_path=Config.INPUT_DATA):
        with open(file_path, "r", encoding="utf-8") as f:
            self.text = f.read()

        # Tokenizer
        self.chars = sorted(list(set(self.text)))
        self.vocab_size = len(self.chars)

        self.str_to_int = {ch: i for i, ch in enumerate(self.chars)}
        self.int_to_str = {i: ch for i, ch in enumerate(self.chars)}

    def encode(self, string):
        """Convert string to list of integers."""
        return [self.str_to_int[c] for c in string]

    def decode(self, int_list):
        """Convert list of integers back to string."""
        return "".join([self.int_to_str[i] for i in int_list])

    def get_train_val_split(self, test_split=0.9):
        """Split data into train and validation sets."""
        data = torch.tensor(self.encode(self.text), dtype=torch.long)
        n = int(test_split * len(data))  # first 90% will be train, the rest is val
        return data[:n], data[n:]
