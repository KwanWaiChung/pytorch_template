import torch.nn as nn


class LSTM(nn.Module):
    """A basic LSTM model for quick testing.

    """

    def __init__(
        self, vocab_size, embedding_dim, hidden_dim, n_layers, n_classes
    ):
        super().__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_dim
        self.embed = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            n_layers,
            batch_first=True,
            bidirectional=False,
        )
        self.linear = nn.Linear(hidden_dim, n_classes)

        # initialize weight
        nn.init.xavier_uniform_(self.embed.weight)
        nn.init.xavier_uniform_(self.linear.weight)
        self.linear.bias.data.fill_(0)

    def forward(self, inputs):
        """
        Returns:
            The hidden state of the last layer in the lsat step.
        """
        embeds = self.embed(inputs)
        out, hidden = self.lstm(embeds)
        return self.linear(out[:, -1, :])


class LinearRegression(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        out = self.linear(x)
        return out
