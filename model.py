import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_dim, embedded_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        
        # Initialize Embedded Layer.
        self.embedded_layer = nn.Embedding(input_dim, embedded_dim)

        # Initialize LSTM.
        self.rnn = nn.LSTM(embedded_dim, hidden_dim, num_layers, dropout=dropout)

        # Initialize Dropout Layer.
        self.dropout= nn.Dropout(dropout)

    def forward(self, src):
        # Embedding Input Tokens
        embedded = self.dropout(self.embedded_layer(src))

        # Embedded Input -> LSTM.
        outputs, (hidden, cell) = self.rnn(embedded)

        return hidden, cell
