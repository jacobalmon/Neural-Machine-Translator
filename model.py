import torch.nn as nn
import torch


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
    
class Decoder(nn.Module):
    def __init__(self, output_dim, embedded_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        
        # Initialize Embedded Layer.
        self.embedded_layer = nn.Embedding(output_dim, embedded_dim)

        # Initialize LSTM.
        self.rnn = nn.LSTM(embedded_dim, hidden_dim, num_layers, dropout=dropout)

        # Initialize Output Layer.
        self.output_layer = nn.Linear(hidden_dim, output_dim)

        # Initialize Dropout Layer.
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell):
        input = input.unsqueeze(0)

        # Embedding Input Tokens
        embedded = self.dropout(self.embedded_layer(input))

        # Embedded Input -> LSTM.
        outputs, (hidden, cell) = self.rnn(embedded)

        # Mapping LSTM Output.
        prediction = self.output_layer(outputs.squeeze(0))

        return prediction, hidden, cell
    
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, force_ratio=0.5):
        # Force Ratio - Probability to use force.

        trg_length = trg.shape[0]
        batch_size = trg.shape[1]
        trg_voab_size = self.decoder.output_layer.out_features

        # Tensor that stores decoder outputs.
        outputs = torch.zeros(trg_length, batch_size, trg_voab_size).to(self.device)

        # Encoder Forwards.
        hidden, cell = self.encoder(src)
        
        # First input to the decoder is the sos token.
        input = trg[0, :]

        for t in range(1, trg_length):
            # Decoder Forwards.
            output, hidden, cell = self.decoder(input, hidden, cell)

            # Store Output.
            outputs[t] = output

            # Get Highest Predicted Token from Output.
            highest_prob = output.argmax(1)

            # Decide to use force or not.
            input = trg[t] if torch.rand(1).item() < force_ratio else highest_prob

        return outputs