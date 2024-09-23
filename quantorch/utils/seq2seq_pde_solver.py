import torch
import torch.nn as nn
import torch.optim as optim

class Seq2SeqPDESolver(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        super(Seq2SeqPDESolver, self).__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.decoder = nn.LSTM(output_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size, trg_len, _ = trg.size()
        outputs = torch.zeros(batch_size, trg_len, trg.size(2)).to(src.device)
        
        # Encode the source sequence
        encoder_outputs, (hidden, cell) = self.encoder(src)
        
        # Initialize decoder input as the first target value
        input = trg[:, 0, :]

        for t in range(1, trg_len):
            # Decode step
            input = input.unsqueeze(1)  # Add sequence dimension
            output, (hidden, cell) = self.decoder(input, (hidden, cell))
            prediction = self.fc(output.squeeze(1))
            outputs[:, t, :] = prediction

            # Decide whether to use teacher forcing
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            input = trg[:, t, :] if teacher_force else prediction
        return outputs