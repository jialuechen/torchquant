import torch
import torch.nn as nn

class Seq2SeqPDESolver(nn.Module):
    """
    Sequence-to-Sequence model for solving Partial Differential Equations (PDEs).

    This model uses an encoder-decoder architecture with LSTM layers to approximate
    solutions to PDEs. It can be used for various financial applications such as
    option pricing or interest rate modeling.
    """

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        """
        Initialize the Seq2SeqPDESolver.

        Args:
            input_dim (int): Dimension of the input features.
            hidden_dim (int): Dimension of the hidden state in LSTM layers.
            output_dim (int): Dimension of the output (solution) at each time step.
            num_layers (int): Number of LSTM layers in both encoder and decoder.
        """
        super(Seq2SeqPDESolver, self).__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.decoder = nn.LSTM(output_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        """
        Forward pass of the Seq2SeqPDESolver.

        Args:
            src (Tensor): Source sequence representing initial/boundary conditions.
                          Shape: (batch_size, src_seq_len, input_dim)
            trg (Tensor): Target sequence representing the expected solution trajectory.
                          Shape: (batch_size, trg_seq_len, output_dim)
            teacher_forcing_ratio (float): Probability of using teacher forcing during training.

        Returns:
            Tensor: Predicted solution trajectory. Shape: (batch_size, trg_seq_len, output_dim)
        """
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