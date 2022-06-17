import torch.nn as nn
import torch

class LSTM(nn.Module):
    def __init__(self, n_mel_channels,
                 symbols_embedding_dim, n_layers):
        super(LSTM, self).__init__()
        self.n_layers = n_layers
        self.symbols_embedding_dim = symbols_embedding_dim
        self.lstm = nn.LSTM(input_size=13,
                            hidden_size=symbols_embedding_dim,
                            num_layers=n_layers,
                            batch_first=True)

        self.proj = nn.Linear(symbols_embedding_dim, n_mel_channels, bias=True)
        # self.local_window_size = local_window_size

    def default_init(self, bs):
        return torch.randn(1*self.n_layers, bs, self.symbols_embedding_dim), torch.randn(1*self.n_layers, bs, self.symbols_embedding_dim)

    def forward(self, inputs, use_gt_pitch=True, pace=1.0, max_duration=75,
                apply_mask=True, dec_cond=None):

        (inputs, hidden, cell, mel_lens,
         speaker) = inputs
        mem_out, (hidden_out, cell_out) = self.lstm(inputs, (hidden.to(inputs.device), cell.to(inputs.device)))
        mel_out = self.proj(mem_out)
        return mel_out, hidden_out, cell_out
