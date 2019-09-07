import torch


class LstmModel(torch.nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LstmModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.lstm = torch.nn.LSTM(input_dim, hidden_dim)
        self.readout = torch.nn.Linear(hidden_dim, output_dim)
        self.num_layers = 1
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

    def forward(self, x):
        batch_size = x.size(0)
        seq_first_x = x.reshape(batch_size, -1, self.input_dim).permute(1, 0, 2).contiguous()
        # seq_first_x.shape == (batch_size, sequence_length, input_dim)
        h = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=self.device)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=self.device)
        _, (h, c) = self.lstm(seq_first_x, (h, c))
        reshaped_hidden = h.view(batch_size, -1)
        return self.readout(reshaped_hidden)
