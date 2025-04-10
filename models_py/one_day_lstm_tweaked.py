import torch
import torch.nn as nn

class LSTMPrediction(nn.Module):
	def __init__(self, input_size, hidden_size, num_layers, output_size, dropout):
		super(LSTMPrediction, self).__init__()

		self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0)

		self.fc = nn.Linear(hidden_size, output_size)

	def forward(self, x):
		output, _ = self.lstm(x)
		output = self.fc(output[:, -1, :])
		return output