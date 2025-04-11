import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
	def __init__(self, input_size, day_input_size, month_input_size, hidden_size, num_layers, dropout):
		super(Encoder, self).__init__()

		self.open = nn.Linear(1, input_size)
		self.high = nn.Linear(1, input_size)
		self.low = nn.Linear(1, input_size)
		self.volume = nn.Linear(1, input_size)
		self.dayOfWeek = nn.Embedding(7, day_input_size)
		self.month = nn.Embedding(12, month_input_size)

		combined_input_size = input_size * 4 + day_input_size + month_input_size
		self.lstm = nn.LSTM(combined_input_size, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0)

	def forward(self, x):
		x_open, x_high, x_low, x_volume, x_dayofweek, x_month = x
		open = F.relu(self.open(x_open))
		high = F.relu(self.high(x_high))
		low = F.relu(self.low(x_low))
		volume = F.relu(self.volume(x_volume))

		x_dayofweek = x_dayofweek.squeeze(-1).long()
		x_month = x_month.squeeze(-1).long()

		day = self.dayOfWeek(x_dayofweek)
		month = self.month(x_month)

		features = torch.cat((open, high, low, volume, day, month), dim=2)

		_, (hidden, cell) = self.lstm(features)
		return hidden, cell
	
class Decoder(nn.Module):
	def __init__(self, input_size, hidden_size, output_size, num_layers, dropout):
		super(Decoder, self).__init__()

		self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
		self.fc = nn.Linear(hidden_size, output_size)

	def forward(self, x, hidden, cell):
		output, (hidden, cell) = self.lstm(x, (hidden, cell))
		prediction = self.fc(output)

		return prediction, hidden, cell
	
class Seq2SeqPrediction(nn.Module):
	def __init__(self, input_size, day_size, month_size, hidden_size, num_layers, dropout):
		super(Seq2SeqPrediction, self).__init__()

		self.encoder = Encoder(input_size, day_size, month_size, hidden_size, num_layers, dropout)
		self.decoder = Decoder(1, hidden_size, 1, num_layers, dropout)

	def forward(self, x, prediction_length):
		hidden, cell = self.encoder(x)

		decoder_input = torch.zeros(x[0].size(0), 1, 1)
		decoder_input = decoder_input

		outputs = []

		for _ in range(prediction_length):
			decoder_output, hidden, cell = self.decoder(decoder_input, hidden, cell)
			outputs.append(decoder_output)
			decoder_input = decoder_output

		return torch.cat(outputs, dim=1)