import torch
import torch.nn as nn

class EncoderLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers = 1, dropout = 0.0):
        super(EncoderLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers = num_layers, 
                           dropout = dropout if num_layers > 1 else 0.0, 
                           batch_first = False)
        
    def forward(self, input_seq):
        _, hidden = self.lstm(input_seq)
        return hidden

class DecoderLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers = 1, dropout = 0.0):
        super(DecoderLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers = num_layers, 
                           dropout = dropout if num_layers > 1 else 0.0, 
                           batch_first = False)
        
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, input_seq, hidden):
        output, hidden = self.lstm(input_seq, hidden)
        output = self.linear(output)
        return output, hidden
    
class Seq2Seq(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers = 1, dropout = 0.0):
        super(Seq2Seq, self).__init__()
        self.encoder = EncoderLSTM(input_size, hidden_size, num_layers, dropout)
        self.decoder = DecoderLSTM(input_size, hidden_size, output_size, num_layers, dropout)
    
    def forward(self, input_seq, target_seq_length):
        encoder_hidden = self.encoder(input_seq)

        outputs = []
        decoder_input = input_seq[-1].unsqueeze(0)
        hidden = encoder_hidden
        outputs = []
        decoder_input = input_seq[-1].unsqueeze(0)
        hidden = encoder_hidden
        for _ in range(target_seq_length):
            decoder_output, hidden = self.decoder(decoder_input, hidden)
            outputs.append(decoder_output.squeeze(0))
            decoder_input = decoder_output
        
        outputs = torch.stack(outputs, dim=0)
        return outputs.transpose(0, 1)
    
model = Seq2Seq(len(feature_columns), HIDDEN_SIZE, len(feature_columns), NUM_LAYERS)
