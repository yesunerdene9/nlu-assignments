import torch.nn as nn
from functions import *

# Weight Tying

class LM_RNN(nn.Module):
    def __init__(self, 
                 emb_size, 
                 hidden_size, 
                 output_size, 
                 pad_index=0,
                 emb_dropout=0.1, 
                 n_layers=2):
        super(LM_RNN, self).__init__()

        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)

        # Applying Variational dropout - creating custom dropout layer with a 
        # self.dropout = VariationalDropout(emb_dropout)
    
        self.rnn = nn.LSTM(emb_size, hidden_size, n_layers, bidirectional=False, batch_first=True)
        self.pad_token = pad_index
        
        self.output = nn.Linear(hidden_size, output_size)
        
        # Applying weight tying - share the same weight on the output layer as the embedding layer
        self.output.weight = self.embedding.weight

    def forward(self, input_sequence):
        emb = self.embedding(input_sequence)

        # emb = self.dropout(emb)
        
        rnn_out, _  = self.rnn(emb)
        output = self.output(rnn_out).permute(0,2,1)
        return output