import torch.nn as nn

class LM_RNN(nn.Module):
    def __init__(self, 
                 emb_size, 
                 hidden_size, 
                 output_size,
                 pad_index=0, 
                 out_dropout=0.5,
                 emb_dropout=0.5, 
                 n_layers=2):
        super(LM_RNN, self).__init__()

        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)

        # Task 2.1 - Add dropout layer after the embedding layer
        self.dropout_emb = nn.Dropout(emb_dropout)

        # Task 1 - Replace RNN with LSTM network - nn.LSTM
        self.rnn = nn.LSTM(emb_size, hidden_size, n_layers, bidirectional=False, batch_first=True)
        self.pad_token = pad_index

        # Task 2.2 - Add dropout layer before the last linear layer
        self.dropout_out = nn.Dropout(out_dropout)
        
        # Linear layer to project the hidden layer to our output space
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, input_sequence):
        emb = self.embedding(input_sequence)

        # Appying dropout on embedding layer
        emb = self.dropout_emb(emb)

        rnn_out, _  = self.rnn(emb)

        # Appying dropout before output layer
        rnn_out = self.dropout_out(rnn_out)

        output = self.output(rnn_out).permute(0,2,1)

        return output