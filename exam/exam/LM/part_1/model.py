import torch.nn as nn

class LM_RNN(nn.Module):
    def __init__(self, 
                 emb_size, 
                 hidden_size, 
                 output_size, 
                 pad_index=0, 
                 out_dropout=0.1,
                 emb_dropout=0.1, 
                 n_layers=1):
        super(LM_RNN, self).__init__()

        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        # Pytorch's RNN layer: https://pytorch.org/docs/stable/generated/torch.nn.RNN.html

        # TO DO 1.1
        self.rnn = nn.LSTM(emb_size, hidden_size, n_layers, bidirectional=False, batch_first=True)
        self.pad_token = pad_index
        
        # Linear layer to project the hidden layer to our output space
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, input_sequence):
        emb = self.embedding(input_sequence)
        rnn_out, _  = self.rnn(emb)
        output = self.output(rnn_out).permute(0,2,1)
        return output
    



# import torch.nn as nn

# class LM_RNN(nn.Module):
#     def __init__(self, 
#                  emb_size, 
#                  hidden_size, 
#                  output_size, 
#                  pad_index=0, 
#                  out_dropout=0.1,
#                  emb_dropout=0.1, 
#                  n_layers=1):
#         super(LM_RNN, self).__init__()

#         self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
#         # Pytorch's RNN layer: https://pytorch.org/docs/stable/generated/torch.nn.RNN.html

#         # 2.1
#         self.dropoutEmb = nn.Dropout(emb_dropout)

#         # TO DO 1.1
#         self.rnn = nn.LSTM(emb_size, hidden_size, n_layers, bidirectional=False, batch_first=True)
#         self.pad_token = pad_index

#         # 2.2
#         self.dropoutOut = nn.Dropout(out_dropout)
        
#         # Linear layer to project the hidden layer to our output space
#         self.output = nn.Linear(hidden_size, output_size)

#     def forward(self, input_sequence):
#         emb = self.embedding(input_sequence)

#         # 2.3
#         dropoutEmb = self.dropoutEmb(emb)
#         # rnn_out, _  = self.rnn(emb)

#         # 2.4
#         rnn_out, _  = self.rnn(dropoutEmb)  # rnn_out, _  = self.rnn(emb) # change 2.1

#         # 2.5
#         dropoutOut = self.dropoutEmb(rnn_out)

#         # 2.6
#         output = self.output(dropoutOut).permute(0,2,1) # output = self.output(rnn_out).permute(0,2,1) # change 2.2
#         # output = self.output(rnn_out).permute(0,2,1)
#         return output