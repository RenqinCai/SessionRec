from torch import nn
import torch
import torch.nn.functional as F

class GRU4REC(nn.Module):
    def __init__(self, window_size, input_size, hidden_size, output_size, num_layers=1, final_act='tanh', dropout_hidden=.8, dropout_input=0, batch_size=50, embedding_dim=-1, use_cuda=False, shared_embedding=True):
        super(GRU4REC, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout_hidden = dropout_hidden
        self.dropout_input = dropout_input
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.use_cuda = use_cuda
        self.window_size = window_size
        self.device = torch.device('cuda' if use_cuda else 'cpu')

        
        self.h2o = nn.Linear(hidden_size, output_size)
            
        self.create_final_activation(final_act)

#         if self.embedding_dim != -1:
        self.look_up = nn.Embedding(input_size, self.embedding_dim).to(self.device)
        self.gru = nn.GRU(self.embedding_dim, self.hidden_size, self.num_layers, dropout=self.dropout_hidden)
#         else:
#             self.onehot_buffer = self.init_emb()
#             self.embed = torch.FloatTensor(self.window_size, self.batch_size, self.output_size)
#             self.embed = onehot_buffer.to(self.device)   
#             self.look_up = self.onehot_encode
#             self.gru = nn.GRU(self.input_size, self.hidden_size, self.num_layers, dropout=self.dropout_hidden)
            
        if shared_embedding:
            print("share embedding")
            self.out_matrix = self.look_up.weight.to(self.device)
        else:
            print("separate embedding")
            self.out_matrix = torch.rand(output_size, hidden_size, requires_grad=True).to(self.device)
            
        self = self.to(self.device)

    def create_final_activation(self, final_act):
        if final_act == 'tanh':
            self.final_activation = nn.Tanh()
        elif final_act == 'relu':
            self.final_activation = nn.ReLU()
        elif final_act == 'softmax':
            self.final_activation = nn.Softmax()
        elif final_act == 'softmax_logit':
            self.final_activation = nn.LogSoftmax()
        elif final_act.startswith('elu-'):
            self.final_activation = nn.ELU(alpha=float(final_act.split('-')[1]))
        elif final_act.startswith('leaky-'):
            self.final_activation = nn.LeakyReLU(negative_slope=float(final_act.split('-')[1]))

    def forward(self, input, hidden, input_len):
        '''
        Args:
            input (B,): a batch of item indices from a session-parallel mini-batch.
            target (B,): torch.LongTensor of next item indices from a session-parallel mini-batch.

        Returns:
            logit (B,C): Variable that stores the logits for the next items in the session-parallel mini-batch
            hidden: GRU hidden state
        '''

    
            # if len(input.size()) == 2:
            #     embedded = input.unsqueeze(0)
            # print("embedding size", embedded.size())
        embedded = input
        embedded = self.look_up(embedded)
    
        embedded = embedded.transpose(0, 1)

        # print("embedded size", embedded.size(), input_len.shape)
        embedded_pad = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_len)
        # print("embedded_pad size", embedded_pad.size())
        output, hidden = self.gru(embedded_pad, hidden) # (sequence, B, H)

        
        # output, _ = torch.nn.utils.rnn.pad_packed_sequence(output)
        # output = output.contiguous()
        # print("output size", output.size())

        # last_output = output[-1, :, :]
        # print("lastoutput size", last_output.size())
        last_output = hidden[-1]
        # print("last_output size", last_output.size())

        last_output = last_output.view(-1, last_output.size(-1))  # (B,H)
        output = F.linear(last_output, self.out_matrix)
#         logit = self.final_activation(output) ## (B, output_size)
        logit = output
        return logit, hidden


    def onehot_encode(self, input):
        """
        Returns a one-hot vector corresponding to the input

        Args:
            input (B,): torch.LongTensor of item indices
            buffer (B,output_size): buffer that stores the one-hot vector
        Returns:
            one_hot (B,C): torch.FloatTensor of one-hot vectors
        """

        self.onehot_buffer.zero_()

        index = input.unsqueeze(2)
        # index = input.view(-1, 1)
        one_hot = self.onehot_buffer.scatter_(2, index, 1)

        return one_hot

    def embedding_dropout(self, input):
        p_drop = torch.Tensor(input.size(0), input.size(1), 1).fill_(1 - self.dropout_input)  # (B,1)
        mask = torch.bernoulli(p_drop).expand_as(input) / (1 - self.dropout_input)  # (B,C)
        mask = mask.to(self.device)
        input = input * mask  # (B,C)

        return input

    def init_hidden(self):
        '''
        Initialize the hidden state of the GRU
        '''
        h0 = torch.zeros(self.num_layers, self.batch_size, self.hidden_size).to(self.device)
        return h0