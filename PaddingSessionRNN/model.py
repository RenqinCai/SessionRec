from torch import nn
import torch
import torch.nn.functional as F

class GRU4REC(nn.Module):
    def __init__(self, log, window_size, input_size, hidden_size, output_size, num_layers=1, final_act='tanh', dropout_hidden=.8, dropout_input=0, batch_size=50, embedding_dim=-1, use_cuda=False, shared_embedding=True):
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
        self.m_log = log
        
        self.h2o = nn.Linear(hidden_size, output_size)
            
        self.create_final_activation(final_act)

        self.look_up = nn.Embedding(input_size, self.embedding_dim).to(self.device)
        self.gru = nn.GRU(self.embedding_dim, self.hidden_size, self.num_layers, dropout=self.dropout_hidden, batch_first=True)
  
        if shared_embedding:
            msg = "share embedding"
            self.m_log.addOutput2IO(msg)
            self.out_matrix = self.look_up.weight.to(self.device)
        else:
            msg = "separate embedding"
            self.m_log.addOutput2IO(msg)
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

        embedded = input
        embedded = self.look_up(embedded)
    
        # embedded = embedded.transpose(0, 1)

        embedded_pad = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_len, batch_first=True)
        output, hidden = self.gru(embedded_pad, hidden) # (sequence, B, H)

        last_output = hidden[-1]

        last_output = last_output.view(-1, last_output.size(-1))  # (B,H)
        output = F.linear(last_output, self.out_matrix)
        logit = self.final_activation(output) ## (B, output_size)
        # logit = output
        return logit, hidden


    def onehot_encode(self, input):

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