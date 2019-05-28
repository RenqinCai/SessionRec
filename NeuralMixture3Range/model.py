from torch import nn
import torch
import torch.nn.functional as F


class M3R(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, final_act='softmax', num_layers=1, use_cuda=True, batch_size=50, dropout_input=0, dropout_hidden=0.5, embedding_dim=-1):
        super(M3R, self).__init__()

        self.m_input_size = input_size
        self.m_hidden_size = hidden_size
        self.m_output_size = output_size

        self.m_dropout_input = dropout_input
        self.m_dropout_hidden = dropout_hidden

        self.m_embedding_dim = embedding_dim
        self.m_num_layers = num_layers

        self.m_batch_size = batch_size
        self.m_use_cuda = use_cuda

        # print("--"*10)
        self.m_device = torch.device('cuda' if use_cuda else 'cpu')
        # print("self device", self.m_device)

        # self.m_onehot_buffer = self.init_emb()

        if self.m_embedding_dim != -1:
            self.m_look_up = nn.Embedding(input_size, self.m_embedding_dim)
            self.m_gru = nn.GRU(self.m_embedding_dim, self.m_hidden_size, self.m_num_layers, dropout=self.m_dropout_hidden)
        else:
            self.m_gru = nn.GRU(self.m_input_size, self.m_hidden_size, self.m_num_layers, dropout=self.m_dropout_hidden)

        self.m_relu = torch.nn.ReLU()

        self.m_tiny_layer = nn.Linear(self.m_input_size, self.m_hidden_size)

        self.m_final_layer = nn.Tanh()
        # self.m_final_layer = nn.Softmax(dim=-1)

        self.m_h2o = nn.Linear(self.m_hidden_size, self.m_output_size)

        self = self.to(self.m_device)

    def forward(self, input, hidden):
        if self.m_embedding_dim == -1:
            embedded = input
            # embedded = self.onehot_encode(input)

            if self.training and self.m_dropout_input > 0:
                # print("training")
                # print("input", input.size())
                embedded = self.embedding_dropout(embedded)

            if len(input.size()) == 2:
                embedded = embedded.unsqueeze(0)
        
            # 
        else:
            if len(input.size()) == 2:
                embedded = input.unsqueeze(0)
            embedded = self.m_look_up(embedded)

        ### embedded size: seq_len*batch_size*embedding_size
        relu_embed = self.m_relu(embedded)
        
        ### tiny_output size: 1*batch_size*embedding
        tiny_output = relu_embed[-1, :, :]
        # print("tiny_output size", tiny_output.size())

        tiny_output = self.m_tiny_layer(tiny_output)

        gru_output, gru_hidden = self.m_gru(embedded, hidden)

        ### short_output: 1*batch_size*embedding
        short_output = gru_hidden.view(-1, gru_hidden.size(-1))

        # print(short_output.size())
        long_input = gru_hidden.view(-1, gru_hidden.size(-1))

        long_output = self.attention(gru_output, long_input)

        # output = short_output
        output = tiny_output+short_output+long_output
        # output = long_output
        #  = torch.cat((, ), -1)
        # relu_output = output
        relu_output = self.m_relu(output)

        ### relu_output: batch_size*embedding
        # relu_output = relu_output.view(-1, relu_output.size(-1))

        logit = self.m_final_layer(self.m_h2o(relu_output))
        # print("forward", logit.size())
        return logit, hidden

    def attention(self, pre_hidden, last_hidden):
        
        pre_hidden = pre_hidden.transpose(0, 1)
        cos_sim = torch.matmul(pre_hidden, last_hidden.unsqueeze(-1))
        # cos_sim = cos_sim.squeeze(-1)
        cos_sim = F.softmax(cos_sim, dim=1)
        weighted_pre = pre_hidden*cos_sim
        weightedSum_pre = torch.sum(weighted_pre, dim=1)
        # print("weighted sum pre size", weightedSum_pre.size())
        return weightedSum_pre

    def embedding_dropout(self, input):
        p_drop = None
        if len(input.size()) == 2:
            p_drop = torch.Tensor(input.size(0), 1).fill_(1-self.m_dropout_input)
        else:
            p_drop = torch.Tensor(input.size(0), input.size(1), 1).fill_(1-self.m_dropout_input)

        mask = torch.bernoulli(p_drop).expand_as(input)/(1-self.m_dropout_input)
        mask = mask.to(self.m_device)

        input = input*mask

        return input

    # def init_emb(self):
    #     self.m_window_size = 1
    #     if self.m_window_size > 1:
    #         onehot_buffer = torch.FloatTensor(self.m_window_size, self.m_batch_size, self.m_output_size)
    #         onehot_buffer = onehot_buffer.to(self.m_device)
    #     else:
    #         onehot_buffer = torch.FloatTensor(self.m_batch_size, self.m_output_size)
    #         onehot_buffer = onehot_buffer.to(self.m_device)

    #     return onehot_buffer

    # def onehot_encode(self, input):
    #     self.m_onehot_buffer.zero_()

    #     if len(input.size()) == 1:
    #         index = input.view(-1, 1)
    #         one_hot = self.m_onehot_buffer.scatter_(1, index, 1)
    #         return one_hot
    #     else:
    #         input_ = torch.unsqueeze(input, 2)
    #         one_hot = self.m_onehot_buffer.scatter_(2, input_, 1)

    #         return one_hot
        
    def init_hidden(self):
        h0 = torch.zeros(self.m_num_layers, self.m_batch_size, self.m_hidden_size).to(self.m_device)

        return h0