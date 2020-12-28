from torch import nn
import torch
import torch.nn.functional as F
import numpy as np
from torch.nn.utils.rnn import pad_sequence

class GRU4REC(nn.Module):
    def __init__(self, window_size, input_size, hidden_size, output_size, device, num_layers=1, final_act='tanh', dropout_hidden=.8, dropout_input=0, batch_size=50, embedding_dim=-1, shared_embedding=True):
        super(GRU4REC, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout_hidden = dropout_hidden
        self.dropout_input = dropout_input
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.window_size = window_size
        
        self.h2o = nn.Linear(hidden_size, output_size)
            
        self.create_final_activation(final_act)

        self.look_up = nn.Embedding(input_size, self.embedding_dim)
        self.m_cate_gru = nn.GRU(self.embedding_dim, self.hidden_size, self.num_layers, dropout=self.dropout_hidden, batch_first=True)
        self.m_short_gru = nn.GRU(self.embedding_dim, self.hidden_size, self.num_layers, dropout=self.dropout_hidden, batch_first=True)


        if shared_embedding:
            print("share embedding")
            self.out_matrix = self.look_up.weight
        else:
            print("separate embedding")
            self.out_matrix = torch.rand(output_size, hidden_size, requires_grad=True)
        
        if torch.cuda.device_count() > 1:
            print("there are ", torch.cuda.device_count(), " GPUs")

            self = nn.DataParallel(self)
        else:
            print("there are ", torch.cuda.device_count(), " GPUs")

        # self = self.to(self.device)

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

    def forward(self, input, mask, max_subseqNum, max_actionNum, subseqLen_batch, seqLen_batch, device):
        '''
        Args:
            input (B,): a batch of item indices from a session-parallel mini-batch.
            target (B,): torch.LongTensor of next item indices from a session-parallel mini-batch.

        Returns:
            logit (B,C): Variable that stores the logits for the next items in the session-parallel mini-batch
            hidden: GRU hidden state
        '''

        embedded = input
        embedded = self.look_up(embedded)

        ##actionNum_subseq*batch_size*hidden_size
        # embedded = embedded.transpose(0, 1)
        # print("input_subseq_len", input_subseq_len)

        batch_size = embedded.size(0)
        
        # embedded_pad = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_subseq_len)
        hidden_subseq = self.init_hidden(batch_size, device)

        # print("embed size", embedded.size())
        # print("hidden size", hidden_subseq.size())

        ### embedded: batch_size*seq_len*hidden_size
        output_subseq, hidden_subseq = self.m_cate_gru(embedded, hidden_subseq) # (sequence, B, H)

        mask = mask.unsqueeze(-1)
         ###output_subseq: batch_size*action_num*hidden_size
        output_subseq = output_subseq*mask
        
        pad_subseqLen_batch = [i-1 if i > 0 else 0 for i in subseqLen_batch]
        first_dim_index = torch.arange(batch_size).long().to(device)
        second_dim_index = torch.LongTensor(pad_subseqLen_batch).to(device)
        
        input_seq = output_subseq[first_dim_index, second_dim_index, :]

        ### batch_size_seq*seq_len*hidden
        input_seq = input_seq.reshape(-1, max_subseqNum, output_subseq.size(-1))
        input_seq = input_seq.contiguous()
        # pad_input_seq_batch, pad_input_seq_len_batch, input_seq_index_list = self.batchify(last_output_subseq, input_subseq_index, input_subseqIndex_seq)

        # pad_input_seq_batch = pad_input_seq_batch.transpose(0, 1)

        # input_seq = input_seq.transpose(0, 1)
        batch_size_seq = input_seq.size(0)
        hidden_seq = self.init_hidden(batch_size_seq, device)

        # pad_input_seq_batch = 
        # pad_input_seq_batch = torch.nn.utils.rnn.pack_padded_sequence(pad_input_seq_batch, pad_input_seq_len_batch)

        output_seq, hidden_seq = self.m_short_gru(input_seq, hidden_seq)
        
        # output_seq = output_seq.transpose(0, 1)
        seqLen_batch = seqLen_batch - 1
        # print("seqLen_batch", seqLen_batch)
        first_dim_index =  torch.arange(batch_size_seq).long().to(device)
        second_dim_index = torch.LongTensor(seqLen_batch).to(device)
        # print("output_seq size", output_seq.size())

        last_output = output_seq[first_dim_index, second_dim_index, :]
        last_output = last_output.contiguous()
       
        output = F.linear(last_output, self.out_matrix)
        
#         logit = self.final_activation(output) ## (B, output_size)
        logit = output

        return logit


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

    def embedding_dropout(self, input, device):
        p_drop = torch.Tensor(input.size(0), input.size(1), 1).fill_(1 - self.dropout_input)  # (B,1)
        mask = torch.bernoulli(p_drop).expand_as(input) / (1 - self.dropout_input)  # (B,C)
        mask = mask.to(device)
        input = input * mask  # (B,C)

        return input

    def init_hidden(self, batch_size, device):
        '''
        Initialize the hidden state of the GRU
        '''
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        return h0