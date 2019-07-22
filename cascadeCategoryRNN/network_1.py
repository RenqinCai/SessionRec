from torch import nn
import torch
import torch.nn.functional as F
import numpy as np
from torch.nn.utils.rnn import pad_sequence


### input_cate_batch: action_cate_long_batch
### cate_subseq_batch: cate_long_batch

### mask_cate_batch: action_cate_mask_long_batch
### mask_cate_seq_batch: cate_mask_long_batch

### max_actionNum_cate_batch: max_actionNum_cate_long_batch
### max_cateNum_batch: max_cateNum_long_batch

### subseqLen_cate_batch: actionNum_cate_long_batch
### seqLen_cate_batch: cateNum_long_batch

### input_batch: action_short_batch
### cate_batch: cate_short_batch

### mask_batch: action_mask_short_batch

### seqLen_batch: actionNum_short_batch


class NN4REC(nn.Module):
    def __init__(self, log, ss, input_size, hidden_size, output_size, embedding_dim, cate_input_size, cate_output_size, cate_embedding_dim, cate_hidden_size, num_layers=1, final_act='tanh', dropout_hidden=.8, dropout_input=0, use_cuda=False, shared_embedding=True, cate_shared_embedding=True):
        super(NN4REC, self).__init__()
        self.m_log = log
        self.device = torch.device('cuda' if use_cuda else 'cpu')

        self.m_itemNN = ITEMNN(input_size, hidden_size, output_size, embedding_dim, num_layers, final_act, dropout_hidden, dropout_input, use_cuda, shared_embedding)
        self.m_cateNN = CATENN(cate_input_size, cate_output_size, cate_embedding_dim, cate_hidden_size, num_layers, final_act, dropout_hidden, dropout_input, use_cuda, cate_shared_embedding)

        self.fc = nn.Linear(hidden_size*2+cate_hidden_size, hidden_size)
        self.m_ss = ss

        if shared_embedding:
            message = "share embedding"
            self.m_log.addOutput2IO(message)
            self.m_ss.params.weight = self.m_itemNN.look_up.weight

        self = self.to(self.device)

    def forward(self, action_cate_long_batch, cate_long_batch, action_cate_mask_long_batch, cate_mask_long_batch, max_actionNum_cate_long_batch, max_cateNum_long_batch, actionNum_cate_long_batch, cateNum_long_batch, action_short_batch, cate_short_batch, action_mask_short_batch, actionNum_short_batch, y_cate_batch, train_test_flag):
        seq_cate_input, seq_short_input = self.m_itemNN(action_cate_long_batch, cate_long_batch, action_cate_mask_long_batch, cate_mask_long_batch, max_actionNum_cate_long_batch, max_cateNum_long_batch, actionNum_cate_long_batch, cateNum_long_batch, action_short_batch, action_mask_short_batch, actionNum_short_batch, train_test_flag)

        logit_cate_short = self.m_cateNN(cate_short_batch, action_mask_short_batch, actionNum_short_batch, train_test_flag)

        cate_prob_mask = F.softmax(logit_cate_short, dim=-1)

        short_batch_size = cate_short_batch.size(0)

        first_dim_index = torch.arange(short_batch_size).to(self.device).reshape(-1, 1)
        second_dim_index = torch.from_numpy(cate_long_batch).to(self.device)

        ### weight_normalized: batch_size*subseq_num
        weight_long = cate_prob_mask[first_dim_index, second_dim_index]

        # ### weight_normalized: batch_size*subseq_num
        weight_mask_long = weight_long*cate_mask_long_batch

        weight_mask_sum_long = torch.sum(weight_mask_long, dim=1)
        weight_mask_sum_long = weight_mask_sum_long.unsqueeze(-1)
        weight_mask_normalized_long = weight_mask_long/weight_mask_sum_long

        weight_mask_normalized_long = weight_mask_normalized_long.unsqueeze(-1)
        weighted_seq_cate_input= weight_mask_normalized_long*seq_cate_input
      
        sum_seq_cate_input = torch.sum(weighted_seq_cate_input, dim=1)

        seq_short_input = seq_short_input.squeeze()

        ### cate embedding mix
        y_cate_embedded = self.m_itemNN.look_up(y_cate_batch)
        print("y_cate embed", y_cate_embedded.size())

        mixture_output = torch.cat((sum_seq_cate_input, seq_short_input, y_cate_embedded), dim=1)
        fc_output = self.fc(mixture_output)

        return fc_output, logit_cate_short

class ITEMNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, embedding_dim, num_layers=1, final_act='tanh', dropout_hidden=.8, dropout_input=0,use_cuda=False, shared_embedding=True):
        super(ITEMNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout_hidden = dropout_hidden
        self.dropout_input = dropout_input
        self.embedding_dim = embedding_dim
        self.use_cuda = use_cuda
        self.device = torch.device('cuda' if use_cuda else 'cpu')

        self.look_up = nn.Embedding(input_size, self.embedding_dim)
        self.m_cate_session_gru = nn.GRU(self.embedding_dim, self.hidden_size, self.num_layers, dropout=self.dropout_hidden, batch_first=True)
        self.m_short_gru = nn.GRU(self.embedding_dim, self.hidden_size, self.num_layers, dropout=self.dropout_hidden, batch_first=True)
        

    def forward(self, action_cate_long_batch, cate_long_batch, action_cate_mask_long_batch, cate_mask_long_batch, max_actionNum_cate_long_batch, max_cateNum_long_batch, actionNum_cate_long_batch, cateNum_long_batch, action_short_batch, action_mask_short_batch, actionNum_short_batch, train_test_flag):

        action_cate_input = action_cate_long_batch
        action_cate_embedded = self.look_up(action_cate_input)

        action_cate_batch_size = action_cate_embedded.size(0)
        action_cate_hidden = self.init_hidden(action_cate_batch_size, self.hidden_size)

        ### embedded: batch_size*seq_len*hidden_size
        action_cate_output,  action_cate_hidden = self.m_cate_session_gru(action_cate_embedded, action_cate_hidden) # (sequence, B, H)

        action_cate_mask = action_cate_mask_long_batch.unsqueeze(-1)

        # output_subseq_cate 
        action_cate_output_mask = action_cate_output*action_cate_mask
        
        # pad_subseqLen_cate_batch = np.array([i-1 if i > 0 else 0 for i in subseqLen_cate_batch])
        first_dim_index = torch.arange(action_cate_batch_size).to(self.device)
        second_dim_index = torch.from_numpy(actionNum_cate_long_batch).to(self.device)
        
        seq_cate_input = action_cate_output_mask[first_dim_index, second_dim_index, :]

        ### batch_size_seq*seq_len*hidden_size
        seq_cate_input = seq_cate_input.reshape(-1, max_cateNum_long_batch, action_cate_output_mask.size(-1))
        seq_cate_input = seq_cate_input.contiguous()
        
        #####################
        ### get the output from 5 latest actions
        action_short_input = action_short_batch
        action_short_embedded = self.look_up(action_short_input)

        short_batch_size = action_short_embedded.size(0) 
        
        action_short_hidden = self.init_hidden(short_batch_size, self.hidden_size)
        action_short_output, action_short_hidden = self.m_short_gru(action_short_embedded, action_short_hidden)

        action_mask_short_batch = action_mask_short_batch.unsqueeze(-1).float()
        action_short_output_mask = action_short_output*action_mask_short_batch

        # pad_seqLen_batch = [i-1 if i > 0 else 0 for i in seqLen_batch]
        first_dim_index = torch.arange(short_batch_size).to(self.device)
        second_dim_index = torch.from_numpy(actionNum_short_batch).to(self.device)

        ### batch_size*hidden_size
        seq_short_input = action_short_output_mask[first_dim_index, second_dim_index, :]
        # output_seq, hidden_seq = self.m_short_gru(input_seq, hidden_seq)

        return seq_cate_input, seq_short_input

    def init_hidden(self, batch_size, hidden_size):
        '''
        Initialize the hidden state of the GRU
        '''
        h0 = torch.zeros(self.num_layers, batch_size, hidden_size).to(self.device)
        return h0

class CATENN(nn.Module):
    def __init__(self, cate_input_size, cate_output_size, cate_embedding_dim, cate_hidden_size, num_layers=1, final_act='tanh', dropout_hidden=.8, dropout_input=0, use_cuda=False, cate_shared_embedding=True):
        super(CATENN, self).__init__()
        self.num_layers = num_layers
        self.dropout_hidden = dropout_hidden

        self.m_cate_input_size = cate_input_size
        self.m_cate_embedding_dim = cate_embedding_dim
        self.m_cate_hidden_size = cate_hidden_size
        self.m_cate_output_size = cate_output_size

        self.m_cate_embedding = nn.Embedding(self.m_cate_input_size, self.m_cate_embedding_dim)
        self.m_cate_gru = nn.GRU(self.m_cate_embedding_dim, self.m_cate_hidden_size, self.num_layers, dropout=self.dropout_hidden, batch_first=True)
        self.m_cate_h2o = nn.Linear(self.m_cate_hidden_size, self.m_cate_output_size)
        self.device = torch.device('cuda' if use_cuda else 'cpu')
        if cate_shared_embedding:
            self.m_cate_h2o.weight = self.m_cate_embedding.weight

    def forward(self, cate_short_batch, action_mask_short_batch, actionNum_short_batch, train_test_flag):
        cate_short_embedded = self.m_cate_embedding(cate_short_batch)
        short_batch_size = cate_short_embedded.size(0)
        cate_short_hidden = self.init_hidden(short_batch_size, self.m_cate_hidden_size)
        cate_short_output, cate_short_hidden = self.m_cate_gru(cate_short_embedded, cate_short_hidden)

        cate_short_output_mask = cate_short_output*action_mask_short_batch

        first_dim_index = torch.arange(short_batch_size).to(self.device)
        second_dim_index = torch.from_numpy(actionNum_short_batch).to(self.device)
 
        seq_cate_short_input = cate_short_output_mask[first_dim_index, second_dim_index, :]
        
        ### logit_cate_mask: category prediction
        logit_cate_short = self.m_cate_h2o(seq_cate_short_input)
        # cate_prob_mask = F.softmax(cate_logit_mask, dim=-1)

        return logit_cate_short
    
    def init_hidden(self, batch_size, hidden_size):
        '''
        Initialize the hidden state of the GRU
        '''
        h0 = torch.zeros(self.num_layers, batch_size, hidden_size).to(self.device)
        return h0
