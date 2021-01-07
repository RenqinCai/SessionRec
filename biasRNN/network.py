from torch import nn
import torch
import torch.nn.functional as F
import datetime
import numpy as np

class NETWORK(nn.Module):
    def __init__(self, input_size, ss, args, device):
        super(NETWORK, self).__init__()
        self.m_input_size = input_size
        self.m_hidden_size = args.hidden_size

        self.m_num_layers = args.num_layers
        self.m_dropout_hidden_rate = args.dropout_hidden
        self.m_dropout_input_rate = args.dropout_input
        self.m_embedding_dim = args.embedding_dim
    
        self.m_device = device
       
        self.m_embed_dropout = nn.Dropout(p=self.m_dropout_input_rate)

        self.m_item_embed = nn.Embedding(self.m_input_size, self.m_embedding_dim)
        self.m_short_gru = nn.GRU(self.m_embedding_dim, self.m_hidden_size, self.m_num_layers, dropout=self.m_dropout_hidden_rate, batch_first=True)

        self.m_ss = ss

        if args.shared_embedding:
            self.m_ss.params.weight = self.m_item_embed.weight

        if self.m_embedding_dim != self.m_hidden_size:
            self.m_fc = nn.Linear(self.m_hidden_size, self.m_embedding_dim)
            self.m_fc_relu = nn.ReLU()
            
        self = self.to(self.m_device)

    def forward(self, action_short_batch, action_mask_short_batch, actionNum_short_batch):
        action_short_input = action_short_batch.long()
        action_short_embedded = self.m_item_embed(action_short_input)
        action_short_embedded = self.m_embed_dropout(action_short_embedded)
        
        short_batch_size = action_short_embedded.size(0) 
        
        action_short_hidden = self.init_hidden(short_batch_size)
        action_short_output, action_short_hidden = self.m_short_gru(action_short_embedded, action_short_hidden)

        action_mask_short_batch = action_mask_short_batch.unsqueeze(-1).float()
        action_short_output_mask = action_short_output*action_mask_short_batch

        first_dim_index = torch.arange(short_batch_size).to(self.m_device)
        second_dim_index = torch.from_numpy(actionNum_short_batch).to(self.m_device)

        ### batch_size*hidden_size
        seq_short_input = action_short_output_mask[first_dim_index, second_dim_index, :]

        last_output = seq_short_input
        if self.m_embedding_dim != self.m_hidden_size:
            last_output = self.m_fc(seq_short_input)
            last_output = self.m_fc_relu(last_output)

        return last_output

    def init_hidden(self, batch_size):
        '''
        Initialize the hidden state of the GRU
        '''
        # print(self.num_layers, batch_size, hidden_size)
        h0 = torch.zeros(self.m_num_layers, batch_size, self.m_hidden_size).to(self.m_device)

        return h0