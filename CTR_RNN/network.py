from torch import nn
import torch
import torch.nn.functional as F
import datetime
from transformer import TransformerEncoderLayer, TransformerEncoder
from attention import MultiheadAttention
import numpy as np

class CTR_RNN(nn.Module):
    def __init__(self, log, ss, input_size, hidden_size, output_size, decay_size, num_heads, num_layers=1, dropout_hidden=.8, dropout_input=0, embedding_dim=-1, use_cuda=True, shared_embedding=True):
        super(CTR_RNN, self).__init__()
        self.m_input_size = input_size
        self.m_hidden_size = hidden_size
        self.m_output_size = output_size
        self.m_num_layers = num_layers
        self.m_dropout_hidden = dropout_hidden
        self.m_dropout_input = dropout_input
        self.m_embedding_dim = embedding_dim
        self.m_num_heads = num_heads
        self.m_decay_size = decay_size
    
        self.m_use_cuda = use_cuda
        self.m_device = torch.device('cuda' if use_cuda else 'cpu')
        print("self device", self.m_device)
        self.m_log = log
        
        self.m_input_embed = nn.Embedding(self.m_input_size, self.m_embedding_dim)

        # self.m_self_attn_module = SELFATTN(self.m_hidden_size, self.m_num_heads, num_layers=self.m_num_layers, dropout_hidden=self.m_dropout_hidden)

        self.m_self_RNN_module = SELFRNN(self.m_num_layers, self.m_hidden_size, self.m_dropout_hidden, self.m_device)

        self.m_friend_module = FRIEND(self.m_num_layers, self.m_hidden_size, self.m_dropout_hidden, self.m_device)

        # self.m_friend_RNN_module = FRIENDRNN(self.m_num_layers, self.m_hidden_size, self.m_dropout_hidden, self.m_device)

        # self.m_decay_rate_module = FRIENDDECAYRATE(self.m_hidden_size, self.m_decay_size)

        # self.m_friendship_module = FRIENDSHIP(self.m_hidden_size, self.m_hidden_size, self.m_hidden_size)

        # self.m_friend_attn_module = FRIENDATTN(self.m_hidden_size, self.m_hidden_size)

        self.m_ss = ss

        if shared_embedding:
            message = "share embedding"
            self.m_log.addOutput2IO(message)

            self.m_ss.params.weight = self.m_input_embed.weight

        if self.m_embedding_dim != self.m_hidden_size:
            self.m_fc = nn.Linear(self.m_hidden_size, self.m_embedding_dim)
            self.m_fc_relu = nn.ReLU()

        self.m_output_fc = nn.Linear(self.m_hidden_size*2, self.m_hidden_size)
        self.m_output_relu = nn.ReLU()
            
        self = self.to(self.m_device)

    def forward(self, self_src, common_src, common_time, friend_src, friend_num_src_tensor):
        ### self_src: batch_size*seq_len
        ### common_src: (batch_size*friend_num)*common_num
        ### common_time: (batch_size*friend_num)*common_num
        ### friend_src: (batch_size*friend_num)*seq_len
        ### friend_num_src: batch_size*1

        st = datetime.datetime.now()

        batch_size = common_src.size()[0]
        # print("batch_size", batch_size)

        self_src_mask = (self_src != 0)
        common_src_mask = (common_src != 0)
        common_time_mask = (common_time != 0)
        friend_src_mask = (friend_src != 0)

        self_x = self.m_input_embed(self_src)
        common_x = self.m_input_embed(common_src)
        friend_x = self.m_input_embed(friend_src)

        if self.m_embedding_dim != self.m_hidden_size:
            self_x = self.m_fc_relu(self.m_fc(self_x))
            common_x = self.m_fc_relu(self.m_fc(common_x))
            friend_x = self.m_fc_relu(self.m_fc(friend_x))
        
        ### self_src_mask size: batch_size*seq_len
        ### first obtain the self x
        ### input self_x size: batch_size*seq_len*hidden_size
        ### output self_x size: batch_size*hidden_size
        self_x = self.m_self_RNN_module(self_src_mask, self_x)

        friend_num_src = friend_num_src_tensor.cpu().tolist()
        friend_x = self.m_friend_module(self_x, common_x, common_time, common_src_mask, friend_x, friend_src_mask, friend_num_src, friend_num_src_tensor)
        # print("self x size", self_x.size())

        ### obtain the decay rate
        ### common_x size: (batch_size*friend_num)*common_num*hidden_size
        ### friend_decay_rate size: batch_size*friend_num
        # friend_decay_rate = self.m_decay_rate_module(common_x, friend_num_src, common_src_mask)
        # print("friend decay rate size", friend_decay_rate.size())

        ### obtain the friendship
        ### common_x size: (batch_size*friend_num)*common_num*hidden_size
        ### friendship: batch_size*friend_num

        ### friend_src_mask size: (batch_size*friend_num)*seq_len
        ### friend_x size: (batch_size*friend_num)*hidden_size
        # friend_x = self.m_friend_RNN_module(friend_src_mask, friend_x)
        # print("1 friend_x", friend_x)

        ### friendship size: batch_size*friend_num
        # friendship = self.m_friendship_module(self_x, friend_x, common_x, common_time, friend_num_src, friend_num_src_tensor, common_src_mask)
        
        ### friend_x size: batch_size*friend_num*hidden_size
        # friend_x = torch.split(friend_x, split_size_or_sections=friend_num_src, dim=0)
        # friend_x = torch.nn.utils.rnn.pad_sequence(friend_x, batch_first=True)

        ### decay_friendship
        ### decay_friendship size: batch_size*friend_num
        # decay_friendship = friendship*friend_decay_rate
        # decay_friendship = friendship

        ### normalized_decay_friendship size: batch_size*friend_num
        # print("decay_friendship", decay_friendship)
        # normalized_decay_friendship = F.softmax(decay_friendship, dim=-1)

        ### obtain the friend attn
        ### friend_diff_x size: (batch_size*friend_num)*diff_num*hidden_size
        ### friend_x size: batch_size*friend_num*hidden_size
        # friend_x = self.m_friend_attn_module(friend_diff_x, self_x, friend_num_src, friend_num_src_tensor, friend_diff_src_mask)

        ### decay_friendship weighted friend x
        ### normalized_decay_friendship: batch_size*friend_num*1
        ### decay_friend_x size: batch_size*friend_num*hidden_size
        # normalized_decay_friendship = normalized_decay_friendship.unsqueeze(-1)
        # decay_friend_x = normalized_decay_friendship*friend_x

        # print("normalized_decay_friendship", normalized_decay_friendship)

        ### decay_friend_x size: batch_size*1*hidden_size
        # decay_friend_x = torch.sum(decay_friend_x, dim=1)
        # decay_friend_x = decay_friend_x.squeeze(1)

        # print("decay_friend_x", decay_friend_x)
        ### self_x size: batch_size*hidden_size
        ### weighted_friend_x: batch_size*hidden_size
        ### output size: 
        output = torch.cat([self_x, friend_x], dim=-1)
        # output = self_x

        if output.size()[-1] != self.m_hidden_size:
            output = self.m_output_relu(self.m_output_fc(output))

        et = datetime.datetime.now()
        duration = et-st
        print("network duration", duration)

        return output

    def embedding_dropout(self, input):
        p_drop = torch.Tensor(input.size(0), input.size(1), 1).fill_(1 - self.m_dropout_input)  # (B,1)
        mask = torch.bernoulli(p_drop).expand_as(input) / (1 - self.m_dropout_input)  # (B,C)
        mask = mask.to(self.m_device)
        input = input * mask  # (B,C)

        return input

    def sample_loss(self, output, y, sample_ids, true_freq, sample_freq, acc_hits, remove_match, sample_full_flag):
        if sample_full_flag == "sample":
            sampled_logit_batch, sampled_target_batch = self.m_ss(output, y, sample_ids, true_freq, sample_freq, acc_hits, self.m_device, remove_match, sample_full_flag)
            return sampled_logit_batch, sampled_target_batch
        
        if sample_full_flag == "full":
            sampled_logit_batch, sampled_target_batch = self.m_ss(output, y, sample_ids, true_freq, sample_freq, acc_hits, self.m_device, remove_match, sample_full_flag)
            
            return sampled_logit_batch, sampled_target_batch

class FRIEND(nn.Module):
    def __init__(self, num_layers, hidden_size, dropout, device):
        super(FRIEND, self).__init__()
        self.m_friend_RNN_module = FRIENDRNN(num_layers, hidden_size, dropout, device)

        self.m_friendship_module = FRIENDSHIP(hidden_size, hidden_size, hidden_size)

    def forward(self, self_x, common_x, common_time, common_src_mask, friend_x, friend_src_mask, friend_num_src, friend_num_src_tensor):
        ### input friend_x size: (batch_size*friend_num)*seq_len*hidden_size
        ### output friend_x size: (batch_size*friend_num)*hidden_size
        friend_x = self.m_friend_RNN_module(friend_src_mask, friend_x)
       
        friendship = self.m_friendship_module(self_x, common_x, common_time, common_src_mask, friend_x, friend_num_src, friend_num_src_tensor)
        decay_friendship = friendship

        friend_x = torch.split(friend_x, split_size_or_sections=friend_num_src, dim=0)
        friend_x = torch.nn.utils.rnn.pad_sequence(friend_x, batch_first=True)

        ### normalized_decay_friendship: batch_size*friend_num*1
        normalized_decay_friendship = F.softmax(decay_friendship, dim=-1)
        normalized_decay_friendship = normalized_decay_friendship.unsqueeze(-1)

        ### friend_x size: batch_size*friend_num*hidden_size
        decay_friend_x = normalized_decay_friendship*friend_x
        
        decay_friend_x = torch.sum(decay_friend_x, dim=1)
        decay_friend_x = decay_friend_x.squeeze(1)
    
        return decay_friend_x

class SELFRNN(nn.Module):
    def __init__(self, num_layers, hidden_size, dropout, device):
        super(SELFRNN, self).__init__()
        self.m_num_layers = num_layers
        # self.m_batch_size = batch_size
        self.m_hidden_size = hidden_size
        self.m_dropout = dropout
        self.m_device = device

        self.m_gru = nn.GRU(self.m_hidden_size, self.m_hidden_size, self.m_num_layers, dropout=self.m_dropout, batch_first=True)
    
    def forward(self, src_mask, embed_x, debug=False):
        ### embed_x size: batch_size*seq_len*hidden_size
        batch_size = embed_x.size()[0]

        init_hidden = self.f_init_hidden(batch_size)

        ### embed_x size: batch_size*seq_len*hidden_size
        self_x, self_hidden = self.m_gru(embed_x, init_hidden)

        mask_self_x = self_x*(src_mask.unsqueeze(-1).float())

        first_dim_index = torch.arange(batch_size).to(self.m_device)

        second_dim_index = src_mask.sum(dim=1, keepdim=False).float()-1

        second_dim_index = second_dim_index.long()
        # print("first dim size, second dim size", first_dim_index.size(), second_dim_index.size())

        self_x = mask_self_x[first_dim_index, second_dim_index, :]

        # self_x = self_x.squeeze(1)

        # print("self x size", self_x.size())

        return self_x

    def f_init_hidden(self, batch_size):
        '''
        Initialize the hidden state of the GRU
        '''
        # print(self.num_layers, batch_size, hidden_size)
        h0 = torch.zeros(self.m_num_layers, batch_size, self.m_hidden_size).to(self.m_device)

        return h0

class FRIENDRNN(nn.Module):
    def __init__(self, num_layers, hidden_size, dropout, device):
        super(FRIENDRNN, self).__init__()
        self.m_num_layers = num_layers
        # self.m_batch_size = batch_size
        self.m_hidden_size = hidden_size
        self.m_dropout = dropout
        self.m_device = device

        self.m_gru = nn.GRU(self.m_hidden_size, self.m_hidden_size, self.m_num_layers, dropout=self.m_dropout, batch_first=True)
    
    def forward(self, src_mask, embed_x, debug=False):
        ### embed_x size: batch_size*seq_len*hidden_size
        batch_size = embed_x.size()[0]

        init_hidden = self.f_init_hidden(batch_size)

        ### embed_x size: batch_size*seq_len*hidden_size
        self_x, self_hidden = self.m_gru(embed_x, init_hidden)

        mask_self_x = self_x*(src_mask.unsqueeze(-1).float())

        first_dim_index = torch.arange(batch_size).to(self.m_device)

        second_dim_index = src_mask.sum(dim=1, keepdim=False).float()-1

        second_dim_index = second_dim_index.long()
        # print("first dim size, second dim size", first_dim_index.size(), second_dim_index.size())

        self_x = mask_self_x[first_dim_index, second_dim_index, :]

        # self_x = self_x.squeeze(1)

        # print("self x size", self_x.size())

        return self_x

    def f_init_hidden(self, batch_size):
        '''
        Initialize the hidden state of the GRU
        '''
        # print(self.num_layers, batch_size, hidden_size)
        h0 = torch.zeros(self.m_num_layers, batch_size, self.m_hidden_size).to(self.m_device)

        return h0

class FRIENDSHIP(nn.Module):
    def __init__(self, self_input_size, self_friend_input_size, friendship_size):
        super(FRIENDSHIP, self).__init__()
        self.m_linear_friend = nn.Linear(self_friend_input_size*2, friendship_size, bias=False)
        self.m_linear_beta = nn.Linear(self_input_size, friendship_size, bias=False)
        self.m_tau = 1e6
        self.m_bias = 1.0

    def forward(self, self_x, common_x, common_time, common_src_mask, friend_x, friend_num_src, friend_num_src_tensor):
        ### common_x size: (batch_size*friend_num)*common_num*hidden_size
        ### self_x size: batch_size*hidden_size
        ### friend_x size: (batch_size*friend_num)*hidden_size
        ### self_friend_x size: (batch_size*friend_num)*(hidden_size*2)
        ### common_src_mask size: (batch_size*friend_num)*common_num
        ### self_repeat_x size: (batch_size*friend_num)*hidden_size
        self_repeat_x = torch.repeat_interleave(self_x, repeats=friend_num_src_tensor, dim=0)
        self_friend_x = torch.cat([self_repeat_x, friend_x], dim=-1)

        self_friend_x_size = self_friend_x.size()

        common_x_size = common_x.size()

        if self_friend_x_size[-1] != common_x_size[-1]:
            proj_self_friend_x = self.m_linear_friend(self_friend_x)
            self_friend_x = proj_self_friend_x

        ### self_friend_x size: (batch_size*friend_num)*hidden_size*1
        self_friend_x = self_friend_x.unsqueeze(-1)

        ### proj_common_x size: (batch_size*friend_num)*common_num*hidden_size
        proj_common_x = self.m_linear_beta(common_x)
        ### content_weight size: (batch_size*friend_num)*common_num*1
        content_friendship = torch.bmm(proj_common_x, self_friend_x)

        ### content_weight size: (batch_size*friend_num)*common_num
        content_friendship = content_friendship.squeeze(-1)

        # print("before softplus content_friendship", content_friendship)
        content_friendship = F.softplus(content_friendship)

        ### common_time size: (batch_size*friend_num)*common_num
        ### temporal_weight size: (batch_size*friend_num)*common_num
        temporal_weight = torch.exp(-common_time/self.m_tau+self.m_bias)

        # print("temporal weight", temporal_weight)
        # print("after softplus content_friendship", content_friendship)

        content_temporal_weight = content_friendship*temporal_weight

        mask_content_temporal_weight = content_temporal_weight*(common_src_mask.float())
        
        ### weight_sum size: (batch_size*friend_num)
        temporal_friendship = torch.sum(mask_content_temporal_weight, dim=-1)

        friend_temporal_friendship = torch.split(temporal_friendship, split_size_or_sections=friend_num_src, dim=0)

        ### friend_temporal_friendship size: batch_size*friend_num
        friend_temporal_friendship = torch.nn.utils.rnn.pad_sequence(friend_temporal_friendship, batch_first=True)

        return friend_temporal_friendship

class FRIENDDECAYRATE(nn.Module):
    def __init__(self, input_size, decay_size):
        super(FRIENDDECAYRATE, self).__init__()
        self.m_linear = nn.Linear(input_size, decay_size, bias=False)
        self.m_weight = nn.Linear(decay_size, 1)

    def forward(self, common_x, friend_num_src, common_src_mask):
        ### common_x size: (batch_size*friend_num)*common_num*hidden_size
        ### friend_num_src size: batch_size
        ### decay_common_x size: (batch_size*friend_num)*common_num*decay_size
        decay_common_x = self.m_linear(common_x)

        decay_common_x = decay_common_x*common_src_mask.float().unsqueeze(-1)
        ### decay_common_x size: (batch_size*friend_num)*common_num*decay_size
        ### avg_decay_common_x size: (batch_size*friend_num)*1*decay_size
        avg_decay_common_x = decay_common_x.mean(dim=1)

        ### avg_decay_common_x size: (batch_size*friend_num)*decay_size
        avg_decay_common_x = avg_decay_common_x.squeeze(1)

        friend_avg_decay_common_x = torch.split(avg_decay_common_x, split_size_or_sections=friend_num_src, dim=0)

        ### friend_avg_decay_common_x size: batch_size*friend_num*decay_size
        friend_avg_decay_common_x = torch.nn.utils.rnn.pad_sequence(friend_avg_decay_common_x, batch_first=True)

        # print("friend_avg_decay_common_x", friend_avg_decay_common_x.size())

        ### decay_rate size: batch_size*friend_num*1
        decay_rate = self.m_weight(friend_avg_decay_common_x)

        # print("decay_rate 1", decay_rate.size())
        decay_rate = decay_rate.squeeze(-1)

        ### forget multiplying mask
        ### decay_rate size: batch_size*friend_num
        decay_rate = F.softplus(decay_rate)

        # print("decay_rate 2", decay_rate.size())
        return decay_rate

class FRIENDATTN(nn.Module):
    def __init__(self, friend_input_size, diff_size):
        super(FRIENDATTN, self).__init__()
        self.m_linear_diff = nn.Linear(friend_input_size, diff_size, bias=False)

    def forward(self, friend_diff_x, self_x, friend_num_src, friend_num_src_tensor, friend_diff_src_mask):
        ### friend_diff_x: (batch_size*friend_num)*diff_num*hidden_size
        ### self_x size: batch_size*hidden_size
        ### friend_diff_src_mask size: (batch_size*friend_num)*diff_num
        
        ### broadcast self_x into (batch_size*friend_num)*hidden_size*1
        self_x = torch.repeat_interleave(self_x, repeats=friend_num_src_tensor, dim=0)
        self_x = self_x.unsqueeze(-1)

        ### content_weight size: (batch_size*friend_num)*diff_num*1
        content_weight = torch.bmm(friend_diff_x, self_x)

        ### content_weight size: (batch_size*friend_num)*diff_num
        content_weight = content_weight.squeeze(-1)
        
        ### content_weight size: (batch_size*friend_num)*diff_num
        ### weight size: (batch_size*friend_num)*diff_num
        weight = F.softmax(content_weight, dim=-1)

        ### weight size: (batch_size*friend_num)*diff_num*1
        weight = weight.unsqueeze(-1)

        ### weighted_friend_diff_x size: (batch_size*friend_num)*diff_num*hidden_size
        weighted_friend_diff_x = weight*friend_diff_x

        mask_weighted_friend_diff_x = weighted_friend_diff_x*friend_diff_src_mask.float().unsqueeze(-1)

        ### weighted_friend_x size: (batch_size*friend_num)*1*hidden_size
        weighted_friend_x = torch.sum(mask_weighted_friend_diff_x, dim=1)

        ### weighted_friend_x size: (batch_size*friend_num)*hidden_size
        weighted_friend_x = weighted_friend_x.squeeze(1)

        friend_weighted_friend_x = torch.split(weighted_friend_x, split_size_or_sections=friend_num_src, dim=0)

        ### friend_temporal_friendship size: batch_size*friend_num*hidden_size
        friend_weighted_friend_x = torch.nn.utils.rnn.pad_sequence(friend_weighted_friend_x, batch_first=True)

        return friend_weighted_friend_x

class SELFATTN(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout_hidden, dim_feedforward=2048, num_layers=1):
        ### dim_feedforward is unspecified
        super(SELFATTN, self).__init__()
        encoder_layer = TransformerEncoderLayer(hidden_size, num_heads, dim_feedforward=dim_feedforward, dropout=dropout_hidden)
        norm = nn.LayerNorm(hidden_size)
        self.encode = TransformerEncoder(encoder_layer, num_layers, norm=norm)
        
    def forward(self, src_mask, embed_x, debug=False):
        ### embed_x size: batch_size*seq_len*hidden_size
        x = embed_x.transpose(0,1)
      
        # if self.pe != None:
        #     x = self.pe(x)  

        if debug:
            x, alpha = self.encode(x, src_key_padding_mask=src_mask, debug=True )
        else:
            x = self.encode(x, src_key_padding_mask=src_mask)

        self_x = x[-1, :, :]

        return self_x