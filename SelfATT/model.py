from torch import nn
from transformer import TransformerEncoderLayer, TransformerEncoder
from attention import MultiheadAttention
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import math, copy

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

# self-attention model with mask
class SATT(nn.Module):
    def __init__(self, log, ss, input_size, hidden_size, output_size, num_layers=1, num_heads=1, use_cuda=True, batch_size=50, dropout_input=0, dropout_hidden=0.5, embedding_dim=-1, position_embedding=False, shared_embedding=True, window_size=8, kernel_type='exp-1', contextualize_opt=None):
        super().__init__()
        
        self.m_log = log
        self.m_ss = ss

        self.device = torch.device('cuda' if use_cuda else 'cpu')
        
        self.embed = nn.Embedding(input_size, hidden_size, padding_idx=0).to(self.device)
        self.pe = PositionalEncoding(hidden_size, dropout_input,  max_len=window_size)

        if shared_embedding:
            self.out_matrix = self.embed.weight.to(self.device)
        else:
            self.out_matrix = nn.Parameter(torch.rand(output_size, hidden_size, requires_grad=True, device=self.device))
   
        encoder_layer = TransformerEncoderLayer(hidden_size, num_heads, dim_feedforward=2048, dropout=dropout_hidden)
        norm = nn.LayerNorm(hidden_size)
        self.encode = TransformerEncoder(encoder_layer, num_layers, norm=norm).to(self.device)
        
        self.decoder = MultiheadAttention(hidden_size, num_heads, dropout=dropout_hidden)

        if shared_embedding:
            message = "share embedding"
            self.m_log.addOutput2IO(message)

            self.m_ss.params.weight = self.embed.weight

        self = self.to(self.device)

    def forward(self, src, debug=False):
        x_embed = self.embed(src)
        src_mask = (src == 0)
        src_mask_neg = (src != 0)
        
        x = x_embed.transpose(0,1)
      
        if self.pe != None:
            x = self.pe(x)  

        if debug:
            x, alpha = self.encode(x, src_key_padding_mask=src_mask, debug=True )
        else:
            x = self.encode(x, src_key_padding_mask=src_mask )
            
        d_output = x[-1,:,:] ### last hidden state 
        
        return d_output
        # output = F.linear(d_output.squeeze(0), self.out_matrix)
        
        # if debug:      
        #     return output, alpha

        # return output
