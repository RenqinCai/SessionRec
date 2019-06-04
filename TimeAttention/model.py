from torch import nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import math, copy

class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len = 80):
        super().__init__()
        self.d_model = d_model
        
        # create constant 'pe' matrix with values dependant on pos and i
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))
                
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # make embeddings relatively larger
        x = x * math.sqrt(self.d_model)
        #add constant to embedding
        seq_len = x.size(1)
        x = x + Variable(self.pe[:,:seq_len], requires_grad=False)
        return x

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=128, dropout = 0.5):
        super().__init__() 

        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)
    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x

class Transformer(nn.Module):
    def __init__(self, d_model, heads, dropout = 0.5):
        super().__init__()
        
        self.attn = nn.MultiheadAttention(d_model, heads, dropout=dropout)
        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model)
        
    def forward(self, q, k, v, mask):
        attn_output, attn_output_weights = self.attn(q, k, v, key_padding_mask=mask)
        q = q + attn_output
        q = self.norm_1(q)
        q = q + self.ff(q)
        q = self.norm_2(q)  
        return q


# temporal-attention model with mask
class TATT(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, num_heads=1, use_cuda=True, batch_size=50, dropout_input=0, dropout_hidden=0.5, embedding_dim=-1, position_embedding=False, shared_embeding=True):
        super().__init__()
        
        self.device = torch.device('cuda' if use_cuda else 'cpu')
        
        self.embed = nn.Embedding(input_size, hidden_size, padding_idx=0).to(self.device)
        self.pe = PositionalEncoder(hidden_size, max_seq_len=30) if position_embedding else None
        self.encode_layers = nn.ModuleList([Transformer(hidden_size, num_heads, dropout=dropout_hidden) for i in range(num_layers)])
#         self.decode = Transformer(hidden_size, num_heads, dropout=dropout_hidden)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout_hidden)
        if shared_embeding:
            self.out_matrix = self.embed.weight.to(self.device)
        else:
            self.out_matrix = torch.rand(hidden_size, output_size, requires_grad=True).to(self.device)
        
        self.beta = torch.tensor(1.0, requires_grad=True).to(self.device)
        self.gamma = torch.tensor(1.0, requires_grad=True).to(self.device)
        self = self.to(self.device)

    def forward(self, src, t, src_len):
        src_mask = (src == 0)
        x = self.embed(src)    
        if self.pe != None:
            x = self.pe(x)

        x = x.transpose(0,1)
        for i, layer in enumerate(self.encode_layers):
            x = layer(x, x, x, src_mask) ### encoded input sequence
        
        trg = self.embed(src[:, -1]).unsqueeze(0)  ### last input
        
        t = self.decay(t)
#         d_output = self.decode(trg, x, x, src_mask)
        _, weight = self.attn(trg, x, x, src_mask)
    
        alpha = F.softmax(weight.squeeze(1) * t * src_mask.float(), dim=1 ).unsqueeze(2)
        d_output = torch.sum(alpha * x.transpose(0,1), 1) 
        output = F.linear(d_output.squeeze(0), self.out_matrix)
        
        return output
    
    def decay(self, t):
        return self.beta * torch.log(t + 1.0) + self.gamma

    
    
    
    