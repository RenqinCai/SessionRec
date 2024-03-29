from torch import nn
import torch

class HierGRU4REC(nn.Module):
    def __init__(self, input_size, output_size, sess_hidden_size, user_hidden_size, user_output_size, sess_num_layers=1, user_num_layers=1, final_act='tanh', dropout_hidden=0.5, dropout_input=0, batch_size=50, embedding_dim=-1, use_cuda=False):
        super(HierGRU4REC, self).__init__()
        self.m_input_size = input_size
        self.m_output_size = output_size
        self.m_sess_hidden_size = sess_hidden_size
        # self.m_sess_output_size = sess_output_size

        self.m_user_hidden_size = user_hidden_size
        self.m_user_output_size = user_output_size

        self.m_sess_num_layers = sess_num_layers
        self.m_user_num_layers = user_num_layers

        self.dropout_hidden = dropout_hidden
        self.dropout_input = dropout_input
        self.m_embedding_dim = embedding_dim
        self.m_batch_size = batch_size
        self.m_use_cuda = use_cuda
        self.device = torch.device('cuda' if use_cuda else 'cpu')

        self.m_h2o = nn.Linear(self.m_sess_hidden_size, self.m_output_size)
        self.m_final_act = nn.Tanh()
        self.create_final_activation(final_act)  

        if self.m_embedding_dim != -1:
            self.m_look_up = nn.Embedding(self.m_input_size,                                self.m_embedding_dim)
            self.m_sess_gru = nn.GRU(self.m_embedding_dim,                                  self.m_sess_hidden_size, self.m_sess_num_layers,                         dropout=self.dropout_hidden)
            self.m_user_gru = nn.GRU(self.m_sess_hidden_size,                                   self.m_user_hidden_size, self.m_user_num_layers,dropout=self.dropout_hidden)
        else:
            print("error embedding dim is -1")

        self = self.to(self.device)

    def create_final_activation(self, final_act):
        if final_act == "tanh":
            self.m_final_act =  nn.Tanh()
        elif final_act == "relu":
            self.m_final_act = nn.ReLU()
        elif final_act == "softmax":
            self.m_final_act = nn.Softmax()
        elif final_act == "softmax_logit":
            self.m_final_act = nn.LogSoftmax()
    
    def forward(self, input_x, sess_hidden, user_hidden, mask_sess, mask_user):
        # embedded = input
        # print(input_x)
        embedded = self.m_look_up(input_x)
        
        # embedded = embedded.transpose(0, 1)
        embedded = embedded.unsqueeze(0)

        ### update user gru
        ## initialize these two
        next_user_hidden = user_hidden
        next_sess_hidden = sess_hidden
        
        torch.autograd.set_detect_anomaly(True)

        if len(mask_user) != 0:
            # print("mask user 1")
            user_hidden[:, mask_user, :] = 0
            sess_hidden[:, mask_user, :] = 0

        next_sess_output, next_sess_hidden = self.m_sess_gru(embedded, sess_hidden)

        if len(mask_sess) != 0:
            mask_sess_hidden_user = next_sess_hidden[:, mask_sess, :]
            mask_user_hidden_user = user_hidden[:, mask_sess, :]

            mask_user_output, mask_next_user_hidden = self.m_user_gru(mask_sess_hidden_user, mask_user_hidden_user)

            next_user_hidden[:, mask_sess, :] = mask_next_user_hidden

            next_sess_hidden[:, mask_sess, :] = mask_next_user_hidden
            
        sess_output = next_sess_output.view(-1, next_sess_output.size(-1))
        logit = self.m_final_act(self.m_h2o(sess_output))
     
        return logit, next_sess_hidden, next_user_hidden

    def init_hidden(self):
        
        h0 = torch.zeros(self.m_sess_num_layers, self.m_batch_size, self.m_sess_hidden_size).to(self.device)
        
        return h0

class USERGRU4REC(nn.Module):
    def __init__(self, input_hidden_size, output_hidden_size, num_layers=1, dropout_hidden=0.5, dropout_input=0, batch_size=50, use_cuda=False):
        super(USERGRU4REC, self).__init__
        self.m_input_hidden_size = input_hidden_size
        self.m_output_hidden_size = output_hidden_size
        self.m_num_layers = num_layers
        
        self.m_dropout_hidden = dropout_hidden
        self.m_dropout_input = dropout_input
        self.m_batch_size = batch_size
        self.m_use_cuda = use_cuda

        self.device = torch.device('cuda' if use_cuda else 'cpu')

        self.m_gru = nn.GRU(self.m_input_hidden_size, self.m_output_hidden_size, self.m_num_layers, dropout=self.m_dropout_hidden)

        self = self.to(self.device)

    ## do not forget to add whether we need to update hidden or not for a new user
    def forward(self, input_x, hidden):

        input_x = torch.mean(input_x)

        output, hidden = self.m_gru(input_x, hidden)

        return hidden
    
    def init_hidden(self):
        h0 = torch.zeros(self.m_num_layers, self.m_batch_size, self.m_input_hidden_size).to(self.device)
        
        return h0
