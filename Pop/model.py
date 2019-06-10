from torch import nn
import torch

class SEQPOP():
    def __init__(self, input_size):
        print("sequential popularity")

    def train(self, input):
        return

    def test(self, input):
        return


class GLOBALPOP():
    def __init__(self, input_size, use_cuda):
        print("global popularity")

        self.device = torch.device('cuda' if use_cuda else 'cpu')
        self.m_vocab_size = input_size
        self.m_item_freq = torch.FloatTensor(input_size).to(self.device)
        self.m_item_freq.zero_()

    def train(self, input):
        
        add_on = input.cpu().float().histc(self.m_vocab_size, min=0, max=self.m_vocab_size).to(self.device)
        # print("inptu", input)
        # print(add_on)
        self.m_item_freq += add_on

    def test(self, input):
        return self.m_item_freq
