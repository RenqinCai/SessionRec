# import lib
from evaluation import *
import time
import torch
import numpy as np
import os
from dataset import *

class Trainer(object):
    def __init__(self, model, train_data, eval_data, use_cuda, topk, args):
        self.model = model
        self.train_data = train_data
        self.eval_data = eval_data
        self.topk = topk
        self.evaluation = Evaluation(self.model, use_cuda, self.topk)
        self.device = torch.device('cuda' if use_cuda else 'cpu')
        self.args = args

    def train(self, start_epoch, end_epoch, batch_size, start_time=None):
        if start_time is None:
            self.start_time = time.time()
        else:
            self.start_time = start_time

        for epoch in range(start_epoch, end_epoch + 1):
            print("*"*10, epoch, "*"*5)
            st = time.time()
            self.train_epoch(epoch, batch_size)
            recall, mrr = self.evaluation.eval(self.train_data, batch_size)
            print("*"*10, "train Epoch: {}, recall: {:.4f}, mrr: {:.4f}, time: {}".format(epoch, recall, mrr, time.time() - st))

            recall, mrr = self.evaluation.eval(self.eval_data, batch_size)
            print("-"*10,"Epoch: {}, recall: {:.4f}, mrr: {:.4f}, time: {}".format(epoch, recall, mrr, time.time() - st))
            # checkpoint = {
            #     'model': self.model.state_dict(),
            #     'args': self.args,
            #     'epoch': epoch,
            #     'optim': self.optim,
            #     'recall': recall,
            #     'mrr': mrr
            # }
            # model_name = os.path.join(self.args.checkpoint_dir, "model_{0:05d}.pt".format(epoch))
            # torch.save(checkpoint, model_name)
            # print("Save model as %s" % model_name)

    def train_epoch(self, epoch, batch_size):
        
        dataloader = self.train_data
        for idx_input, input, target, mask in dataloader:
            input = input.to(self.device)
            target = target.to(self.device)
           
            # print("input size", input.size())
            self.model.train(input)
           