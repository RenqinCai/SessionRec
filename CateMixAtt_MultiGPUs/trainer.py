# import lib
from evaluation import *
import time
import torch
import numpy as np
import os
from dataset import *
import datetime

class Trainer(object):
    def __init__(self, model, train_data, eval_data, optim, device, loss_func, topk, args):
        self.model = model
        self.train_data = train_data
        self.eval_data = eval_data
        self.optim = optim
        self.loss_func = loss_func
        self.topk = topk
        self.evaluation = Evaluation(self.model, self.loss_func, device, self.topk, warm_start=args.warm_start)
        self.device = device
        # self.device = torch.device('cuda' if use_cuda else 'cpu')
        self.args = args

    def train(self, start_epoch, end_epoch, batch_size, output_f, start_time=None):
        if start_time is None:
            self.start_time = time.time()
        else:
            self.start_time = start_time

        for epoch in range(start_epoch, end_epoch + 1):
            print("*"*10, epoch, "*"*5)
            output_f.write("*"*10+str(epoch)+"*"*5+"\n")
            st = time.time()
            train_loss = self.train_epoch(epoch, batch_size)
            loss, recall, mrr = self.evaluation.eval(self.train_data, batch_size)
           
            print("train Epoch: {}, train loss: {:.4f},  loss: {:.4f},recall: {:.4f}, mrr: {:.4f}, time: {}".format(epoch, train_loss, loss, recall, mrr, time.time() - st))
            output_f.write("train Epoch: {}, train loss: {:.4f},  loss: {:.4f},recall: {:.4f}, mrr: {:.4f}, time: {}".format(epoch, train_loss, loss, recall, mrr, time.time() - st)+"\n")

            loss, recall, mrr = self.evaluation.eval(self.eval_data, batch_size)
            print("Epoch: {}, loss: {:.4f}, recall: {:.4f}, mrr: {:.4f}, time: {}".format(epoch, loss, recall, mrr, time.time() - st))
            output_f.write("Epoch: {}, loss: {:.4f}, recall: {:.4f}, mrr: {:.4f}, time: {}".format(epoch, loss, recall, mrr, time.time() - st)+"\n")
            output_f.flush()
#             checkpoint = {
#                 'model': self.model.state_dict(),
#                 'args': self.args,
#                 'epoch': epoch,
#                 'optim': self.optim,
#                 'loss': loss,
#                 'recall': recall,
#                 'mrr': mrr
#             }
#             model_name = os.path.join(self.args.checkpoint_dir, "model_{0:05d}.pt".format(epoch))
#             torch.save(checkpoint, model_name)
#             print("Save model as %s" % model_name)

    def train_epoch(self, epoch, batch_size):
        self.model.train()
        losses = []

        def reset_hidden(hidden, mask):
            """Helper function that resets hidden state when some sessions terminate"""
            if len(mask) != 0:
                hidden[:, mask, :] = 0
            return hidden
       
        dataloader = self.train_data
        for x_batch, y_batch, _, mask_batch, max_subseqNum, max_acticonNum, subseqLen_batch, seqLen_batch in dataloader:
            st = datetime.datetime.now()
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            mask_batch = mask_batch.to(self.device)

            # batch_size = x_batch.size(0)

            self.optim.zero_grad()
            # hidden_subseq = self.model.init_hidden(batch_size)
            # hidden_seq = self.model.init_hidden(batch_size)

            logit_batch = self.model(x_batch, mask_batch, max_subseqNum, max_acticonNum, subseqLen_batch, seqLen_batch, self.device)

            # y_batch = y_batch[x_seq_index_list]
            ### batch_size*batch_size
            logit_sampled_batch = logit_batch[:, y_batch.view(-1)]

            loss_batch = self.loss_func(logit_sampled_batch, y_batch)
            losses.append(loss_batch.item())
            loss_batch.backward()
            max_norm = 5.0

            torch.nn.utils.clip_grad_norm(self.model.parameters(), max_norm)

            self.optim.step()
            et = datetime.datetime.now()
            print("batch time", et-st)

        mean_losses = np.mean(losses)
        return mean_losses