# import lib
from evaluation import *
import time
import torch
import numpy as np
import os
from dataset import *
import datetime

class Trainer(object):
    def __init__(self, log, model, train_data, eval_data, optim, use_cuda, loss_func, topk, args):
        self.m_log = log
        self.model = model
        self.train_data = train_data
        self.eval_data = eval_data
        self.optim = optim
        self.loss_func = loss_func
        self.topk = topk
        self.evaluation = Evaluation(self.m_log, self.model, self.loss_func, use_cuda, self.topk, warm_start=args.warm_start)
        self.device = torch.device('cuda' if use_cuda else 'cpu')
        self.args = args

        ### early stopping
        self.m_patience = args.patience
        self.m_best_recall = 0.0
        self.m_early_stop = False
        self.m_counter = 0

    def saveModel(self, epoch, loss, recall, mrr):
        checkpoint = {
            'model': self.model.state_dict(),
            'args': self.args,
            'epoch': epoch,
            'optim': self.optim,
            'loss': loss,
            'recall': recall,
            'mrr': mrr
        }
        model_name = os.path.join(self.args.checkpoint_dir, "model_{0:05d}.pt".format(epoch))
        torch.save(checkpoint, model_name)

    def train(self, start_epoch, end_epoch, batch_size, start_time=None):

        if start_time is None:
            self.start_time = time.time()
        else:
            self.start_time = start_time

        for epoch in range(start_epoch, end_epoch + 1):
            
            msg = "*"*10+str(epoch)+"*"*5
            self.m_log.addOutput2IO(msg)

            st = time.time()
            train_loss = self.train_epoch(epoch, batch_size)
            et = time.time()
            print("train duration", et-st)

            loss, recall, mrr = self.evaluation.eval(self.train_data, batch_size)

            msg = "train Epoch: {}, train loss: {:.4f},  loss: {:.4f},recall: {:.4f}, mrr: {:.4f}, time: {}".format(epoch, train_loss, loss, recall, mrr, time.time() - st)
            self.m_log.addOutput2IO(msg)
            self.m_log.addScalar2Tensorboard("train_loss", train_loss, epoch)
            self.m_log.addScalar2Tensorboard("train_loss_eval", loss, epoch)
            self.m_log.addScalar2Tensorboard("train_recall", recall, epoch)
            self.m_log.addScalar2Tensorboard("train_mrr", mrr, epoch)

            loss, recall, mrr = self.evaluation.eval(self.eval_data, batch_size)
            msg = "Epoch: {}, loss: {:.4f}, recall: {:.4f}, mrr: {:.4f}, time: {}".format(epoch, loss, recall, mrr, time.time() - st)
            self.m_log.addOutput2IO(msg)
            self.m_log.addScalar2Tensorboard("test_loss", loss, epoch)
            self.m_log.addScalar2Tensorboard("test_recall", recall, epoch)
            self.m_log.addScalar2Tensorboard("test_mrr", mrr, epoch)
            
            if self.m_best_recall == 0:
                self.m_best_recall = recall
                self.saveModel(epoch, loss, recall, mrr)
            elif self.m_best_recall > recall:
                self.m_counter += 1
                if self.m_counter > self.m_patience:
                    break
                msg = "early stop counter "+str(self.m_counter)
                self.m_log.addOutput2IO(msg)
            else:
                self.m_best_recall = recall
                self.saveModel(epoch, loss, recall, mrr)
                self.m_counter = 0
           

    def train_epoch(self, epoch, batch_size):
        self.model.train()
        losses = []

        def reset_hidden(hidden, mask):
            """Helper function that resets hidden state when some sessions terminate"""
            if len(mask) != 0:
                hidden[:, mask, :] = 0
            return hidden
       
        dataloader = self.train_data
        for x_cate_batch, mask_cate, mask_cate_seq, max_acticonNum_cate, max_subseqNum_cate, subseqLen_cate, seqLen_cate, x_batch, mask_batch, seqLen_batch, y_batch, _ in dataloader:
            x_cate_batch = x_cate_batch.to(self.device)
            mask_cate = mask_cate.to(self.device)
            mask_cate_seq = mask_cate_seq.to(self.device)

            x_batch = x_batch.to(self.device)
            mask_batch = mask_batch.to(self.device)

            y_batch = y_batch.to(self.device)

            # batch_size = x_batch.size(0)

            self.optim.zero_grad()

            logit_batch, target_batch = self.model(x_cate_batch, mask_cate, mask_cate_seq, max_acticonNum_cate, max_subseqNum_cate, subseqLen_cate, seqLen_cate, x_batch, mask_batch, seqLen_batch, y_batch, "train")
            
            loss_batch = self.loss_func(logit_batch, target_batch)
            losses.append(loss_batch.item())
            
            loss_batch.backward()
            max_norm = 5.0

            torch.nn.utils.clip_grad_norm(self.model.parameters(), max_norm)

            self.optim.step()
    
        mean_losses = np.mean(losses)
        return mean_losses