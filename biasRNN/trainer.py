
# import lib
from evaluation import *
import time
import torch
import numpy as np
import os
from dataset import *
import datetime

import sys
sys.path.insert(0, '../PyTorch_GBW_LM')
sys.path.insert(0, '../PyTorch_GBW_LM/log_uniform')

from log_uniform import LogUniformSampler

class Trainer(object):
    def __init__(self, log, model, train_data, eval_data, optim, use_cuda, loss_func, topk, sample_full_flag, input_size, args):
        self.model = model
        self.train_data = train_data
        self.eval_data = eval_data
        self.optim = optim
        self.loss_func = loss_func
        self.topk = topk
        self.evaluation = Evaluation(log, self.model, self.loss_func, use_cuda, self.topk, warm_start=args.warm_start)
        self.device = torch.device('cuda' if use_cuda else 'cpu')
        self.args = args
        self.m_log = log
        self.m_sample_full_flag = sample_full_flag

        self.m_patience = args.patience
        self.m_best_recall = 0.0
        self.m_best_mrr = 0.0
        self.m_early_stop = False
        self.m_counter = 0
        self.m_batch_iter = 0

        self.m_sampler = LogUniformSampler(input_size)
        self.m_nsampled = args.negative_num
        self.m_remove_match = True

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
        
        # checkpoint_dir = "../log/"+self.args.model_name+"/"+self.args.checkpoint_dir
        # model_name = os.path.join(self.args.checkpoint_dir, "model_{0:05d}.pt".format(epoch))
        model_name = os.path.join(self.args.checkpoint_dir, "model_best.pt")
        # model_name = os.path.join(self.args.checkpoint_dir, "model_{0:05d}.pt".format(epoch))
        torch.save(checkpoint, model_name)

    def train(self, start_epoch, end_epoch, batch_size, start_time=None):
        if start_time is None:
            self.start_time = time.time()
        else:
            self.start_time = start_time

        for epoch in range(start_epoch, end_epoch + 1):
            msg = "*"*15+str(epoch)+"*"*15
            self.m_log.addOutput2IO(msg)

            st = time.time()
            train_loss = self.train_epoch(epoch, batch_size)
            loss, recall, mrr = self.evaluation.eval(self.train_data, batch_size, "train")
           
            msg = "train Epoch: {}, train loss: {:.4f},  loss: {:.4f},recall: {:.4f}, mrr: {:.4f}, time: {}".format(epoch, train_loss, loss, recall, mrr, time.time() - st)
            self.m_log.addOutput2IO(msg)
            self.m_log.addScalar2Tensorboard("train_loss", train_loss, epoch)
            self.m_log.addScalar2Tensorboard("train_loss_eval", loss, epoch)
            self.m_log.addScalar2Tensorboard('train_recall', recall, epoch)
            self.m_log.addScalar2Tensorboard("train_mrr", mrr, epoch)           
            loss, recall, mrr = self.evaluation.eval(self.eval_data, batch_size, "test")
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
                self.m_best_mrr = mrr
                self.saveModel(epoch, loss, recall, mrr)
                self.m_counter = 0

            msg = "best recall: "+str(self.m_best_recall)+"\t best mrr: \t"+str(self.m_best_mrr)
            self.m_log.addOutput2IO(msg)

    def train_epoch(self, epoch, batch_size):
        self.model.train()
        losses = []

        def reset_hidden(hidden, mask):
            """Helper function that resets hidden state when some sessions terminate"""
            if len(mask) != 0:
                hidden[:, mask, :] = 0
            return hidden
       
        dataloader = self.train_data

        batch_index = 0

        for x_short_action_batch, mask_short_action_batch, pad_x_short_actionNum_batch, y_action_batch, y_action_idx_batch in dataloader:
            
            sample_ids = None 
            true_freq = None
            sample_freq = None
            acc_hits = None
            sampled_logit_batch = None
            sampled_target_batch = None

            if self.m_sample_full_flag == "sample":
                sample_values = self.m_sampler.sample(self.m_nsampled, y_action_batch)
                sample_ids, true_freq, sample_freq = sample_values
                # print("sample ids size", sample_ids.size())

                if self.m_remove_match:
                    acc_hits = self.m_sampler.accidental_match(y_action_batch, sample_ids)
                    acc_hits = list(zip(*acc_hits))

            x_short_action_batch = x_short_action_batch.to(self.device)
            mask_short_action_batch = mask_short_action_batch.to(self.device)
            # x_short_cate_batch = x_short_cate_batch.to(self.device)

            y_action_batch = y_action_batch.to(self.device)
            y_action_idx_batch = y_action_idx_batch.to(self.device)

            if batch_index%10000 == 0:
                print("batch_index", batch_index)
           
            self.optim.zero_grad()

            output_batch = self.model(x_short_action_batch, mask_short_action_batch, pad_x_short_actionNum_batch)

            if self.m_sample_full_flag == "sample":
                sampled_logit_batch, sampled_target_batch = self.model.m_ss(output_batch, y_action_batch, sample_ids, true_freq, sample_freq, acc_hits, self.device, self.m_remove_match, "sample")
            
            if self.m_sample_full_flag == "full":
                sampled_logit_batch, sampled_target_batch = self.model.m_ss(output_batch, y_action_batch, sample_ids, true_freq, sample_freq, acc_hits, self.device, self.m_remove_match, "full")
        
            loss_batch = self.loss_func(sampled_logit_batch, sampled_target_batch)
            losses.append(loss_batch.item())
            loss_batch.backward()
            max_norm = 5.0
        
            self.m_batch_iter += 1

            torch.nn.utils.clip_grad_norm(self.model.parameters(), max_norm)

            self.optim.step()

            batch_index += 1
            # et = datetime.datetime.now()
            # print("duration batch", et-st)
        # for name, param in self.model.named_parameters():
        #     if name == "m_ss.params.bias":
        # param = self.model.
        # self.m_log.addHistogram2Tensorboard(name, param.clone().cpu().data.numpy(), epoch)

        mean_losses = np.mean(losses)
        return mean_losses