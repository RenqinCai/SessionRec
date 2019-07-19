
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

        self.m_patience = args.patience
        self.m_best_recall = 0.0
        self.m_best_mrr = 0.0
        self.m_early_stop = False
        self.m_counter = 0
        self.m_batch_iter = 0

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
            msg = "*"*10+str(epoch)+"*"*5
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

        for x_batch, y_batch, x_len_batch, _ in dataloader:
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            if batch_index%10000 == 0:
                print("batch_index", batch_index)
            # st = datetime.datetime.now()
            # input_x_batch, target_y_batch, x_len_batch = batchifyData(input_x, target_y)
    
            self.optim.zero_grad()
            hidden = self.model.init_hidden()

            output_batch = self.model(x_batch, hidden, x_len_batch)

            # output_batch.retain_grad()
            # output_batch.register_hook(print)
            # name = "last output"
            # if output_batch.grad:
            #     print(output_batch.grad)
        # # # print("last_outputgrad", last_output.grad)
        #     if output_batch.grad:
        #         self.m_log.addHistogram2Tensorboard(name, output_batch.grad, self.m_batch_iter)

            sampled_logit_batch, sampled_target_batch = self.model.m_ss(output_batch, y_batch)

            # sampled_logit_batch.retain_grad()

            loss_batch = self.loss_func(sampled_logit_batch, sampled_target_batch)
            losses.append(loss_batch.item())
            loss_batch.backward()
            max_norm = 5.0
            
            # name = "sampled_logit_batch"
            # if self.m_batch_iter > 20000:
            #     print(name, sampled_logit_batch.grad[0])
            # print("size", sampled_logit_batch.grad.size())
            # print(torch.max(sampled_logit_batch.grad[0]), torch.min(sampled_logit_batch.grad[0]))

            # self.m_log.addHistogram2Tensorboard(name, sampled_logit_batch.grad[0], self.m_batch_iter)
        
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