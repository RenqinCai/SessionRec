# import lib
from evaluation import *
import time
import torch
import numpy as np
import os
from torch.utils.tensorboard import SummaryWriter


class Trainer(object):
    def __init__(self, model, train_data, eval_data, optim, use_cuda, loss_func, topk, args):
        self.model = model
        self.train_data = train_data
        self.eval_data = eval_data
        self.optim = optim
        self.loss_func = loss_func
        self.topk = topk
        self.evaluation = Evaluation(self.model, self.loss_func, use_cuda, self.topk, warm_start=args.warm_start)
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
            train_loss = self.train_epoch(epoch, batch_size)
            loss, recall, mrr = self.evaluation.eval(self.train_data, batch_size)
            print("Train Epoch: {}, train loss: {:.4f},  loss: {:.4f}, recall: {:.4f}, mrr: {:.4f}, time: {}".format(epoch, train_loss, loss, recall, mrr, time.time() - st))

            loss, recall, mrr = self.evaluation.eval(self.eval_data, batch_size)
            print("Test  Epoch: {}, loss: {:.4f}, recall: {:.4f}, mrr: {:.4f}, time: {}".format(epoch, loss, recall, mrr, time.time() - st))

#             print('embedding', self.model.out_matrix.data[1])
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
            print("Save model as %s" % model_name)

    def train_epoch(self, epoch, batch_size):
        self.model.train()
        losses = []
        torch.autograd.set_detect_anomaly(False)
        
        logging = True
#         writer = SummaryWriter()

        dataloader = self.train_data
        for input_x_batch, target_y_batch, input_t_batch, x_len_batch in dataloader:
            input_x_batch = input_x_batch.to(self.device)
            input_t_batch = input_t_batch.to(self.device)
            target_y_batch = target_y_batch.to(self.device)

            self.optim.zero_grad()
            
            logit_batch = self.model(input_x_batch, input_t_batch)
            

            
            ### batch_size*batch_size
            logit_sampled_batch = logit_batch[:, target_y_batch.view(-1)]
#             print(logit_sampled_batch.size(), target_y_batch.size() )
#             print(logit_sampled_batch)
#             print(target_y_batch)
            
            loss_batch = self.loss_func(logit_sampled_batch, target_y_batch)
        
#             if logging:
#                 self.model.beta.retain_grad()
#                 self.model.gamma.retain_grad()
# #                 self.model.alpha.retain_grad()
# #                 writer.add_graph(self.model, input_to_model= (input_x_batch, input_t_batch) )
#                 logging = False
    
#             print(loss_batch)
            losses.append(loss_batch.item())
    
#             print('beta:', self.model.beta.data, self.model.beta.grad)
#             print('gamma:', self.model.gamma.data, self.model.gamma.grad)
#             print('alpha:', self.model.alpha.data, self.model.alpha.grad)
            loss_batch.backward()

            self.optim.step()
         
#         writer.close()      
            
        print(list(self.model.parameters()) )
        print('beta:', self.model.beta.data, self.model.beta.grad)
        print('gamma:', self.model.gamma.data, self.model.gamma.grad)
        
        return np.mean(losses)
