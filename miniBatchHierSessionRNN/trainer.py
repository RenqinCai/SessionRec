# import lib
from evaluation import *
import time
import torch
import numpy as np
import os
from dataset import *


class Trainer(object):
	def __init__(self, model, train_data, eval_data, optim, use_cuda, loss_func, topk, args):
		self.model = model
		self.train_data = train_data
		self.eval_data = eval_data
		self.optim = optim
		self.loss_func = loss_func
		self.topk = topk
		self.evaluation = Evaluation(self.model, self.loss_func, use_cuda, self.topk)
		self.device = torch.device('cuda' if use_cuda else 'cpu')
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
			output_f.write("Epoch: {}, loss: {:.4f}, recall: {:.4f}, mrr: {:.4f}, time: {}".format(epoch, loss, recall, mrr, time.time() - st))

			loss, recall, mrr = self.evaluation.eval(self.eval_data, batch_size)
			print("Epoch: {}, loss: {:.4f}, recall: {:.4f}, mrr: {:.4f}, time: {}".format(epoch, loss, recall, mrr, time.time() - st))
			output_f.write("Epoch: {}, loss: {:.4f}, recall: {:.4f}, mrr: {:.4f}, time: {}".format(epoch, loss, recall, mrr, time.time() - st))
			output_f.flush()


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
		
		dataloader = self.train_data
		sess_hidden = self.model.init_hidden()
		user_hidden = self.model.init_hidden()

		for idx_input, idx_target, mask_sess_start, mask_user_start, start_user_mask, _ in dataloader:
			idx_input = idx_input.to(self.device)
			idx_target = idx_target.to(self.device)
			self.optim.zero_grad()

			# print("idx_input", idx_input)
			# print("idx_target", idx_target)

			# hidden = reset_hidden(hidden, mask).detach()
			# sess_hidden, user_hidden = reset_hidden(sess_hidden, user_hidden, mask_sess, mask_user)
			# print("input size", input.size())
			
			logit, sess_hidden, user_hidden = self.model(idx_input, sess_hidden, user_hidden, mask_sess_start, mask_user_start)

			sess_hidden = sess_hidden.detach()
			user_hidden = user_hidden.detach()
			# output sampling
			# exit()
			logit_sampled = logit[:, idx_target.view(-1)]
			loss = self.loss_func(logit_sampled)
			losses.append(loss.item())
			loss.backward()
			self.optim.step()

		mean_losses = np.mean(losses)
		return mean_losses