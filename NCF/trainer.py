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

	def train(self, start_epoch, end_epoch, batch_size, start_time=None):
		if start_time is None:
			self.start_time = time.time()
		else:
			self.start_time = start_time

		for epoch in range(start_epoch, end_epoch+1):
			st = time.time()
			train_loss = self.train_epoch(epoch, batch_size)
			loss, recall, mrr = self.evaluation.eval(self.eval_data, batch_size)

			# print(epoch)
			# print(loss)
			# print(recall)
			# print(mrr)
			print("epoch: {}, loss: {:.2f}, recall:{:.2f}, mrr:{:.2f}, time:{}".format(epoch, loss, recall, mrr, time.time()-st))

			checkpoint = {'model':self.model, 'args':self.args, 'epoch':epoch, 'optim':self.optim, 'loss':loss, 'recall':recall, 'mrr':mrr}

			model_name = os.path.join(self.args.checkpoint_dir, "model_{0:05d}.pt".format(epoch))
			torch.save(checkpoint, model_name)
			print("save model as %s"%model_name)

	def train_epoch(self, epoch, batch_size):
		self.model.train()

		total_loss = 0.0

		# dataloader = DataLoader(self.train_data, batch_size)
		for user, item, target in self.train_data:
			user = user.to(self.device)	
			item = item.to(self.device)
			target = target.to(self.device)

			loss = self.train_single_batch(user, item, target)

			total_loss += loss

		# self._writer.add_scalar('model/loss', total_loss, epoch_id)

	def train_single_batch(self, user, item, target):
		
		self.optim.zero_grad()

		target_pred = self.model(user, item)
		loss = self.loss_func(target_pred, target)

		loss.backward()

		self.optim.step()
		loss = loss.item()

		return loss