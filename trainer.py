import time
import torch
import numpy as np
import os
from dataset import *
from evaluation import *

class Trainer(object):
	def __init__(self, model, train_data, eval_data, optim, use_cuda, loss_func, args):
		self.model = model
		self.train_data = train_data
		self.eval_data = eval_data

		self.optim = optim
		self.loss_func = loss_func
		self.evaluation = Evaluation(self.model, self.loss_func, use_cuda)

		self.device = torch.device('cuda' if use_cuda else 'cpu')

		self.args = args

	def train_epoch(self, epoch, batchSize, tensorBoardWriter):
		self.model.train()
		losses = []


		def reset_hidden(hidden, mask):
			if len(mask) != 0:
				hidden[:, mask, :] = 0

			return hidden

		hidden = self.model.init_hidden()
		
		dataloader = DataLoader(self.train_data, batchSize)

		back_iter = epoch*7409+0

		for input, target, mask in dataloader:
			input = input.to(self.device)
			target = target.to(self.device)
			self.optim.zero_grad()

			# print("input size", input.size())

			hidden = reset_hidden(hidden, mask).detach()
			logit, hidden = self.model(input, hidden)
			logit_sampled = logit[:, target.view(-1)]
			loss = self.loss_func(logit_sampled)
			losses.append(loss.item())
			loss.backward()

			self.optim.step()

			tensorBoardWriter.add_scalar("train_loss", loss, back_iter)
			back_iter += 1

		print("back iter", back_iter)

		mean_losses = np.mean(losses)
		return mean_losses

	def train(self, tensorBoardWriter, start_epoch, end_epoch, batchSize, start_time=None):
		if start_time is None:
			self.start_time = time.time()
		else:
			self.start_time = start_time

		for epoch in range(start_epoch, end_epoch+1):
			st = time.time()

			train_loss = self.train_epoch(epoch, batchSize, tensorBoardWriter)
			loss, recall, mrr = self.evaluation.eval(self.eval_data, batchSize)

			tensorBoardWriter.add_scalar("val_loss", loss, epoch)

			loss = loss.item()
			mrr = mrr.item()

			print("Epoch: {}, loss: {:.2f}, recall: {:.2f}, mrr: {:.2f}, time: {}".format(epoch, loss, recall, mrr, time.time() - st))
			checkpoint = {
				'model': self.model,
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