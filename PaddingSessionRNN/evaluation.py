import numpy as np
import torch
import dataset
from metric import *

class Evaluation(object):
	def __init__(self, model, loss_func, use_cuda, k=20):
		self.model = model
		self.loss_func = loss_func

		self.topk = k
		self.device = torch.device('cuda' if use_cuda else 'cpu')

	def eval(self, eval_data, batch_size):
		self.model.eval()

		losses = []
		recalls = []
		mrrs = []

		# losses = None
		# recalls = None
		# mrrs = None

		dataloader = eval_data

		eval_iter = 0

		with torch.no_grad():
			total_test_num = []
			for input_x_batch, target_y_batch, x_len_batch in dataloader:
				input_x_batch = input_x_batch.to(self.device)
				target_y_batch = target_y_batch.to(self.device)
				
				# input_x_batch, target_y_batch, x_len_batch = batchifyData(input_x, target_y)

				hidden = self.model.init_hidden()

				logit_batch, hidden = self.model(input_x_batch, hidden, x_len_batch)

				logit_sampled_batch = logit_batch[:, target_y_batch.view(-1)]
				loss_batch = self.loss_func(logit_sampled_batch, target_y_batch)

				losses.append(loss_batch.item())
				
				recall_batch, mrr_batch = evaluate(logit_batch, target_y_batch, k=self.topk)

				recalls.append(recall_batch)
				mrrs.append(mrr_batch)

				total_test_num.append(target_y_batch.view(-1).size(0))

		# def reset_hidden(hidden, mask):
		# 	if len(mask) != 0:
		# 		hidden[:, mask, :] = 0

		# 	return hidden

		# with torch.no_grad():
		# 	hidden = self.model.init_hidden()
		# 	total_test_num = []
		# 	for idx_input, input, target, mask in dataloader:
		# 		input = input.to(self.device)
		# 		target = target.to(self.device)

		# 		hidden = reset_hidden(hidden, mask).detach()

		# 		logit, hidden = self.model(input, hidden)
		# 		# print("preds", logit.size())
		# 		logit_sampled = logit[:, target.view(-1)]
		# 		loss = self.loss_func(logit_sampled)

		# 		recall, mrr = evaluate(logit, target, k=self.topk)

		# 		eval_iter += 1

		# 		total_test_num.append(target.view(-1).size(0))

		# 		losses.append(loss.item())
		# 		recalls.append(recall)
		# 		mrrs.append(mrr.item())

		mean_losses = np.mean(losses)
		mean_recall = np.mean(recalls)
		mean_mrr = np.mean(mrrs)
		print("total_test_num", np.sum(total_test_num))

		return mean_losses, mean_recall, mean_mrr

