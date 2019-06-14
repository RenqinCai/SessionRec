import numpy as np
import torch
import dataset
from metric import *

class Evaluation(object):
	def __init__(self, model, loss_func, use_cuda, k=20, warm_start=5):
		self.model = model
		self.loss_func = loss_func
		self.warm_start = warm_start
		self.topk = k
		self.device = torch.device('cuda' if use_cuda else 'cpu')

	def eval(self, eval_data, batch_size):
		self.model.eval()

		losses = []
		recalls = []
		mrrs = []
		weights = []
		# losses = None
		# recalls = None
		# mrrs = None

		dataloader = eval_data

		eval_iter = 0

		def reset_hidden(hidden, mask):
			if len(mask) != 0:
				hidden[:, mask, :] = 0

			return hidden

		with torch.no_grad():
			hidden = self.model.init_hidden()
			total_test_num = []
			for x_input, y_target, mask, idx_tensor in dataloader:
				x_input = x_input.to(self.device)
				y_target = y_target.to(self.device)
				warm_start_mask = (idx_tensor >= self.warm_start).to(self.device)

				# hidden = reset_hidden(hidden, mask)
				hidden = reset_hidden(hidden, mask).detach()
				# hidden = self.model.init_hidden()
				logit, hidden = self.model(x_input, hidden)
				# print("preds", logit.size())
				logit_sampled = logit[:, y_target.view(-1)]
				loss = self.loss_func(logit_sampled)

				recall, mrr = evaluate(logit, y_target, warm_start_mask, k=self.topk)

				eval_iter += 1

				total_test_num.append(y_target.view(-1).size(0))

				weights.append( int( warm_start_mask.int().sum() ) )
				losses.append(loss.item())
				recalls.append(recall)
				mrrs.append(mrr)

		mean_losses = np.mean(losses)
		mean_recall = np.average(recalls, weights=weights)
		mean_mrr = np.average(mrrs, weights=weights)
		print("total_test_num", np.sum(total_test_num))

		return mean_losses, mean_recall, mean_mrr

