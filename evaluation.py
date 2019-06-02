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

		def reset_hidden(hidden, mask):
			if len(mask) != 0:
				hidden[:, mask, :] = 0

			return hidden

		with torch.no_grad():
			hidden = self.model.init_hidden()
			total_test_num = []
			for idx_input, input, target, mask in dataloader:
				input = input.to(self.device)
				target = target.to(self.device)

				hidden = reset_hidden(hidden, mask)
				# hidden = reset_hidden(hidden, mask).detach()

				logit, hidden = self.model(input, hidden)
				# print("preds", logit.size())
				logit_sampled = logit[:, target.view(-1)]
				loss = self.loss_func(logit_sampled)

				recall, mrr = evaluate(logit, target, k=self.topk)

				eval_iter += 1

				total_test_num.append(target.view(-1).size(0))

				losses.append(loss.item())
				recalls.append(recall)
				mrrs.append(mrr.item())

		mean_losses = np.mean(losses)
		mean_recall = np.mean(recalls)
		mean_mrr = np.mean(mrrs)
		print("total_test_num", np.sum(total_test_num))

		return mean_losses, mean_recall, mean_mrr

