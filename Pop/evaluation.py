import numpy as np
import torch
import dataset
from metric import *

class Evaluation(object):
	def __init__(self, model, use_cuda, k=20):
		self.model = model

		self.topk = k
		self.device = torch.device('cuda' if use_cuda else 'cpu')

	def eval(self, eval_data, batch_size):

		recalls = []
		mrrs = []

		# losses = None
		# recalls = None
		# mrrs = None

		dataloader = eval_data

		eval_iter = 0

		with torch.no_grad():

			total_test_num = []
			for idx_input, input, target, mask in dataloader:
				input = input.to(self.device)
				target = target.to(self.device)

				logit = self.model.test(input)

				recall, mrr = evaluate(logit, target, k=self.topk)

				eval_iter += 1

				total_test_num.append(target.view(-1).size(0))

				recalls.append(recall)
				mrrs.append(mrr.item())

		# mean_losses = np.mean(losses)
		# mean_recall = np.mean(recalls)
		# mean_mrr = np.mean(mrrs)
		recall_sum = np.sum(recalls)
		mrr_sum = np.sum(mrrs)

		total_test_num = np.sum(total_test_num)
		print("total_test_num", total_test_num)

		mean_recall = recall_sum/total_test_num
		mean_mrr = mrr_sum/total_test_num

		return mean_recall, mean_mrr

