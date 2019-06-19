import numpy as np
import torch
import dataset
from metric import *

class Evaluation(object):
	def __init__(self, model, loss_func, device, k=20, warm_start=5):
		self.model = model
		self.loss_func = loss_func

		self.topk = k
		self.warm_start = warm_start
		self.device = device

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

		with torch.no_grad():
			total_test_num = []

			for x_batch, y_batch, idx_batch, mask_batch, max_subseqNum, max_acticonNum, subseqLen_batch, seqLen_batch in dataloader:
				x_batch = x_batch.to(self.device)
				y_batch = y_batch.to(self.device)
				warm_start_mask = (idx_batch>=self.warm_start).to(self.device)
				mask_batch = mask_batch.to(self.device)
                
				# hidden = self.model.init_hidden()

				logit_batch = self.model(x_batch, mask_batch, max_subseqNum, max_acticonNum, subseqLen_batch, seqLen_batch)
				
				logit_sampled_batch = logit_batch[:, y_batch.view(-1)]
				loss_batch = self.loss_func(logit_sampled_batch, y_batch)

				losses.append(loss_batch.item())
				
				recall_batch, mrr_batch = evaluate(logit_batch, y_batch, warm_start_mask, k=self.topk)

				weights.append( int( warm_start_mask.int().sum() ) )
				recalls.append(recall_batch)
				mrrs.append(mrr_batch)

				total_test_num.append(y_batch.view(-1).size(0))

		mean_losses = np.mean(losses)
		mean_recall = np.average(recalls, weights=weights)
		mean_mrr = np.average(mrrs, weights=weights)
		print("total_test_num", np.sum(total_test_num))

		return mean_losses, mean_recall, mean_mrr
