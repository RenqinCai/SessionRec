import numpy as np
import torch
import dataset
from metric import *
import datetime
import torch.nn.functional as F
import random

class Evaluation(object):
	def __init__(self, log, model, loss_func, use_cuda, k=20, warm_start=5):
		self.model = model
		self.loss_func = loss_func

		self.topk = k
		self.warm_start = warm_start
		self.device = torch.device('cuda' if use_cuda else 'cpu')

		self.m_log = log

	def eval(self, eval_data, batch_size, train_test_flag):
		self.model.eval()

		losses = []
		recalls = []
		mrrs = []
		weights = []

		# losses = None
		# recalls = None
		# mrrs = None

		dataloader = eval_data

		with torch.no_grad():
			total_test_num = []

			for x_short_action_batch, mask_short_action_batch, pad_x_short_actionNum_batch, y_action_batch, y_action_idx_batch in dataloader:

				if train_test_flag == "train":
					eval_flag = random.randint(1,101)
					if eval_flag != 10:
						continue

				x_short_action_batch = x_short_action_batch.to(self.device)
				mask_short_action_batch = mask_short_action_batch.to(self.device)
				y_action_batch = y_action_batch.to(self.device)
			
				warm_start_mask = (y_action_idx_batch>=self.warm_start)
	
				output_batch = self.model(x_short_action_batch, mask_short_action_batch, pad_x_short_actionNum_batch)

				sampled_logit_batch, sampled_target_batch = self.model.m_ss(output_batch, y_action_batch, None, None, None, None, None, None, "full")

				loss_batch = self.loss_func(sampled_logit_batch, sampled_target_batch)
				losses.append(loss_batch.item())

				# et_2 = datetime.datetime.now()
				# print("duration 2", et_2-et_1)

				# logit_batch = self.model.m_ss.params(output_batch)
				recall_batch, mrr_batch = evaluate(sampled_logit_batch, sampled_target_batch, warm_start_mask, k=self.topk)

				weights.append( int( warm_start_mask.int().sum() ) )
				recalls.append(recall_batch)
				mrrs.append(mrr_batch)

				# et_3 = datetime.datetime.now()
				# print("duration 3", et_3-et_2)

				total_test_num.append(y_action_batch.view(-1).size(0))

		mean_losses = np.mean(losses)
		mean_recall = np.average(recalls, weights=weights)
		mean_mrr = np.average(mrrs, weights=weights)

		msg = "total_test_num"+str(np.sum(total_test_num))
		self.m_log.addOutput2IO(msg)

		return mean_losses, mean_recall, mean_mrr
