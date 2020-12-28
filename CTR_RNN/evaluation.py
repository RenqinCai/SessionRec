import numpy as np
import torch
import dataset
from metric import *
import datetime
import torch.nn.functional as F
import random

class Evaluation(object):
	def __init__(self, log, model, loss_func, use_cuda, multiGPU, k=20, warm_start=5):
		self.model = model
		self.loss_func = loss_func

		self.topk = k
		self.warm_start = warm_start
		self.device = torch.device('cuda' if use_cuda else 'cpu')

		self.m_log = log
		self.m_multiGPU = multiGPU

	def eval(self, eval_data, batch_size, train_test_flag):
		self.model.eval()

		losses = []
		recalls = []
		mrrs = []
		weights = []

		dataloader = eval_data

		with torch.no_grad():
			total_test_num = []

			for batch_self_src, batch_common_src, batch_common_time, batch_friend_diff_src, batch_friend_num, batch_y, batch_y_id in dataloader:

			# for batch_self_src, batch_y, batch_y_id in dataloader:
				if train_test_flag == "train":
					eval_flag = random.randint(1, 101)
					if eval_flag != 10:
						continue

				batch_self_src = batch_self_src.to(self.device)
				batch_common_src = batch_common_src.to(self.device)
				batch_common_time = batch_common_time.to(self.device)
				batch_friend_diff_src = batch_friend_diff_src.to(self.device)
				batch_friend_num_tensor = torch.tensor(batch_friend_num).to(self.device)
				batch_y = batch_y.to(self.device)
			
				warm_start_mask = (batch_y_id>=self.warm_start)

				if self.m_multiGPU:
					batch_common_src = torch.split(batch_common_src, split_size_or_sections=batch_friend_num, dim=0)
					batch_common_src = torch.nn.utils.rnn.pad_sequence(batch_common_src, batch_first=True)

					batch_common_time = torch.split(batch_common_time, split_size_or_sections=batch_friend_num, dim=0)
					batch_common_time = torch.nn.utils.rnn.pad_sequence(batch_common_time, batch_first=True)

					batch_friend_diff_src = torch.split(batch_friend_diff_src, split_size_or_sections=batch_friend_num, dim=0)
					batch_friend_diff_src = torch.nn.utils.rnn.pad_sequence(batch_friend_diff_src, batch_first=True)

					batch_friend_num_tensor = batch_friend_num_tensor.unsqueeze(-1)

				output_batch = self.model(batch_self_src, batch_common_src, batch_common_time, batch_friend_diff_src, batch_friend_num_tensor)

				# output_batch = self.model(batch_self_src)

				if self.m_multiGPU:
					sampled_logit_batch, sampled_target_batch = self.model.module.sample_loss(output_batch, batch_y, None, None, None, None, None, "full")
				else:
					sampled_logit_batch, sampled_target_batch = self.model.sample_loss(output_batch, batch_y, None, None, None, None, None, "full")
				# sampled_logit_batch, sampled_target_batch = self.model.sample_loss(output_batch, batch_y, None, None, None, None, None, "full")

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

				total_test_num.append(batch_y.view(-1).size(0))

		# print("weights", weights)
		mean_losses = np.mean(losses)
		mean_recall = np.average(recalls, weights=weights)
		mean_mrr = np.average(mrrs, weights=weights)

		msg = "total_test_num"+str(np.sum(total_test_num))
		self.m_log.addOutput2IO(msg)

		return mean_losses, mean_recall, mean_mrr
