import numpy as np
import torch
import dataset
from metric import *
import datetime
import torch.nn.functional as F

class Evaluation(object):
	def __init__(self, log, model, loss_func, use_cuda, k=20, warm_start=5):
		self.model = model
		self.loss_func = loss_func

		self.topk = k
		self.warm_start = warm_start
		self.device = torch.device('cuda' if use_cuda else 'cpu')
		self.m_log = log

	def eval(self, eval_data, batch_size):
		self.model.eval()

		mixture_losses = []

		losses = []
		recalls = []
		mrrs = []
		weights = []

		cate_losses = []
		cate_recalls = []
		cate_mrrs = []
		cate_weights = []

		dataloader = eval_data

		with torch.no_grad():
			total_test_num = []

			for x_cate_batch, input_cate_subseq, mask_cate, mask_cate_seq, max_acticonNum_cate, max_subseqNum_cate, subseqLen_cate, seqLen_cate, x_batch, cate_batch, mask_batch, seqLen_batch, y_batch, y_cate, idx_batch in dataloader:

				# print("*"*10)
				x_cate_batch = x_cate_batch.to(self.device)
				mask_cate = mask_cate.to(self.device)
				mask_cate_seq = mask_cate_seq.to(self.device)
				
				x_batch = x_batch.to(self.device)
				mask_batch = mask_batch.to(self.device)

				y_batch = y_batch.to(self.device)
				cate_batch = cate_batch.to(self.device)

				y_cate = y_cate.to(self.device)

				warm_start_mask = (idx_batch>=self.warm_start)
				
				output_batch, cate_logit = self.model(x_cate_batch, input_cate_subseq, mask_cate, mask_cate_seq, max_acticonNum_cate, max_subseqNum_cate, subseqLen_cate, seqLen_cate, x_batch, mask_batch, seqLen_batch, cate_batch, "test")

				sampled_logit_batch, sampled_target_batch = self.model.m_ss(output_batch, y_batch)
				
				loss_batch = self.loss_func(sampled_logit_batch, sampled_target_batch)

				cate_loss_batch = self.loss_func(cate_logit, y_cate)

				mixture_loss_batch = loss_batch+cate_loss_batch

				mixture_losses.append(mixture_loss_batch.item())

				losses.append(loss_batch.item())
				cate_losses.append(cate_loss_batch.item())

				# logit_batch = F.linear(output_batch, self.model.m_ss.params.weight)
				# logit_batch = self.model.m_ss.params(output_batch)
				
				recall_batch, mrr_batch = evaluate(sampled_logit_batch, sampled_target_batch, warm_start_mask, k=self.topk)

				weights.append( int( warm_start_mask.int().sum() ) )
				recalls.append(recall_batch)
				mrrs.append(mrr_batch)

				cate_recall_batch, cate_mrr_batch = evaluate(cate_logit, y_cate, warm_start_mask, k=self.topk)

				cate_weights.append(int( warm_start_mask.int().sum() ))
				cate_recalls.append(cate_recall_batch)
				cate_mrrs.append(cate_mrr_batch)

				total_test_num.append(y_batch.view(-1).size(0))

		mean_mixture_losses = np.mean(mixture_losses)
		
		mean_losses = np.mean(losses)
		mean_recall = np.average(recalls, weights=weights)
		mean_mrr = np.average(mrrs, weights=weights)

		mean_cate_losses = np.mean(cate_losses)
		mean_cate_recall = np.average(cate_recalls, weights=cate_weights)
		mean_cate_mrr = np.average(cate_mrrs, weights=cate_weights)

		msg = "total_test_num"+str(np.sum(total_test_num))
		self.m_log.addOutput2IO(msg)

		return mean_mixture_losses, mean_losses, mean_recall, mean_mrr, mean_cate_losses, mean_cate_recall, mean_cate_mrr
