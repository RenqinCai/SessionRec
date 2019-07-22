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
		beam_size = 5
		self.m_beam_size = beam_size

	def eval(self, eval_data, batch_size, train_test_flag):
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
			
			# st_eval = datetime.datetime.now()

			for x_cate_batch, input_cate_subseq, mask_cate, mask_cate_seq, max_acticonNum_cate, max_subseqNum_cate, subseqLen_cate, seqLen_cate, x_batch, cate_batch, mask_batch, seqLen_batch, y_batch, y_cate, idx_batch in dataloader:
				if train_test_flag == "train":
					eval_flag = random.randint(1,101)
					if eval_flag != 10:
						continue
				# st = datetime.datetime.now()
				# print("*"*10)
				x_cate_batch = x_cate_batch.to(self.device)
				mask_cate = mask_cate.to(self.device)
				mask_cate_seq = mask_cate_seq.to(self.device)
				
				x_batch = x_batch.to(self.device)
				mask_batch = mask_batch.to(self.device)

				y_batch = y_batch.to(self.device)
				cate_batch = cate_batch.to(self.device)

				y_cate = y_cate.to(self.device)
				# input_cate_subseq = torch.from_numpy(input_cate_subseq)
				input_cate_subseq = input_cate_subseq.to(self.device)

				warm_start_mask = (idx_batch>=self.warm_start)

				### cateNN
				logit_cate_short = self.model.m_cateNN(cate_batch, mask_batch, seqLen_batch, "test")

				prob_cate_short = F.softmax(logit_cate_short, dim=-1)
				### cate_prob_beam: batch_size*beam_size
				cate_prob_beam, cate_id_beam = prob_cate_short.topk(dim=1, k=self.m_beam_size)

				item_prob_flag = False

				# seq_cate_input, seq_short_input = self.model.m_itemNN(x_cate_batch, input_cate_subseq, mask_cate, mask_cate_seq, max_acticonNum_cate, max_subseqNum_cate, subseqLen_cate, seqLen_cate, x_batch, mask_batch, seqLen_batch, "test")
				# seq_short_input = seq_short_input.squeeze()
				
				# self.m_beam_size = 1
				for beam_index in range(self.m_beam_size):
					# st = datetime.datetime.now()
					
					cate_id_beam_batch = cate_id_beam[:, beam_index]
					# cate_id_beam_batch = y_cate
					# output_batch, cate_logit = self.model(x_cate_batch, input_cate_subseq, mask_cate, mask_cate_seq, max_acticonNum_cate, max_subseqNum_cate, subseqLen_cate, seqLen_cate, x_batch, cate_batch, mask_batch, seqLen_batch, cate_id_beam_batch, "test")
					cate_id_beam_batch = cate_id_beam_batch.reshape(-1, 1)

					seq_cate_input, seq_short_input = self.model.m_itemNN(x_cate_batch, input_cate_subseq, mask_cate, mask_cate_seq, max_acticonNum_cate, max_subseqNum_cate, subseqLen_cate, seqLen_cate, x_batch, mask_batch, seqLen_batch, cate_id_beam_batch, "test")

					# y_cate_index = (input_cate_subseq == cate_id_beam_batch).float()
					# y_cate_index = y_cate_index.unsqueeze(-1)

					# weighted_seq_cate_input = seq_cate_input*y_cate_index
					# weighted_seq_cate_input = torch.sum(weighted_seq_cate_input, dim=1)
					
					mixture_output = torch.cat((seq_cate_input, seq_short_input), dim=1)
					output_batch = self.model.fc(mixture_output)

					### sampled_logit_batch
					sampled_logit_batch, sampled_target_batch = self.model.m_ss(output_batch, y_batch, None, None, None, None, None, None)
					
					### batch_size*voc_size
					prob_batch = F.softmax(sampled_logit_batch, dim=-1)

					## batch_size*1
					cate_prob_batch = cate_prob_beam[:, beam_index]
					
					item_prob_batch = prob_batch*cate_prob_batch.reshape(-1, 1)

					# if not item_prob:
					if not item_prob_flag:
						item_prob_flag = True
						item_prob = item_prob_batch
					else:
						item_prob += item_prob_batch
			
				# et_beam = datetime.datetime.now()
				# duration = et_beam-st
				# print("beam duration", duration)

				### beam for itemNN
				
				# output_batch, cate_logit = self.model(x_cate_batch, input_cate_subseq, mask_cate, mask_cate_seq, max_acticonNum_cate, max_subseqNum_cate, subseqLen_cate, seqLen_cate, x_batch, cate_batch, mask_batch, seqLen_batch, "test")
				
				# loss_batch = self.loss_func(sampled_logit_batch, sampled_target_batch)

				cate_loss_batch = self.loss_func(logit_cate_short, y_cate)

				# mixture_loss_batch = loss_batch+cate_loss_batch

				# mixture_losses.append(mixture_loss_batch.item())

				# losses.append(loss_batch.item())
				cate_losses.append(cate_loss_batch.item())

				# logit_batch = F.linear(output_batch, self.model.m_ss.params.weight)
				# logit_batch = self.model.m_ss.params(output_batch)
				
				recall_batch, mrr_batch = evaluate(item_prob, y_batch, warm_start_mask, k=self.topk)

				weights.append( int( warm_start_mask.int().sum() ) )
				recalls.append(recall_batch)
				mrrs.append(mrr_batch)

				cate_recall_batch, cate_mrr_batch = evaluate(logit_cate_short, y_cate, warm_start_mask, k=self.topk)

				cate_weights.append(int( warm_start_mask.int().sum() ))
				cate_recalls.append(cate_recall_batch)
				cate_mrrs.append(cate_mrr_batch)

				total_test_num.append(y_batch.view(-1).size(0))

				# et = datetime.datetime.now()
				# duration = et-st
				# print("batch duration", duration)

			# et  = datetime.datetime.now()
			# duration = et-st
			# print("eval duration", duration)

		# mean_mixture_losses = np.mean(mixture_losses)
		mean_mixture_losses = 0.0

		# mean_losses = np.mean(losses)
		mean_losses = 0.0
		mean_recall = np.average(recalls, weights=weights)
		mean_mrr = np.average(mrrs, weights=weights)

		mean_cate_losses = np.mean(cate_losses)
		mean_cate_recall = np.average(cate_recalls, weights=cate_weights)
		mean_cate_mrr = np.average(cate_mrrs, weights=cate_weights)

		msg = "total_test_num"+str(np.sum(total_test_num))
		self.m_log.addOutput2IO(msg)

		return mean_mixture_losses, mean_losses, mean_recall, mean_mrr, mean_cate_losses, mean_cate_recall, mean_cate_mrr
