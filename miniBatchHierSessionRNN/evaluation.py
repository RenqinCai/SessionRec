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

		# def reset_hidden(hidden, mask):
		# 	if len(mask) != 0:
		# 		hidden[:, mask, :] = 0

		# 	return hidden

		with torch.no_grad():
			sess_hidden = self.model.init_hidden()
			user_hidden = self.model.init_hidden()
			total_test_num = []

			for idx_input, idx_target, mask_sess, mask_user, start_user_mask in dataloader:
				idx_input = idx_input.to(self.device)
				idx_target = idx_target.to(self.device)

				# hidden = reset_hidden(hidden, mask)
				# hidden = reset_hidden(hidden, mask).detach()

				sess_hidden = sess_hidden.detach()
				user_hidden = user_hidden.detach()

				logit, sess_hidden, user_hidden = self.model(idx_input, sess_hidden, user_hidden, mask_sess, mask_user)

				# print("preds", logit.size())
				logit_sampled = logit[:, idx_target.view(-1)]
				loss = self.loss_func(logit_sampled)

				# logit = logit[np.where(start_user_mask)]
				# idx_target = idx_target[np.where(start_user_mask)]

				recall, mrr = evaluate(logit, idx_target, k=self.topk)
				mrr = mrr.item()

				eval_iter += 1

				total_test_num.append(idx_target.view(-1).size(0))

				losses.append(loss.item())
				recalls.append(recall)
				mrrs.append(mrr)

		mean_losses = np.mean(losses)
		mean_recall = np.mean(recalls)
		mean_mrr = np.mean(mrrs)
		print("total_test_num", np.sum(total_test_num))

		return mean_losses, mean_recall, mean_mrr

