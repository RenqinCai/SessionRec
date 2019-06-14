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

		# def reset_hidden(hidden, mask):
		# 	if len(mask) != 0:
		# 		hidden[:, mask, :] = 0

		# 	return hidden

		with torch.no_grad():
			sess_hidden = self.model.init_hidden()
			user_hidden = self.model.init_hidden()
			total_test_num = []

			for x_input, y_target, mask_sess, mask_user, start_user_mask, idx_tensor in dataloader:
				x_input = x_input.to(self.device)
				y_target = y_target.to(self.device)
				warm_start_mask = (idx_tensor >= self.warm_start).to(self.device)

				# hidden = reset_hidden(hidden, mask)
				# hidden = reset_hidden(hidden, mask).detach()

				sess_hidden = sess_hidden.detach()
				user_hidden = user_hidden.detach()

				logit, sess_hidden, user_hidden = self.model(x_input, sess_hidden, user_hidden, mask_sess, mask_user)

				# print("preds", logit.size())
				logit_sampled = logit[:, y_target.view(-1)]
				loss = self.loss_func(logit_sampled)

				# logit = logit[np.where(start_user_mask)]
				# idx_target = idx_target[np.where(start_user_mask)]

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

