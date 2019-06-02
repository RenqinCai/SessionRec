import numpy as np
import torch
import dataset
from metric import *

from torch.utils import data

class Evaluation(object):
	def __init__(self, model, loss_func, use_cuda, k=20):
		self.model = model
		self.loss_func = loss_func

		self.topk = k 
		self.device=torch.device('cuda' if use_cuda else 'cpu')


	def eval(self, eval_data, batch_size):
		self.model.eval()

		losses = []
		recalls = []
		mrrs = []

		with torch.no_grad():
			# hidden = self.model.init_hidden()

			eval_dataLoader = data.DataLoader(eval_data)
			for user, item, target in eval_dataLoader:
				user = user.to(self.device)
				item = item.to(self.device)

				target = target.to(self.device).float()

				target_pred = self.model(user, item)

				loss = self.loss_func(target_pred, target)

				recall, mrr = evaluate(target_pred, target, self.topk)

				losses.append(loss.item())

				recalls.append(recall)

				mrrs.append(mrr)

		mean_loss = np.mean(losses)
		mean_recall = np.mean(recalls)
		mean_mrr = np.mean(mrrs)

		return mean_loss, mean_recall, mean_mrr