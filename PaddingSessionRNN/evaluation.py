import numpy as np
import torch
import dataset
from metric import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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
			for input_x_batch, target_y_batch, x_len_batch, idx_batch in dataloader:
				input_x_batch = input_x_batch.to(self.device)
				target_y_batch = target_y_batch.to(self.device)
				warm_start_mask = (idx_batch>=self.warm_start).to(self.device)
                
				hidden = self.model.init_hidden()

				logit_batch, hidden = self.model(input_x_batch, hidden, x_len_batch)

				logit_sampled_batch = logit_batch[:, target_y_batch.view(-1)]
				loss_batch = self.loss_func(logit_sampled_batch, target_y_batch)

				losses.append(loss_batch.item())
				
				# self.topk = target_y_batch.size(0)
				# target_y_batch = torch.LongTensor(torch.arange(target_y_batch.size(0))).to(self.device)
				# recall_batch, mrr_batch = evaluate(logit_sampled_batch, target_y_batch, warm_start_mask, k=self.topk)
				# 
				recall_batch, mrr_batch = evaluate(logit_batch, target_y_batch, warm_start_mask, k=self.topk)

				weights.append( int( warm_start_mask.int().sum() ) )
				recalls.append(recall_batch)
				mrrs.append(mrr_batch)

				total_test_num.append(target_y_batch.view(-1).size(0))

		mean_losses = np.mean(losses)
		mean_recall = np.average(recalls, weights=weights)
		mean_mrr = np.average(mrrs, weights=weights)

		msg = "total_test_num"+str(np.sum(total_test_num))
		self.m_log.addOutput2IO(msg)


		fig = plt.figure()
		ax1 = fig.add_subplot(121, projection='3d')
		ax2 = fig.add_subplot(122, projection='3d')

		x1 = # Popularity bin for each time
		y1 = # Time bins for each popularity bin
		z1 = 0
		dx1 = 1
		dy1 = 0 # shows bars further back more clearly
		dz1 = # Performance for each combination of pupularity and time bins
		ax1.bar3d(x1, y1, z1, dx1, dy1, dz1, shade=True)
		ax1.set_title('Recall of Varying Item Popularities Over Time (RNN Xing)')
		ax1.set_xlabel('Popularity')
		ax1.set_ylabel('Time')
		ax1.set_zlabel('Recall@20')

		x2 = 
		y2 =
		z2 = 0
		dx2 = 1
		dy2 = 0
		dz2 =
		ax2.bar3d(x2, y2, z2, dx2, dy2, dz2, shade=True)
		ax2.set_title('MRR of Varying Item Popularities Over Time (RNN Xing)')
		ax2.set_xlabel('Popularity')
		ax2.set_ylabel('Time')
		ax2.set_zlabel('MRR@20')

		fig.savefig('sequential_bias.png')


		return mean_losses, mean_recall, mean_mrr
