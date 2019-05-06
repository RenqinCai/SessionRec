import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class LossFunction(nn.Module):
	def __init__(self, loss_type="TOP1", use_cuda=False):

		super(LossFunction, self).__init__()
		self.loss_type = loss_type
		self.use_cuda = use_cuda

		if loss_type == "BPR":
			self._loss_fn = BPRLoss()

	def forward(self, logit):
		return self._loss_fn(logit)


class BPRLoss(nn.Module):
	def __init__(self):
		super(BPRLoss, self).__init__()

	def forward(self, logit):

		diff = logit.diag().view(-1, 1).expand_as(logit) - logit

		loss = -torch.mean(F.logsigmoid(diff))

		return loss