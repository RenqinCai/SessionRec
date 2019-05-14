import torch
import torch.nn as nn
from torch.autograd import Variable

import torch.nn.functional as F

class LossFunction(nn.Module):
	def __init__(self, loss_type='TOP1', use_cuda=False):
		super(LossFunction, self).__init__()

		self.loss_type = loss_type
		self.use_cuda = use_cuda

		if loss_type == 'BCE':
			self._loss_fn = BCELoss()

		else:
			raise NotImplementedError

	def forward(self, pred, target):
		return self._loss_fn(pred, target)

class BCELoss(nn.Module):
	def __init__(self):
		super(BCELoss, self).__init__()

		self.loss_val = torch.nn.BCELoss()

	def forward(self, pred, target):
		# print("pred", pred.view(-1))
		# print("target", target)
		loss = self.loss_val(pred.view(-1), target)

		return loss