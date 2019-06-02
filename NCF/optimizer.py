import torch.optim as optim

class Optimizer:
	def __init__(self, params, optimizer_type="Adagrad", lr=0.05, momentum=0, weight_decay=0, eps=1e-6):
		if optimizer_type == 'Adagrad':
			self.optimizer = optim.Adagrad(params, lr=lr, weight_decay=weight_decay)
		elif optimizer_type == 'Adadelta':
			self.optimizer = optim.Adadelta(params, lr=lr, weight_decay=weight_decay)

		elif optimizer_type == 'Adam':
			self.optimizer = optim.Adam(params, lr=lr, eps=eps, weight_decay=weight_decay)

		elif optimizer_type == 'SparseAdam':
			self.optimizer = optim.SparseAdam(params, lr=lr, eps=eps)
		elif optimizer_type == 'SGD':
			self.optimizer = optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
		else:
			raise NotImplementedError

	def zero_grad(self):
		self.optimizer.zero_grad()

	def step(self):
		self.optimizer.step()