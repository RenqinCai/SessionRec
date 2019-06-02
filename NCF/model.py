import torch
# from gmf import GMF
# from mlp import MLP
# from engine import Engine
# from utils import resume_checkpoint


class NeuMF(torch.nn.Module):
	def __init__(self, config):
		super(NeuMF, self).__init__()

		self.config = config
		self.num_users = config['num_users']
		self.num_items = config['num_items']
		self.latent_dim_mf = config['latent_dim_mf']
		self.latent_dim_mlp = config['latent_dim_mlp']

		self.embedding_user_mlp = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim_mlp)
		self.embedding_item_mlp = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim_mlp)

		self.embedding_user_mf = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim_mf)
		self.embedding_item_mf = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim_mf)

		self.fc_layers = torch.nn.ModuleList()
		for idx, (in_size, out_size) in enumerate(zip(config['layers'][:-1], config['layers'][1:])):
			self.fc_layers.append(torch.nn.Linear(in_size, out_size))

		self.affine_output = torch.nn.Linear(in_features=config['layers'][-1]+config['latent_dim_mf'], out_features=1)
		self.logistic = torch.nn.Sigmoid()

	def forward(self, user_indices, item_indices):
		user_embedding_mlp = self.embedding_user_mlp(user_indices)
		item_embedding_mlp = self.embedding_item_mlp(item_indices)

		mlp_vector = torch.cat([user_embedding_mlp, item_embedding_mlp], dim=-1)

		user_embedding_mf = self.embedding_user_mf(user_indices)
		item_embedding_mf = self.embedding_item_mf(item_indices)
		mf_vector = user_embedding_mf*item_embedding_mf

		for idx, _ in enumerate(range(len(self.fc_layers))):
			mlp_vector = self.fc_layers[idx](mlp_vector)
			mlp_vector = torch.nn.ReLU()(mlp_vector)

		final_vector = torch.cat([mlp_vector, mf_vector], dim=-1)
		logits = self.affine_output(final_vector)
		output = self.logistic(logits)

		return output

	def init_weights(self, sigma):
		if sigma is not None:
			for p in self.parameters():
				if sigma != -1 and sigma != -2:
					p.data.uniform_(-sigma, sigma)
				elif len(list(p.size())) > 1:
					sigma = np.sqrt(6.0/(p.size(0)+p.size(1)))
					if sigma == -1:
						p.data.uniform_(-sigma, sigma)
					else:
						p.data.uniform_(0, sigma)

	def load_pretrain_weights(self):
		config = self.config
		config['latent_dim'] = config['latent_dim_mlp']

		mlp_model = MLP(config)

		if config['use_cuda'] is True:
			mlp_model.cuda()

		resume_checkpoint(mlp_model, model_dir=config['pretrain_mlp'], device_id=config['device_id'])

		self.embedding_user_mlp.weight.data = mlp_model.embedding_user.weight.data
		self.embedding_item_mlp.weight.data = mlp_model.embedding_item.weight.data

		for idx in range(len(self.fc_layers)):
			self.fc_layers[idx] = mlp_model.fc_layers[idx].weight.data

		config['latent_dim'] = config['latent_dim_mf']
		gmf_model = GMF(config)

		if config['use_cuda'] is True:
			gmf_model.cuda()

		resume_checkpoint(gmf_model, model_dir=config['pretrain_mf'], device_id=config['device_id'])

		self.embedding_user_mf.weight.data = gmf_model.embedding_user.weight.data
		self.embedding_item_mf.weight.data = gmf_model.embedding_item.weight.data

		self.affine_output.weight.data = 0.5*torch.cat([mlp_model.affine_output.weight.data, gmf_model.affine_output.weight.data], dim=-1)
		self.affine_output.bias.data = 0.5*(mlp_model.affine_output.bias.data+gmf_model.affine_output.bias.data)

# class NeuMFNetwork(network):
# 	def __init__(self, config):
# 		self.model = NeuMF(config)

# 		if config['use_cuda'] is True:
# 			use_cuda(True, config['device_id'])
# 			self.model.cuda()

# 		super(NeuMFNetwork, self).__init__(config)
# 		print(self.model)

# 		if config['pretrain']:
# 			self.model.load_pretrain_weights()

# 		