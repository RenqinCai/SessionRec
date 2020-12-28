import argparse
import torch

import numpy as np
import os
import datetime
from dataset import *
from loss import *
from model import *
from optimizer import *
from trainer import *
from torch.utils import data 

parser = argparse.ArgumentParser()
parser.add_argument('--hidden_size', default=50, type=int)
parser.add_argument('--num_layers', default=1, type=int)
parser.add_argument('--batch_size', default=100, type=int)
parser.add_argument('--dropout_input', default=0, type=float)

parser.add_argument('--optimizer_type', default='Adagrad', type=str)
parser.add_argument('--final_act', default='tanh', type=str)
parser.add_argument('--lr', default=0.5, type=float)
parser.add_argument('--weight_decay', default=0, type=float)
parser.add_argument('--momentum', default=0.1, type=float)
parser.add_argument('--eps', default=1e-6, type=float)
parser.add_argument('--layers', default=[16,8])

# parser.add_argument('--layers', default=[16,32,16,8])

parser.add_argument('--seed', type=int, default=7, help='Seed for random initialization')
parser.add_argument('--sigma', type=float, default=None, help='init weight -1: range [-sigma, sigma], -2: range[0, sigma]')
parser.add_argument('--embedding_dim', type=int, default=-1, help='using embedding')

parser.add_argument('--loss_type', default='BCE', type=str)
parser.add_argument('--topk', default=5, type=int)

parser.add_argument('--latent_dim_mf', default=8, type=int)
parser.add_argument('--latent_dim_mlp', default=8, type=int)
parser.add_argument('--num_negative', default=4, type=int)

parser.add_argument('--n_epochs', default=20, type=int)
parser.add_argument('--time_sort', default=False, type=bool)
parser.add_argument('--model_name', default='GMF', type=str)
parser.add_argument('--checkpoint_dir', type=str, default='checkpoint')

args = parser.parse_args()
args.cuda = torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(7)

if args.cuda:
	torch.cuda.manual_seed(args.seed)

neumf_config = {'alias': 'neural mf'} 

def make_checkpoint_dir():
	print("PARAMETER" + "-"*10)
	now = datetime.datetime.now()
	S = '{:02d}{:02d}{:02d}{:02d}'.format(now.month, now.day, now.hour, now.minute)
	save_dir = os.path.join(args.checkpoint_dir, S)
	if not os.path.exists(args.checkpoint_dir):
		os.mkdir(args.checkpoint_dir)

	if not os.path.exists(save_dir):
		os.mkdir(save_dir)

	args.checkpoint_dir = save_dir

	with open(os.path.join(args.checkpoint_dir, 'parameter.txt'), 'w') as f:
		for attr, value in sorted(args.__dict__.items()):
			print("{}={}".format(attr.upper(), value))
			f.write("{}={}\n".format(attr.upper(), value))

	print("---------" + "-"*10)
	

def main():
	### load data
	###

	data_file = "../Data/movielen/ratings.dat"

	num_neg = 100
	batch_size = args.batch_size

	train_data, test_data, user_pool, item_pool = preprocess(data_file, num_neg, batch_size)
	train_data = data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
	valid_data = test_data
	test_data = test_data
	
	make_checkpoint_dir()

	### new model
	### 
	neumf_config = {}
	neumf_config['num_users'] = len(user_pool)
	neumf_config['num_items'] = len(item_pool)

	neumf_config["num_epoch"] = args.n_epochs
	neumf_config['batch_size'] = args.batch_size
	neumf_config['optimizer'] = args.optimizer_type
	neumf_config['adam_lr'] = args.lr

	neumf_config['latent_dim_mf'] = args.latent_dim_mf
	neumf_config['latent_dim_mlp'] = args.latent_dim_mlp

	neumf_config['num_negative'] = args.num_negative
	neumf_config['layers'] = args.layers

	neumf_config["l2_regularization"] = args.momentum
	neumf_config['use_cuda'] = args.cuda

	neumf_config['sigma'] = args.sigma

	# neumf_config['topk'] = args.topk
	# neumf_config['device_id'] = 7
	# neumf_config['pretrain'] = False

	network = NeuMF(neumf_config)

	### init weight
	if neumf_config['use_cuda'] is True:
		network = network.cuda()

	print("network", network)

	network.init_weights(neumf_config['sigma'])
	### new loss function, trainer

	### train model

	optimizer = Optimizer(network.parameters(), optimizer_type=args.optimizer_type, lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum, eps=args.eps)

	print("loss function")

	loss_function = LossFunction(loss_type=args.loss_type, use_cuda=args.cuda)

	print("trainer")

	trainer = Trainer(network, train_data=train_data, eval_data=valid_data, optim=optimizer, use_cuda=args.cuda, loss_func=loss_function, topk=args.topk, args=args)

	trainer.train(0, args.n_epochs-1, batch_size)


if __name__ == '__main__':
	main()