"""
use time to cut sequences
command 
python main_time.py --data_folder ../Data/xing/ --train_data train_item.pickle --valid_data test_item.pickle --test_data test_item.pickle --data_name xing --embedding_dim 300 --hidden_size 300 --lr 0.005
"""
import argparse
import torch
# import lib
import numpy as np
import os
import datetime
from loss import *
from network import *
from optimizer import *
from trainer import *
from torch.utils import data
import pickle
import sys
from dataset_time import *
# from data_time import *
from logger import *
import collections

import sys
sys.path.insert(0, '../PyTorch_GBW_LM')
sys.path.insert(0, '../PyTorch_GBW_LM/log_uniform')

from sampledSoftmax import *

parser = argparse.ArgumentParser()
parser.add_argument('--hidden_size', default=50, type=int)
parser.add_argument('--num_layers', default=1, type=int)
parser.add_argument('--batch_size', default=100, type=int)
parser.add_argument('--dropout_input', default=0, type=float)
parser.add_argument('--dropout_hidden', default=.2, type=float)

# parse the optimizer arguments
parser.add_argument('--optimizer_type', default='Adagrad', type=str)
parser.add_argument('--final_act', default='tanh', type=str)
parser.add_argument('--lr', default=.05, type=float)
parser.add_argument('--weight_decay', default=0.0, type=float)
parser.add_argument('--momentum', default=0.1, type=float)
parser.add_argument('--eps', default=1e-6, type=float)

parser.add_argument("-seed", type=int, default=7,
					 help="Seed for random initialization")
parser.add_argument("-sigma", type=float, default=None,
					 help="init weight -1: range [-sigma, sigma], -2: range [0, sigma]")
parser.add_argument("--embedding_dim", type=int, default=-1,
					 help="using embedding")
# parse the loss type
parser.add_argument('--loss_type', default='TOP1', type=str)
# parser.add_argument('--loss_type', default='BPR', type=str)
parser.add_argument('--topk', default=5, type=int)
# etc
parser.add_argument('--bptt', default=1, type=int)
parser.add_argument('--test_observed', default=5, type=int)
parser.add_argument('--window_size', default=30, type=int)
parser.add_argument('--warm_start', default=5, type=int)

parser.add_argument('--n_epochs', default=20, type=int)
parser.add_argument('--time_sort', default=False, type=bool)
parser.add_argument('--save_dir', default='models', type=str)
parser.add_argument('--data_folder', default='../Data/movielen/1m/', type=str)
parser.add_argument('--data_action', default='item.pickle', type=str)
parser.add_argument('--data_cate', default='cate.pickle', type=str)
parser.add_argument('--data_time', default='time.pickle', type=str)
parser.add_argument("--is_eval", action='store_true')
parser.add_argument('--load_model', default=None,  type=str)
parser.add_argument('--checkpoint_dir', type=str, default='checkpoint')
parser.add_argument('--data_name', default=None, type=str)
parser.add_argument('--shared_embedding', default=None, type=int)
parser.add_argument('--patience', default=1000)
parser.add_argument('--negative_num', default=1000, type=int)
parser.add_argument('--valid_start_time', default=0, type=int)
parser.add_argument('--test_start_time', default=0, type=int)
parser.add_argument('--model_name', default="samplePaddingSessionRNN", type=str)

# Get the arguments
args = parser.parse_args()
args.cuda = torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(7)
random.seed(args.seed)

if args.cuda:
	torch.cuda.manual_seed(args.seed)

def make_checkpoint_dir():
	print("PARAMETER" + "-"*10)
	now = datetime.datetime.now()
	S = '{:02d}{:02d}{:02d}{:02d}'.format(now.month, now.day, now.hour, now.minute)
	checkpoint_dir = "../log/"+args.model_name+"/"+args.checkpoint_dir
	args.checkpoint_dir = checkpoint_dir
	save_dir = os.path.join(args.checkpoint_dir, S)

	if not os.path.exists("../log"):
		os.mkdir("../log")
	
	if not os.path.exists("../log/"+args.model_name):
		os.mkdir("../log/"+args.model_name)

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

def init_model(model):
	if args.sigma is not None:
		for p in model.parameters():
			if args.sigma != -1 and args.sigma != -2:
				sigma = args.sigma
				p.data.uniform_(-sigma, sigma)
			elif len(list(p.size())) > 1:
				sigma = np.sqrt(6.0 / (p.size(0) + p.size(1)))
				if args.sigma == -1:
					p.data.uniform_(-sigma, sigma)
				else:
					p.data.uniform_(0, sigma)

def count_parameters(model):
	parameter_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
	print("parameter_num", parameter_num) 

def main():

	hidden_size = args.hidden_size
	num_layers = args.num_layers
	batch_size = args.batch_size
	dropout_input = args.dropout_input
	dropout_hidden = args.dropout_hidden
	embedding_dim = args.embedding_dim
	final_act = args.final_act
	loss_type = args.loss_type
	topk = args.topk
	optimizer_type = args.optimizer_type
	lr = args.lr
	weight_decay = args.weight_decay
	momentum = args.momentum
	eps = args.eps

	n_epochs = args.n_epochs
	time_sort = args.time_sort

	window_size = args.window_size
	shared_embedding = args.shared_embedding

	log = Logger()
	log.addIOWriter(args)

	msg = "main_time.py "
	msg += "shared_embedding"+str(shared_embedding)
	log.addOutput2IO(msg)

	if embedding_dim == -1:
		msg = "embedding dim not -1 "+str(embedding_dim)
		log.addOutput2IO(msg)
		raise AssertionError()

	data_name = args.data_name

	print("*"*10)
	print("train load")

	observed_threshold = args.test_observed

	data_action = args.data_folder+args.data_action
	data_cate = args.data_folder+args.data_cate
	data_time = args.data_folder+args.data_time
	
	valid_start_time = args.valid_start_time
	test_start_time = args.test_start_time

	st = datetime.datetime.now()
	data_obj = MYDATA(data_action, data_cate, data_time, valid_start_time, test_start_time, observed_threshold, window_size)
	et = datetime.datetime.now()
	print("load data duration ", et-st)

	train_data = data_obj.train_dataset
	valid_data = data_obj.test_dataset
	test_data = data_obj.test_dataset

	print("+"*10)
	print("valid load")

	input_size = data_obj.items()
	output_size = input_size

	message = "input_size "+str(input_size)
	log.addOutput2IO(message)

	negative_num = args.negative_num

	message = "negative_num "+str(negative_num)
	log.addOutput2IO(message)

	train_data_loader = MYDATALOADER(train_data, batch_size)
	valid_data_loader = MYDATALOADER(valid_data, batch_size)
	test_data_loader = MYDATALOADER(valid_data, batch_size)

	if not args.is_eval:
		make_checkpoint_dir()

	if not args.is_eval:
		
		ss = SampledSoftmax(output_size, negative_num, embedding_dim, None)

		network = GRU4REC(log, ss, input_size, hidden_size, output_size,
							final_act=final_act,
							num_layers=num_layers,
							use_cuda=args.cuda,
							dropout_input=dropout_input,
							dropout_hidden=dropout_hidden,
							embedding_dim=embedding_dim,
							shared_embedding=shared_embedding
							)

		# init weight
		# See Balazs Hihasi(ICLR 2016), pg.7
		
		count_parameters(network)

		init_model(network)

		optimizer = Optimizer(network.parameters(),
								  optimizer_type=optimizer_type,
								  lr=lr,
								  weight_decay=weight_decay,
								  momentum=momentum,
								  eps=eps)

		# c_weight_map = dict(collections.Counter(train_data.m_y_action))
		# c_weights = [1.0 for i in range(output_size)]
		# for c_i in range(1, output_size):
		# 	c_weights[c_i] = len(train_data.m_y_action)/c_weight_map[c_i]
		# 	# np.array([1.0 for i in range(output_size)])
		# c_weights = np.log(np.array(c_weights))
		# c_weights[0] = 0.0
		# c_weights[1] = 0.5
		c_weights = np.array([1.0 for i in range(output_size)])
		print("c weights", c_weights)
		loss_function = LossFunction(c_weights=c_weights, loss_type=loss_type, use_cuda=args.cuda)

		trainer = Trainer(log, network,
							  train_data=train_data_loader,
							  eval_data=test_data_loader,
							  optim=optimizer,
							  use_cuda=args.cuda,
							  loss_func=loss_function,
							  topk = args.topk,
							  sample_full_flag = "full",
							  input_size = input_size,
							  args=args)

		trainer.train(0, n_epochs - 1, batch_size)

	else:
		if args.load_model is not None:
			print("Loading pre trained model from {}".format(args.load_model))
			checkpoint = torch.load(args.load_model)
			model = checkpoint["model"]
			model.gru.flatten_parameters()
			optim = checkpoint["optim"]
			loss_function = LossFunction(loss_type=loss_type, use_cuda=args.cuda)
			evaluation = Evaluation(model, loss_function, use_cuda=args.cuda)
			loss, recall, mrr = evaluation.eval(valid_data)
			print("Final result: recall = {:.2f}, mrr = {:.2f}".format(recall, mrr))
		else:
			print("Pre trained model is None!")


if __name__ == '__main__':
	main()