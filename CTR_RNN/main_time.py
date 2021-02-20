"""
use time to cut sequences
command 
python main_time.py --data_folder ../Data/xing/ --train_data train_item.pickle --valid_data test_item.pickle --test_data test_item.pickle --data_name xing --embedding_dim 300 --hidden_size 300 --lr 0.005
"""
import argparse
import torch
from torch.utils import data
import torch.nn as nn
# import lib
import numpy as np
import os
import datetime
from loss import *
from network import *
# from network_parallel import *
from optimizer import *
from trainer import *
import pickle
import sys
from dataset_time import *
from dataset_time import _seq_corpus, _seq
# from data_time import *
from logger import *
import collections
from multiprocessing import Pool
import torch.distributed as dist
import torch.multiprocessing as mp

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
parser.add_argument("--is_eval", action='store_true')
parser.add_argument('--load_model', default=None,  type=str)
parser.add_argument('--checkpoint_dir', type=str, default='checkpoint')
parser.add_argument('--data_name', default=None, type=str)
parser.add_argument('--shared_embedding', default=None, type=int)
parser.add_argument('--patience', default=1000)
parser.add_argument('--negative_num', default=1000, type=int)
parser.add_argument('--valid_start_time', default=0, type=int)
parser.add_argument('--test_start_time', default=0, type=int)
parser.add_argument('--model_name', default="CTR_RNN", type=str)
parser.add_argument('--num_heads', default=4, type=int)
parser.add_argument('--decay_size', default=50, type=int)
parser.add_argument('--gpu_devices', type=int, nargs='+', default=None, help="")
parser.add_argument('--dist_url', default='tcp://127.0.0.1:10325', type=str, help='url used to set up distributed training')
parser.add_argument('--dist_backend', default='nccl', type=str, help='distributed backend')
parser.add_argument('--rank', default=-1, type=int,help='node rank for distributed training')
parser.add_argument('--world_size', default=-1, type=int,help='number of nodes for distributed training')
parser.add_argument('--multiprocessing_distributed', action='store_true', help='Use multi-processing distributed training to launch N processes per node, which has N GPUs. This is the fastest way to use PyTorch for either single node or multi node data parallel training')

# Get the arguments
args = parser.parse_args()
args.cuda = torch.cuda.is_available()
# args.cuda = False
# gpu_devices = ','.join([str(id) for id in args.gpu_devices])
# os.environ["CUDA_VISIBLE_DEVICES"] = gpu_devices

np.random.seed(args.seed)
torch.manual_seed(7)
random.seed(args.seed)

if args.cuda:
	print("gpu")
	torch.cuda.manual_seed(args.seed)
else:
	print("cpu")

def make_checkpoint_dir(log):
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
			msg = "{}={}".format(attr.upper(), value)
			log.addOutput2IO(msg)
			f.write("{}={}\n".format(attr.upper(), value))

	msg = "---------" + "-"*10
	log.addOutput2IO(msg)

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

def load_train_corpus_helper(train_corpus_file):
	
	train_corpus_f = open(train_corpus_file, "rb")
	train_seq_corpus_i = pickle.load(train_corpus_f)
	
	train_corpus_f.close()
	print("train corpus file", train_corpus_file)
	print(len(train_seq_corpus_i))
	# train_i_map = {}
	# train_i_map[0] = train_seq_corpus_i
	# return train_corpus_file
	a = train_seq_corpus_i

	return a

def load_train_corpus(args_model):
	train_seq_corpus = []
	pool_num = 8
	file_list = []
	st = datetime.datetime.now()
	for i in range(pool_num):
		train_corpus_file = args_model.data_folder+args_model.data_action+"_train_"+str(i)+".pickle"
		file_list.append(train_corpus_file)

	# print("file_list", file_list)
	results = map(load_train_corpus_helper, file_list)
	
	# pool = Pool(processes=pool_num)
	# for i in range(pool_num):
	# 	# print("args", file_list[i])
	# 	result_i = pool.apply(load_train_corpus_helper, args=(args_model, i, ))
	# 	train_seq_corpus.extend(result_i)

	# pool.close()
	# pool.join()

	# print("result", "*"*10)
	for result_i in results:
		train_seq_corpus.extend(result_i)

	et = datetime.datetime.now()
	print("train data load duration", et-st)
	# print(train_seq_corpus)
	# exit()
	return train_seq_corpus

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
	BPTT = args.bptt
	decay_size = args.decay_size
	num_heads = args.num_heads

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
	
	st = datetime.datetime.now()

	# train_seq_corpus = []
	# train_corpus_file = args.data_folder+args.data_action+"_"+str(0)+"_train_debug.pickle"
	# train_corpus_f = open(train_corpus_file, "rb")
	# train_seq_corpus_i = pickle.load(train_corpus_f)
	# train_seq_corpus.extend(train_seq_corpus_i)
	# train_corpus_f.close()

	# train_seq_corpus = load_train_corpus(args)
	et = datetime.datetime.now()
	print("train data load duration", et-st)

	st = datetime.datetime.now()
	valid_corpus_file = args.data_folder+args.data_action+"_valid.pickle"
	test_corpus_file = args.data_folder+args.data_action+"_test.pickle"

	valid_corpus_f = open(valid_corpus_file, "rb")
	test_corpus_f = open(test_corpus_file, "rb")

	valid_seq_corpus = pickle.load(valid_corpus_f)
	test_seq_corpus = pickle.load(test_corpus_f)
	
	et = datetime.datetime.now()
	print("valid test duration", et-st)

	# test_seq_corpus = valid_seq_corpus
		
	# train_data_loader = MYDATALOADER(train_seq_corpus, batch_size, "train")
	# valid_data_loader = MYDATALOADER(valid_seq_corpus.m_seq_list, batch_size, "valid")

	train_data_loader = MYDATALOADER(valid_seq_corpus.m_seq_list, batch_size, "train")
	valid_data_loader = train_data_loader

	test_data_loader = valid_data_loader
	
	valid_corpus_f.close()
	test_corpus_f.close()

	print("+"*10)
	print("valid load")

	# input_size = train_data_loader.m_words_num
	input_size = 2989
	output_size = input_size

	message = "input_size "+str(input_size)
	log.addOutput2IO(message)

	negative_num = args.negative_num

	message = "negative_num "+str(negative_num)
	log.addOutput2IO(message)

	if not args.is_eval:
		make_checkpoint_dir(log)

	if not args.is_eval:
		
		ss = SampledSoftmax(output_size, negative_num, embedding_dim, None)

		network = CTR_RNN(log, ss, input_size, hidden_size, output_size, decay_size=hidden_size,
		num_heads = num_heads,
		num_layers=num_layers,
		use_cuda=args.cuda,
		dropout_input=dropout_input,
		dropout_hidden=dropout_hidden,
		embedding_dim=embedding_dim,
		shared_embedding=shared_embedding)
							
		count_parameters(network)

		init_model(network)

		# print("args.dist_backend", args.dist_backend)
		# print("args.dist_url", args.dist_url)
		# print("args.world_size", args.world_size)
		# print("args.rank", args.rank)

		# dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank)

		# ngpus_per_node = torch.cuda.device_count()
		# # gpu_devices = args.gpu_devices
		# # print("gpu devices", gpu_devices)

		# if ngpus_per_node > 1:
		# 	print("ngpus_per_node", ngpus_per_node)
		# 	network = nn.DataParallel(network)
		# elif ngpus_per_node == 0:
		# 	print("no gpu !!!")
		# 	exit()

		multiGPU = False

		optimizer = Optimizer(network.parameters(),
		optimizer_type=optimizer_type, lr=lr,weight_decay=weight_decay, momentum=momentum,eps=eps)

		c_weights = None
		# print("c weights", c_weights)
		loss_function = LossFunction(c_weights=c_weights, loss_type=loss_type, use_cuda=args.cuda)

		trainer = Trainer(log, network,train_data=train_data_loader,eval_data=test_data_loader,
		optim=optimizer,
		use_cuda=args.cuda,
		multiGPU=multiGPU,
		loss_func=loss_function,
		topk = args.topk,
		input_size = input_size,
		sample_full_flag = "sample",
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