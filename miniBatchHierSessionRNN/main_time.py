import argparse
import torch
# import lib
import numpy as np
import os
import datetime
from loss import *
from model import *
from optimizer import *
from trainer import *
from dataset_time import *

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
parser.add_argument('--weight_decay', default=0, type=float)
parser.add_argument('--momentum', default=0.1, type=float)
parser.add_argument('--eps', default=1e-6, type=float)
parser.add_argument('--warm_start', default=5, type=int)

parser.add_argument("-seed", type=int, default=7,
					 help="Seed for random initialization")
parser.add_argument("-sigma", type=float, default=None,
					 help="init weight -1: range [-sigma, sigma], -2: range [0, sigma]")
parser.add_argument("--embedding_dim", type=int, default=-1,
					 help="using embedding")
parser.add_argument('--shared_embedding', default=None, type=bool)

# parse the loss type
parser.add_argument('--loss_type', default='TOP1', type=str)
# parser.add_argument('--loss_type', default='BPR', type=str)
parser.add_argument('--topk', default=5, type=int)
# etc
parser.add_argument('--bptt', default=1, type=int)

parser.add_argument('--n_epochs', default=20, type=int)
parser.add_argument('--time_sort', default=False, type=bool)
parser.add_argument('--model_name', default='HierRNN', type=str)
parser.add_argument('--save_dir', default='models', type=str)
parser.add_argument('--data_folder', default='../Data/movielen/1m/', type=str)
parser.add_argument('--data_action', default='item.pickle', type=str)
parser.add_argument('--data_cate', default='cate.pickle', type=str)
parser.add_argument('--data_time', default='time.pickle', type=str)
parser.add_argument("--is_eval", action='store_true')
parser.add_argument('--load_model', default=None,  type=str)
parser.add_argument('--checkpoint_dir', type=str, default='checkpoint')
parser.add_argument('--data_name', default=None, type=str)
parser.add_argument('--valid_start_time', default=0, type=int)
parser.add_argument('--test_start_time', default=0, type=int)

# Get the arguments
args = parser.parse_args()
args.cuda = torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(7)

if args.cuda:
	torch.cuda.manual_seed(args.seed)

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
	myhost = os.uname()[1]
	file_time = datetime.datetime.now().strftime('%H_%M_%d_%m')

	output_file = myhost+"_"+file_time
	
	output_file = output_file +"_"+ str(args.hidden_size)+"_"+str(args.batch_size)+"_"+str(args.embedding_dim)+"_"+str(args.optimizer_type)+"_"+str(args.lr)+"_"+"_"+str(args.shared_embedding)
	output_file = output_file+"_"+str(args.data_name)
	output_f = open(output_file, "w")

	train_data_action = args.data_folder+"train_"+args.data_action
	valid_data_action = args.data_folder+"test_"+args.data_action
	test_data_action = args.data_folder+"test_"+args.data_action

	print("Loading train data from {}".format(train_data_action))
	print("Loading valid data from {}".format(valid_data_action))
	print("Loading test data from {}\n".format(test_data_action))

	output_f.write("Loading train data from {}".format(train_data_action))
	output_f.write("Loading valid data from {}".format(valid_data_action))
	output_f.write("Loading test data from {}".format(test_data_action))
	output_f.flush()

	data_name = args.data_name

	data_action = args.data_folder+args.data_action
	data_cate = args.data_folder+args.data_cate
	data_time = args.data_folder+args.data_time

	valid_start_time = args.valid_start_time
	test_start_time = args.test_start_time
	
	mydata_obj = MYDATA(data_action, data_cate, data_time, valid_start_time, test_start_time)

	train_data = mydata_obj.train_dataset
	valid_data = mydata_obj.test_dataset
	test_data = mydata_obj.test_dataset

	train_data.segment2Session()
	valid_data.segment2Session()
	# test_data.segment2Session()

	# train_data = dataset.Dataset(train_data_action, train_data_time, data_name)
	# valid_data = dataset.Dataset(valid_data_action, valid_data_time, data_name, itemmap=train_data.m_item_map)
	# test_data = dataset.Dataset(test_data_action, test_data_time, data_name)

	if not args.is_eval:
		make_checkpoint_dir()

	input_size = mydata_obj.items()

	print("input_size", input_size)

	hidden_size = args.hidden_size
	num_layers = args.num_layers
	output_size = input_size

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

	n_epochs = args.n_epochs
	time_sort = args.time_sort

	train_data_loader = DataLoader(train_data, batch_size)
	valid_data_loader = DataLoader(valid_data, batch_size) 
	
	# train_data_loader = dataset.DataLoader(train_data, BPTT, batch_size)
	# BPTT_valid = 1
	# valid_data_loader = dataset.DataLoader(valid_data, BPTT_valid, batch_size)

	sess_hidden_size = hidden_size
	user_hidden_size = hidden_size

	user_output_size = output_size

	if not args.is_eval:
		model = HierGRU4REC(input_size, output_size, sess_hidden_size, user_hidden_size, user_output_size,
							final_act=final_act,
							sess_num_layers=num_layers,
							user_num_layers=num_layers,
							use_cuda=args.cuda,
							batch_size=batch_size,
							dropout_input=dropout_input,
							dropout_hidden=dropout_hidden,
							embedding_dim=embedding_dim
							)

		# init weight
		# See Balazs Hihasi(ICLR 2016), pg.7
		
		count_parameters(model)

		init_model(model)

		optimizer = Optimizer(model.parameters(),
								  optimizer_type=optimizer_type,
								  lr=lr,
								  weight_decay=weight_decay,
								  momentum=momentum,
								  eps=eps)

		loss_function = LossFunction(loss_type=loss_type, use_cuda=args.cuda)

		trainer = Trainer(model,
							  train_data=train_data_loader,
							  eval_data=valid_data_loader,
							  optim=optimizer,
							  use_cuda=args.cuda,
							  loss_func=loss_function,
							  topk = args.topk,
							  args=args)

		trainer.train(0, n_epochs - 1, batch_size, output_f)
		output_f.close()
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
