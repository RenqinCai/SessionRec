import argparse
import torch
# import lib
import numpy as np
import os
import datetime
from dataset import *
from loss import *
from model import *
from optimizer import *
from trainer import *
from torch.utils import data
import pickle
import sys
from dataset_time import *
from logger import *

import sys
sys.path.insert(0, '../PyTorch_GBW_LM')
sys.path.insert(0, '../PyTorch_GBW_LM/log_uniform')

from sparse_model import RNNModel, SampledSoftmax


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
parser.add_argument('--model_name', default='GRU4REC', type=str)
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

# Get the arguments
args = parser.parse_args(args=[])
args.cuda = torch.cuda.is_available()

args.data_folder = "../Data/tmall/100k_unknown_cate/"
args.data_action = "item_time.pickle"
args.data_cate = "cate_time.pickle"
args.data_time = "time_time.pickle"
args.data_name = "taobao"
args.valid_start_time = 1512172800
valid_start_time = args.valid_start_time
args.test_start_time = 1512259200
test_start_time = args.test_start_time

args.embedding_dim = 300
args.hidden_size = 300
args.lr = 0.0001
args.window_size = 20
args.test_observed = 5
args.n_epochs = 300
args.shared_embedding = 1
args.batch_size = 300
args.optimizer_type = "Adam"

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

n_epochs = args.n_epochs
time_sort = args.time_sort

window_size = args.window_size
shared_embedding = args.shared_embedding

if embedding_dim == -1:
    print("embedding dim not -1", embedding_dim)
    raise AssertionError()

observed_threshold = args.test_observed

data_action = args.data_folder+args.data_action
data_cate = args.data_folder+args.data_cate
data_time = args.data_folder+args.data_time

def read_data(action_file, cate_file, time_file, valid_start_time, test_start_time, observed_thresh, window_size):
	action_f = open(action_file, "rb")
	action_total = pickle.load(action_f)
	action_seq_num = len(action_total)
	print("action seq num", action_seq_num)

	time_f = open(time_file, "rb")
	time_total = pickle.load(time_f)
	time_seq_num = len(time_total)
	print("time seq num", time_seq_num)

	return action_total

data_obj = read_data(data_action, data_cate, data_time, valid_start_time, test_start_time, observed_threshold, window_size)
# train_data = data_obj.train_dataset

print("+"*10)
print("valid load")

if embedding_dim == -1:
	print("embedding dim not -1", embedding_dim)
	raise AssertionError()

# valid_data = data_obj.test_dataset

valid_action_list_num = len(data_obj)
print("valid_action_list_num", valid_action_list_num)

import gc
import datetime
# Returns length of longest common 
# substring of X[0..m-1] and Y[0..n-1] 
def LCSubStr(X, Y, m, n): 
	
	# Create a table to store lengths of 
	# longest common suffixes of substrings. 
	# Note that LCSuff[i][j] contains the 
	# length of longest common suffix of 
	# X[0...i-1] and Y[0...j-1]. The first 
	# row and first column entries have no 
	# logical meaning, they are used only 
	# for simplicity of the program. 
	
	# LCSuff is the table with zero 
	# value initially in each cell 
	LCSuff = [[0 for k in range(n+1)] for l in range(m+1)] 
	
	# To store the length of 
	# longest common substring 
	result = 0
	# Following steps to build 
	# LCSuff[m+1][n+1] in bottom up fashion 
	for i in range(m + 1): 
		for j in range(n + 1): 
			if (i == 0 or j == 0): 
				LCSuff[i][j] = 0
			elif (X[i-1] == Y[j-1]): 
				LCSuff[i][j] = LCSuff[i-1][j-1] + 1
				result = max(result, LCSuff[i][j]) 
			else: 
				LCSuff[i][j] = 0
	
	del LCSuff
	# gc.collect()
	return result 

max_common_len = 0
# q = 1000
valid_action_list_num = 1000
s_time = datetime.datetime.now()
for i in range(valid_action_list_num):
	if i%500 == 0:
		print("i", i)
	action_list_i = data_obj[i]
	for j in range(i+1, valid_action_list_num):

		action_list_j = data_obj[j]

		m = len(action_list_i) 
		n = len(action_list_j) 

		common_len = LCSubStr(action_list_i, action_list_j, m, n)
		if common_len > 4:
			print("++"*20)
			print(action_list_i)
			print(action_list_j)
			
		if common_len > max_common_len:

			max_common_len = common_len

e_time = datetime.datetime.now()
print("max_common_len", max_common_len)
print("duration", e_time-s_time)