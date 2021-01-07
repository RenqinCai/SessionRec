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

def load_args(model_path):
    model_file = os.path.join(model_path, "model_best.pt")
    print("args file load", model_file)
    check_point = torch.load(model_file)
    args = check_point['args']

def load_model(network, model_path):
    print("reload model")
    model_file = os.path.join(model_path, "model_best.pt")
    print("model file", model_file)
    check_point = torch.load(model_file)

    network.load_state_dict(check_point['model'])

def count_parameters(model):
    parameter_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("parameter_num", parameter_num) 

def main():
    
    model_path = args.checkpoint_dir
    load_args(model_path)

    BPTT = args.bptt

    device = torch.device('cuda' if args.cuda else 'cpu')
    print("device", device)

    if args.embedding_dim == -1:
        raise AssertionError()

    data_name = args.data_name

    print("*"*10+"train load"+"*"*10)

    observed_threshold = args.test_observed

    data_action = args.data_folder+args.data_action
    data_cate = args.data_folder+args.data_cate
    data_time = args.data_folder+args.data_time
    
    valid_start_time = args.valid_start_time
    test_start_time = args.test_start_time

    st = datetime.datetime.now()
    data_obj = MYDATA(data_action, data_cate, data_time, valid_start_time, test_start_time, observed_threshold, args.window_size)
    et = datetime.datetime.now()
    print("load data duration ", et-st)

    train_data = data_obj.train_dataset
    valid_data = data_obj.test_dataset
    test_data = data_obj.test_dataset

    print("+"*10+"valid load"+"+"*10)

    input_size = data_obj.items()
    output_size = input_size

    negative_num = args.negative_num

    train_data_loader = MYDATALOADER(train_data, args.batch_size)
    valid_data_loader = MYDATALOADER(valid_data, args.batch_size)
    test_data_loader = MYDATALOADER(valid_data, args.batch_size)

    ss = SampledSoftmax(output_size, negative_num, args.embedding_dim, None)

    network = NETWORK(input_size, ss, args, device)
    load_model(network, model_path)

    ### eval
    loss_function = LossFunction(device, loss_type=args.loss_type)

    topk = args.topk
    eval = Evaluation(None, network, loss_function, device, topk, args.warm_start)

    # train_item_freq_dict = dict(collections.Counter(train_data.m_y_action))
    # print(len(train_item_freq_dict))
    # itemfreq_list = list(train_item_freq_dict.values())
    # print(len(itemfreq_list))
    # for item in train_item_freq_dict:
    # 	train_item_freq_dict[item] = train_item_freq_dict[item]/len(train_data.m_y_action)

    # train_itemfreq_file = "train_item_freq.txt"
    # f = open(train_itemfreq_file, "w")
    # for itemfreq in itemfreq_list:
    #     f.write(str(itemfreq))
    #     f.write("\n")
    # f.close()
    
    train_item_freq_dict, train_itemid_bucketid_dict, train_bucketid_itemidlist_dict = eval.set_bucket4item(train_data)

    # exit()
    print("--"*10+"eval train"+"--"*10)
    mean_loss, mean_recall, mean_mrr = eval.eval(train_data_loader, "train")
    msg = "train loss: {:.4f}, recall: {:.4f}, mrr: {:.4f}".format(mean_loss, mean_recall, mean_mrr)
    print(msg)

    eval.bias_eval(train_data_loader, train_itemid_bucketid_dict, "train")

    print("--"*10+"eval test"+"--"*10)
    mean_loss, mean_recall, mean_mrr = eval.eval(test_data_loader, "test")
    msg = "eval loss: {:.4f}, recall: {:.4f}, mrr: {:.4f}".format(mean_loss, mean_recall, mean_mrr)
    print(msg)

    # test_item_freq_dict = dict(collections.Counter(test_data.m_y_action))
    # print(len(test_item_freq_dict))
    # itemfreq_list = list(test_item_freq_dict.values())
    # print(len(itemfreq_list))
    # for item in test_item_freq_dict:
    # 	test_item_freq_dict[item] = test_item_freq_dict[item]/len(test_data.m_y_action)

    eval.bias_eval(test_data_loader, train_itemid_bucketid_dict, "test")
    

if __name__ == '__main__':
    main()