"""
first order 
"""

import pickle
import random
import numpy as np
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--topk', default=5, type=int)
# etc
parser.add_argument('--bptt', default=5, type=int)

parser.add_argument('--data_folder', default='../Data/movielen/1m/', type=str)
parser.add_argument('--train_data', default='train_item.pickle', type=str)
parser.add_argument('--valid_data', default='test_item.pickle', type=str)
parser.add_argument('--test_data', default='test_item.pickle', type=str)
parser.add_argument("--is_eval", action='store_true')
parser.add_argument('--checkpoint_dir', type=str, default='checkpoint')
# Get the arguments
args = parser.parse_args()

print("*"*10, "first order", "*"*10)

train_file = args.data_folder+args.train_data
print("train file", train_file)
train_file_reader = open(train_file, "rb")
train_total_action_list = pickle.load(train_file_reader)

test_file = args.data_folder+args.test_data
print("test file", test_file)
test_file_reader = open(test_file, "rb")
test_total_action_list = pickle.load(test_file_reader)

item_id_map = {}
train_user_num = len(train_total_action_list)

for train_index in range(train_user_num):
    train_action_user_list = train_total_action_list[train_index]
    
    train_action_num = len(train_action_user_list)
    
    for train_action_index in range(train_action_num):
        train_item = train_action_user_list[train_action_index]
        
        if train_item not in item_id_map:
            item_new_id = len(item_id_map)
            item_id_map[train_item] = item_new_id

item_distinct_num = len(item_id_map)
print("item num", len(item_id_map))

item_map_pickle = args.data_folder+"train_item_map.pickle"
item_map_file = open(item_map_pickle, "wb")
pickle.dump(item_id_map, item_map_file)

train_user_num = len(train_total_action_list)

first_order_mat = np.zeros((item_distinct_num, item_distinct_num))

for train_index in range(train_user_num):
    train_action_user_list = train_total_action_list[train_index]
    train_action_user_num = len(train_action_user_list)
    
    for train_action_user_index in range(train_action_user_num-1):
        train_user_item = train_action_user_list[train_action_user_index]
        next_train_user_item = train_action_user_list[train_action_user_index+1]
        
        train_item_id = item_id_map[train_user_item]
        next_train_item_id = item_id_map[next_train_user_item]
        
        first_order_mat[train_item_id, next_train_item_id] += 1.0

topk = 5

correct_num = 0
total_num = 0

test_user_num = len(test_total_action_list)
seq_action_num_threshold = args.bptt

for test_index in range(test_user_num):
    test_action_user_list = test_total_action_list[test_index]
    test_action_user_num = len(test_action_user_list)
    
    for test_action_user_index in range(seq_action_num_threshold, test_action_user_num):
        test_user_item = test_action_user_list[test_action_user_index]
        last_user_item = test_action_user_list[test_action_user_index-1]
        
        test_user_item_id = item_id_map[test_user_item]
        last_user_item_id = item_id_map[last_user_item]
        
        candidate_item_prob_list = first_order_mat[last_user_item_id, :]
        
        sorted_candidate_item_list = np.argsort(np.array(-1*candidate_item_prob_list))
        
        if test_user_item_id in sorted_candidate_item_list[:topk]:
            correct_num += 1.0
        total_num += 1.0
        
print("correct num", correct_num)
print("total num", total_num)
print("hit@{0:d}: {1:.4f}".format(topk, correct_num*1.0/total_num))