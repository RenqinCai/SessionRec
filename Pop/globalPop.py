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

train_file = args.data_folder+args.train_data
print("train file", train_file)
train_file_reader = open(train_file, "rb")
train_total_action_list = pickle.load(train_file_reader)

test_file = args.data_folder+args.test_data
print("test file", test_file)
test_file_reader = open(test_file, "rb")
test_total_action_list = pickle.load(test_file_reader)

train_user_num = len(train_total_action_list)

print("*"*10, "seq pop", "*"*10)
test_user_num = len(test_total_action_list)

seq_action_num_threshold = 1

topk = args.topk

correct_num = 0
total_num = 0

# for test_index in range(test_user_num):
#     test_action_user_list = test_total_action_list[test_index]
#     test_action_user_num = len(test_action_user_list)
    
#     seq_item_map = {}
#     for test_action_user_index in range(seq_action_num_threshold):
#         test_user_item = test_action_user_list[test_action_user_index]
        
#         if test_user_item not in seq_item_map:
#             seq_item_map[test_user_item] = 0.0
            
#         seq_item_map[test_user_item] += 1.0
        
#     for test_action_user_index in range(seq_action_num_threshold, test_action_user_num):
#         test_user_item = test_action_user_list[test_action_user_index]
        
#         sorted_test_user_item_list = sorted(seq_item_map, key=seq_item_map.__getitem__, reverse=True)
        
#         if test_user_item in sorted_test_user_item_list[:topk]:
#             correct_num += 1.0
            
#         total_num += 1.0
        
#         if test_user_item not in seq_item_map:
#             seq_item_map[test_user_item] = 0.0
            
#         seq_item_map[test_user_item] += 1.0
        
# print("correct num", correct_num)
# print("total num", total_num)
# print("hit@{0:d}: {1:.4f}".format(topk, correct_num/total_num))

print("*"*10, "global pop", "*"*10)

train_user_num = len(train_total_action_list)
item_freq_map = {}

item_id_map = {}
id_item_map = {}

for train_index in range(train_user_num):
    train_action_user_list = train_total_action_list[train_index]
    train_action_user_num = len(train_action_user_list)
    
    for train_action_user_index in range(train_action_user_num):
        train_user_item = train_action_user_list[train_action_user_index]
        
        if train_user_item not in item_freq_map:
            item_freq_map[train_user_item] = 0.0

        if train_user_item not in item_id_map:
            item_id = len(item_id_map)
            item_id_map[train_user_item] = item_id
            id_item_map[item_id] = train_user_item
            
        item_freq_map[train_user_item] += 1.0


topk = 5

correct_num = 0
total_num = 0
mrr_num = 0

sorted_item_list = sorted(item_freq_map, key=item_freq_map.__getitem__, reverse=True)
print("top k items", sorted_item_list[:topk])

for i in sorted_item_list[:topk]:
    print(item_id_map[i], item_freq_map[i])

for test_index in range(test_user_num):
    test_action_user_list = test_total_action_list[test_index]
    test_action_user_num = len(test_action_user_list)
    
    for test_action_user_index in range(seq_action_num_threshold, test_action_user_num):
        test_mrr = 0.0
        test_user_item = test_action_user_list[test_action_user_index]
        topk_candidate_item_list = sorted_item_list[:topk]

        if test_user_item in topk_candidate_item_list:
            correct_num += 1.0
            test_rank = list(topk_candidate_item_list).index(test_user_item)
            test_mrr = 1.0/(test_rank+1.0)

        total_num += 1.0
        mrr_num += test_mrr

print("correct num", correct_num)
print("total num", total_num)
print("hit@{0:d}: {1:.4f}".format(topk, correct_num/total_num))
print("mrr@{0:d}: {1:.4f}".format(topk, mrr_num/total_num))

        
