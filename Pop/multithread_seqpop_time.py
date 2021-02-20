"""
This is multi thread seq pop 
"""

from multiprocessing import Pool 
import pickle
import random
import numpy as np

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--topk', default=1, type=int)
# etc
parser.add_argument('--bptt', default=5, type=int)

parser.add_argument('--data_folder', default='../Data/movielen/1m/', type=str)
parser.add_argument('--data_action', default='item.pickle', type=str)
parser.add_argument('--data_cate', default='cate.pickle', type=str)
parser.add_argument('--data_time', default='time.pickle', type=str)

parser.add_argument("--is_eval", action='store_true')
parser.add_argument('--checkpoint_dir', type=str, default='checkpoint')
parser.add_argument('--valid_start_time', default=0, type=int)
parser.add_argument('--test_start_time', default=0, type=int)

# Get the arguments
args = parser.parse_args()

data_action = args.data_folder+args.data_action
data_cate = args.data_folder+args.data_cate
data_time = args.data_folder+args.data_time

action_f = open(data_action, "rb")
action_seq_arr_total = pickle.load(action_f)

cate_f = open(data_cate, "rb")
cate_seq_arr_total = pickle.load(cate_f)

time_f = open(data_time, "rb")
time_seq_arr_totoal = pickle.load(time_f)

action_seq_num = len(action_seq_arr_total)
print("action seq num", action_seq_num)

cate_seq_num = len(cate_seq_arr_total)
print("cate seq num", cate_seq_num)

time_seq_num = len(time_seq_arr_totoal)
print("time seq num", time_seq_num)

valid_start_time = args.valid_start_time
print("valid_start_time", valid_start_time)

test_start_time = args.test_start_time
print("test start time", test_start_time)

test_total_action_list = action_seq_arr_total
test_total_time_list = time_seq_arr_totoal

print("*"*10, "seq pop", "*"*10)
# user_num = len(test_total_action_list)

seq_action_num_threshold = 5

topk = args.topk

correct_num = 0
total_num = 0

recall_topk = 0.0
mrr_topk = 0.0

def seqPopWrapper(args):
    return seqPop(*args)

def seqPop(test_action_list, test_time_list):
    correct_num_pool = 0
    total_num_pool = 0
    mrr_pool = 0.0
    test_user_num = len(test_action_list)

    print("thread user num", test_user_num)
    for test_index in range(test_user_num):
        test_action_user_list = test_action_list[test_index]
        test_time_user_list = test_time_list[test_index]

        test_action_user_num = len(test_action_user_list)
        
        seq_item_map = {}
        for test_action_user_index in range(test_action_user_num):
            test_user_item = test_action_user_list[test_action_user_index]
            test_user_time = test_time_user_list[test_action_user_index]

            if test_user_time > valid_start_time and test_user_time <= test_start_time :
                
                # test_user_item = test_action_user_list[test_action_user_index]
                
                sorted_test_user_item_list = sorted(seq_item_map, key=seq_item_map.__getitem__, reverse=True)
                
                # if test_user_item in sorted_test_user_item_list[:topk]:
                #     correct_num_pool += 1.0

                topk_candidate_item_list = sorted_test_user_item_list[:topk]

                test_mrr = 0.0
                if test_user_item in topk_candidate_item_list:
                    correct_num_pool += 1.0
                    test_rank = list(topk_candidate_item_list).index(test_user_item)
                    test_mrr = 1.0/(test_rank+1.0)
                    print("test_mrr", test_mrr)
                    
                total_num_pool += 1.0
                mrr_pool += test_mrr
            
            if test_user_item not in seq_item_map:
                seq_item_map[test_user_item] = 0.0
                
            seq_item_map[test_user_item] += 1.0
        
    return correct_num_pool, total_num_pool, mrr_pool

pool_num = 20

results = []
args_list = [[] for i in range(pool_num)]

data_action_pool = [[] for i in range(pool_num)]
data_time_pool = [[] for i in range(pool_num)]

test_user_num = len(test_total_action_list)
for test_index in range(test_user_num):
   
    test_action_user_list = test_total_action_list[test_index]
    test_time_user_list = test_total_time_list[test_index]

    pool_index = int(test_index % pool_num)

    data_action_pool[pool_index].append(test_action_user_list)
    data_time_pool[pool_index].append(test_time_user_list)

for pool_index in range(pool_num):
    args_list[pool_index].append(data_action_pool[pool_index])
    args_list[pool_index].append(data_time_pool[pool_index])

pool_obj = Pool(pool_num)
results = pool_obj.map(seqPopWrapper, args_list)
pool_obj.close()
pool_obj.join()

correct_num = 0.0
total_num = 0.0
mrr_num = 0.0

for pool_index in range(pool_num):
    result_pool = results[pool_index]
    correct_num_pool = result_pool[0]
    total_num_pool = result_pool[1]
    mrr_pool = result_pool[2]

    correct_num += correct_num_pool
    total_num += total_num_pool
    mrr_num += mrr_pool

print("correct num", correct_num)
print("total num", total_num)
print("recall@{0:d}: {1:.4f}".format(topk, correct_num/total_num))
print("mrr@{0:d}: {1:.4f}".format(topk, mrr_num/total_num))


