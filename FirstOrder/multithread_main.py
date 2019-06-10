"""
first order multi-thread version
"""

from multiprocessing import Pool 
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

"""
get first order matrix
"""

seq_action_num_threshold = 1

first_order_mat = {}
# first_order_mat = np.zeros((item_distinct_num, item_distinct_num))

# for train_index in range(train_user_num):
#     train_action_user_list = train_total_action_list[train_index]
#     train_action_user_num = len(train_action_user_list)
    
#     for train_action_user_index in range(train_action_user_num-1):
#         train_user_item = train_action_user_list[train_action_user_index]
#         next_train_user_item = train_action_user_list[train_action_user_index+1]
        
#         train_item_id = item_id_map[train_user_item]
#         next_train_item_id = item_id_map[next_train_user_item]
        
#         first_order_mat[train_item_id, next_train_item_id] += 1.0


def firstOrderWrapper(args):
    return firstOrder(*args)

def firstOrder(train_action_list):
    train_user_num_pool = len(train_action_list)
    first_order_map_pool = {}

    # first_order_mat_pool = np.zeros((item_distinct_num, item_distinct_num))

    print("thread train user num", train_user_num_pool)
    for train_index in range(train_user_num_pool):
        train_action_user_list = train_action_list[train_index]
        train_action_user_num = len(train_action_user_list)
        
        for train_action_user_index in range(train_action_user_num-1):
            train_user_item = train_action_user_list[train_action_user_index]
            next_train_user_item = train_action_user_list[train_action_user_index+1]
            
            train_item_id = item_id_map[train_user_item]
            next_train_item_id = item_id_map[next_train_user_item]

            if train_item_id not in first_order_map_pool:
                first_order_map_pool[train_item_id] = {}

                if next_train_item_id not in first_order_map_pool[train_item_id]:
                    first_order_map_pool[train_item_id][next_train_item_id] = 0.0
            else:
                if next_train_item_id not in first_order_map_pool[train_item_id]:
                    first_order_map_pool[train_item_id][next_train_item_id] = 0.0
            first_order_map_pool[train_item_id][next_train_item_id] += 1.0
            
            # first_order_mat_pool[train_item_id, next_train_item_id] += 1.0
            
    return first_order_map_pool

pool_num = 10

results = []
args_list = [[] for i in range(pool_num)]

data_pool = [[] for i in range(pool_num)]
print("train_user num", train_user_num)
for train_index in range(train_user_num):
   
    train_action_user_list = train_total_action_list[train_index]

    pool_index = int(train_index % pool_num)
    data_pool[pool_index].append(train_action_user_list)

for pool_index in range(pool_num):
    args_list[pool_index].append(data_pool[pool_index])

pool_obj = Pool(pool_num)
results = pool_obj.map(firstOrderWrapper, args_list)
pool_obj.close()
pool_obj.join()

for pool_index in range(pool_num):
    result_pool = results[pool_index]
    first_order_map_pool = result_pool
    # print(first_order_mat_pool)
    for cur_item_id in first_order_map_pool:
        for next_item_id in first_order_map_pool[cur_item_id]:
            # first_order_mat[cur_item_id][next_item_id] += first_order_map_pool[cur_item_id][next_item_id]
            if cur_item_id not in first_order_mat:
                first_order_mat[cur_item_id] = {}

            if next_item_id not in first_order_mat[cur_item_id]:
                first_order_mat[cur_item_id][next_item_id] = 0.0
            
            first_order_mat[cur_item_id][next_item_id] += 1.0

for cur_item_id in range(20):
    for next_item_id in range(20):
        if cur_item_id not in first_order_mat:
            print("0", end=" ")
        else:
            if next_item_id not in first_order_mat[cur_item_id]:
                print("0", end=" ")
            else:
                print(first_order_mat[cur_item_id][next_item_id], end=" ")
    print("\n")

diagonal_max_num = 0
for cur_item_id in range(item_distinct_num):
    if cur_item_id not in first_order_mat:
        continue
    next_item_prob = first_order_mat[cur_item_id]
    
    sorted_item_list = sorted(next_item_prob, key=next_item_prob.__getitem__, reverse=True)
    next_item_id = sorted_item_list[0]
    # next_item_id = np.argmax(next_item_prob)

    if next_item_id == cur_item_id:
        diagonal_max_num += 1.0

print("diagonal num", diagonal_max_num, item_distinct_num)


"""
prediction
"""

topk = 5

recall_topk = 0.0
mrr_topk = 0.0

def firstOrderTestWrapper(args):
    return firstOrderTest(*args)

def firstOrderTest(test_action_list):
    test_user_num_pool = len(test_action_list)
    print("thread test user num", test_user_num_pool)
    correct_num_pool = 0.0
    total_num_pool = 0.0
    mrr_pool = 0.0
    
    for test_index in range(test_user_num_pool):
        test_action_user_list = test_action_list[test_index]
        test_action_user_num = len(test_action_user_list)

        new_test_action_user_list = []

        for test_action_user_index in range(test_action_user_num):
            test_user_item = test_action_user_list[test_action_user_index]
            
            if test_user_item not in item_id_map:
                continue
        
            new_test_action_user_list.append(test_user_item)
        
        test_action_list[test_index] = new_test_action_user_list

    for test_index in range(test_user_num_pool):
        test_action_user_list = test_action_list[test_index]
        test_action_user_num = len(test_action_user_list)
        
        for test_action_user_index in range(seq_action_num_threshold, test_action_user_num):
            test_user_item = test_action_user_list[test_action_user_index]
            last_user_item = test_action_user_list[test_action_user_index-1]
            
            test_user_item_id = item_id_map[test_user_item]
            last_user_item_id = item_id_map[last_user_item]
            
            if last_user_item_id not in first_order_mat:
                print("error")
                continue

            next_item_prob = first_order_mat[last_user_item_id]
    
            sorted_candidate_item_list = sorted(next_item_prob, key=next_item_prob.__getitem__, reverse=True)
            # next_item_id = sorted_item_list[0]

            # candidate_item_prob_list = first_order_mat[last_user_item_id, :]
            
            # sorted_candidate_item_list = np.argsort(np.array(-1*candidate_item_prob_list))
            topk_candidate_item_list = sorted_candidate_item_list[:topk]

            test_mrr = 0.0
            if test_user_item_id in topk_candidate_item_list:
                correct_num_pool += 1.0
                test_rank = list(topk_candidate_item_list).index(test_user_item_id)
                test_mrr = 1.0/(test_rank+1.0)
            
            total_num_pool += 1.0

            mrr_pool += test_mrr
            
    # print("correct num pool", correct_num_pool)
    # print("total num pool", total_num_pool)
    return correct_num_pool, total_num_pool, mrr_pool

results = []
args_list = [[] for i in range(pool_num)]

data_pool = [[] for i in range(pool_num)]
test_user_num = len(test_total_action_list)

for test_index in range(test_user_num):
   
    test_action_user_list = test_total_action_list[test_index]

    pool_index = int(test_index % pool_num)
    data_pool[pool_index].append(test_action_user_list)

for pool_index in range(pool_num):
    args_list[pool_index].append(data_pool[pool_index])

pool_obj = Pool(pool_num)
results = pool_obj.map(firstOrderTestWrapper, args_list)
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

    # print("correct num pool", correct_num_pool)

    correct_num += correct_num_pool
    total_num += total_num_pool
    mrr_num += mrr_pool

print("correct num", correct_num)
print("total num", total_num)
print("recall@{0:d}: {1:.4f}".format(topk, correct_num/total_num))
print("mrr@{0:d}: {1:.4f}".format(topk, mrr_num/total_num))
