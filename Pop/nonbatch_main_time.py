import pickle
import random
import numpy as np

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--topk', default=5, type=int)
# etc
parser.add_argument('--bptt', default=5, type=int)

parser.add_argument('--data_folder', default='../Data/movielen/1m/', type=str)
parser.add_argument('--data_action', default='item.pickle', type=str)
parser.add_argument('--data_cate', default='cate.pickle', type=str)
parser.add_argument('--data_time', default='time.pickle', type=str)

parser.add_argument("--is_eval", action='store_true')
parser.add_argument('--checkpoint_dir', type=str, default='checkpoint')
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

time_threshold = 1512000000
print("time_threshold", time_threshold)

train_total_action_list = []
test_total_action_list = []

for seq_index in range(action_seq_num):
	action_seq_arr = action_seq_arr_total[seq_index]
	time_seq_arr = time_seq_arr_totoal[seq_index]

	actionNum_seq = len(action_seq_arr)

	for action_index in range(actionNum_seq):
		item_cur = action_seq_arr[action_index]
		time_cur = time_seq_arr[action_index]

		if time_cur <= time_threshold:
			continue
		else:
			subseq_train = action_seq_arr[:action_index]
			train_total_action_list.append(subseq_train)

			subseq_test = action_seq_arr[action_index:]
			test_total_action_list.append(subseq_test)
			break

seq_action_num_threshold = 5

topk = 5

correct_num = 0
total_num = 0

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

test_user_num = len(test_total_action_list)
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

        