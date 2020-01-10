"""
speed up the batchifying
use bucket to group sequences
"""

import pandas as pd
import numpy as np
import torch
import datetime
import pickle
import random
# import sys
import multiprocessing

class _seq_corpus: 
    def __init__(self):
        self.m_seq_list = []
        self.m_seq_num = 0
        
    def f_add_seq(self, seq_obj):
        self.m_seq_list.append(seq_obj)
        self.m_seq_num += 1
        
    def f_get_seq_num(self):
        return self.m_seq_num
        
class _seq:
    def __init__(self):
        self.m_seq_id = -1
        self.m_target = -1
        self.m_item_list = []
        self.m_time_list = []
        
        self.m_neigh_diff_item_list = []
        self.m_neigh_diff_time_list = []
        
        self.m_neigh_common_item_list = [] ###[[]]
        self.m_neigh_common_time_list = [] ###[[]]
        
    def f_set_self_list(self, item_list, time_list):
        self.m_item_list = item_list
        self.m_time_list = time_list
        
    def f_set_neigh_diff_list(self, item_list, time_list):
        self.m_neigh_diff_item_list = item_list
        self.m_neigh_diff_time_list = time_list
        
    def f_set_neigh_common_list(self, neigh_common_item_list, neigh_common_time_list):
        self.m_neigh_common_item_list = neigh_common_item_list
        self.m_neigh_common_time_list = neigh_common_time_list
    
    def f_set_target(self, target):
        self.m_target = target

class MYDATASET(object):
    def __init__(self, seq_corpus, batch_size, train_valid_test_flag):
        self.m_seq_corpus = seq_corpus
        self.m_batch_size = batch_size

        self.m_seq_num = len(self.m_seq_corpus.m_seq_list)
        self.m_batch_num = self.m_seq_num//batch_size

        self.m_words_num = 0

        start_time = datetime.datetime.now()
        self.f_process_time()
        end_time = datetime.datetime.now()
        print("process duration", end_time-start_time)

        self.m_bucket_list = []

        if train_valid_test_flag == "train":
            self.f_count_vocabulary()

    def f_get_voc_size(self):
        return self.m_words_num

    def f_process_time(self):
        for seq_index in range(self.m_seq_num):
            seq_obj = self.m_seq_corpus.m_seq_list[seq_index]

            new_neigh_common_time_list = []
            for common_time_list in seq_obj.m_neigh_common_time_list:
                new_common_time_list = []
                for (time_1, time_2) in common_time_list:
                    time_diff = time_1+time_2
                    new_common_time_list.append(time_diff)

                new_neigh_common_time_list.append(new_common_time_list)

            seq_obj.m_neigh_common_time_list = new_neigh_common_time_list

    def f_count_vocabulary(self):
        vocab_map = {}
        vocab_map['<PAD>'] = 0
        for seq_index in range(self.m_seq_num):
            seq_obj = self.m_seq_corpus.m_seq_list[seq_index]

            for item in seq_obj.m_item_list:
                if item not in vocab_map:
                    item_id = len(vocab_map)
                    vocab_map[item] = item_id
        
        self.m_words_num = len(vocab_map)

    def f_set_bucket(self):

        batch_size = self.m_batch_size
        self_seq_len_list = []
        seq_id_list = []
        for seq_index in range(self.m_seq_num):
            seq_obj = self.m_seq_corpus.m_seq_list[seq_index]
            self_seq_len = len(seq_obj.m_item_list)

            self_seq_len_list.append(self_seq_len)
            seq_id_list.append(seq_index)

        sorted_data = sorted(zip(self_seq_len_list, seq_id_list), reverse=True)
        sorted_self_seq_len_list, sorted_seq_id_list = zip(*sorted_data)

        for bucket_i in range(self.m_batch_num):
            bucket_obj = BUCKET(bucket_i)
            bucket_seq_obj_list = []

            bucket_seq_id_list = sorted_seq_id_list[bucket_i*batch_size: (bucket_i+1)*batch_size]
            for seq_id in bucket_seq_id_list:
                seq_obj = self.m_seq_corpus.m_seq_list[seq_id]
                bucket_seq_obj_list.append(seq_obj)

            bucket_obj.f_fill_bucket(bucket_seq_obj_list)

            self.m_bucket_list.append(bucket_obj)

        return self.m_bucket_list

class BUCKET(object):
    def __init__(self, bucket_id):
        self.m_bucket_id = bucket_id
        # self.m_seq_list = None
        
        self.m_self_src = []
        self.m_position_self_src = []
        self.m_common_src = []
        self.m_common_time = []
        self.m_friend_diff_src = []
        self.m_friend_num = []
        self.m_y = []
        self.m_y_id = []

    def f_fill_bucket(self, seq_obj_list):
        # self.m_seq_list = seq_obj_list

        for seq in seq_obj_list:
            seq_y = seq.m_target
            self.m_y.append(seq_y)
            self.m_y_id.append(10)

            ### self_src
            self.m_self_src.append(seq.m_item_list)

            self.m_position_self_src.append([i+1 for i in range(len(seq.m_item_list))])

            ### common_src
            self.m_common_src.extend(seq.m_neigh_common_item_list)

            ### common_time
            self.m_common_time.extend(seq.m_neigh_common_time_list)

            ### friend_diff_src
            self.m_friend_diff_src.extend(seq.m_neigh_diff_item_list)

            ### friend_num
            self.m_friend_num.append(len(seq.m_neigh_common_item_list))

class MYDATALOADER(object):
    def __init__(self, bucket_list):
        self.m_bucket_list = bucket_list
        # self.m_seq_corpus = seq_corpus
        # self.m_batch_size = batch_size

        # self.m_seq_num = len(self.m_seq_corpus.m_seq_list)
        # self.m_batch_num = self.m_seq_num//batch_size

        # self.m_words_num = 0

        # self.f_process_time()

        # if train_valid_test_flag == "train":
        #     self.f_count_vocabulary()

   
    def __iter__(self):
        print("shuffling")
        random.shuffle(self.m_bucket_list)

        for bucket_obj in self.m_bucket_list:

            batch_self_src = []
            batch_position_self_src = []
            batch_common_src = []
            batch_common_time = []
            batch_friend_diff_src = []
            batch_friend_num = []
            batch_y = []
            batch_y_id = []

            batch_self_src = bucket_obj.m_self_src
            batch_position_self_src = bucket_obj.m_position_self_src
            batch_common_src = bucket_obj.m_common_src
            batch_common_time = bucket_obj.m_common_time
            batch_friend_diff_src = bucket_obj.m_friend_diff_src
            batch_friend_num = bucket_obj.m_friend_num
            batch_y = bucket_obj.m_y
            batch_y_id = bucket_obj.m_y_id

            # for seq in batch_seq_list:
            #     seq_y = seq.m_target
            #     batch_y.append(seq_y)
            #     batch_y_id.append(10)

            #     ### self_src
            #     batch_self_src.append(seq.m_item_list)

            #     batch_position_self_src.append([i+1 for i in range(len(seq.m_item_list))])

            #     ### common_src
            #     batch_common_src.extend(seq.m_neigh_common_item_list)

            #     ### common_time
            #     batch_common_time.extend(seq.m_neigh_common_time_list)

            #     ### friend_diff_src
            #     batch_friend_diff_src.extend(seq.m_neigh_diff_item_list)

            #     ### friend_num
            #     batch_friend_num.append(len(seq.m_neigh_common_item_list))

            batch_max_len_self_src = max([len(i) for i in batch_self_src])
            batch_max_len_common_src = max([len(i) for i in batch_common_src])
            batch_max_len_common_time = max([len(i) for i in batch_common_time])
            batch_max_len_friend_diff_src = max([len(i) for i in batch_friend_diff_src])
            
            ### padding

            batch_self_src = [i+[0]*(batch_max_len_self_src-len(i)) for i in batch_self_src]
            batch_position_self_src = [i+[0]*(batch_max_len_self_src-len(i)) for i in batch_position_self_src]

            batch_common_src = [i+[0]*(batch_max_len_common_src-len(i)) for i in batch_common_src]

            batch_common_time = [i+[0]*(batch_max_len_common_time-len(i)) for i in batch_common_time]

            batch_friend_diff_src = [i+[0]*(batch_max_len_friend_diff_src-len(i)) for i in batch_friend_diff_src]
            
            batch_self_src = torch.tensor(batch_self_src)
            batch_position_self_src = torch.tensor(batch_position_self_src)
            batch_common_src = torch.tensor(batch_common_src)
            batch_common_time = torch.FloatTensor(batch_common_time)
            batch_friend_diff_src = torch.tensor(batch_friend_diff_src)
            batch_y = torch.tensor(batch_y)
            batch_y_id = torch.tensor(batch_y_id)

            yield batch_position_self_src, batch_self_src, batch_common_src, batch_common_time, batch_friend_diff_src, batch_friend_num, batch_y, batch_y_id
