import pandas as pd
import numpy as np
import torch

import pickle

class Dataset(object):

    def __init__(self, data_prefix, data_name, observed_threshold, window_size, itemKey, timeKey, itemmap=None):
        item_filename = data_prefix + itemKey
        data_file = open(item_filename, "rb")
        
        time_filename = data_prefix + timeKey
        time_seq_arr_total = pickle.load(open(time_filename, "rb"))

        action_seq_arr_total = None
        data_seq_arr = pickle.load(data_file)

        if data_name == "movielen_itemmap":
            action_seq_arr_total = data_seq_arr['action_list']
            itemmap = data_seq_arr['itemmap']

        if data_name == "movielen":
            action_seq_arr_total = data_seq_arr

        if data_name == "xing":
            action_seq_arr_total = data_seq_arr

        seq_num = len(action_seq_arr_total)
        print("seq num", seq_num)

        seq_len_list = []

        self.m_itemmap = itemmap
        if itemmap is None:
            self.m_itemmap = {}
        self.m_itemmap['<PAD>'] = 0

        self.m_seq_list = []

        self.m_input_action_seq_list = []
        self.m_target_action_seq_list = []
        self.m_input_seq_len_list = []
        self.m_time_action_seq_list = []
        
        print("loading item map")

        for seq_index in range(seq_num):
            action_seq_arr = action_seq_arr_total[seq_index]

            action_num_seq = len(action_seq_arr)

            action_seq_list = []

            for action_index in range(action_num_seq):
                item = action_seq_arr[action_index]

                if itemmap is None: 
                    if item not in self.m_itemmap:
                        item_id = len(self.m_itemmap)
                        self.m_itemmap[item] = item_id
                else:
                    if item not in self.m_itemmap:
                        continue

                item_id = self.m_itemmap[item]
                action_seq_list.append(item_id)
                

            self.m_seq_list.append(action_seq_list)

        print("finish loading item map")

        print("loading data")
        for seq_index in range(seq_num):
            action_seq_arr = self.m_seq_list[seq_index]
            time_seq_arr =  time_seq_arr_total[seq_index]

            action_num_seq = len(action_seq_arr)

            if action_num_seq < window_size :
                window_size = action_num_seq

            for action_index in range(observed_threshold, window_size):
                input_sub_seq = action_seq_arr[:action_index]
                target_sub_seq = action_seq_arr[action_index]
                self.m_input_action_seq_list.append(input_sub_seq)
                self.m_target_action_seq_list.append(target_sub_seq)
                self.m_input_seq_len_list.append(action_index)
                self.m_time_action_seq_list.append(  np.asarray( time_seq_arr[action_index] - time_seq_arr[:action_index]) )
                

            for action_index in range(window_size, action_num_seq):
                input_sub_seq = action_seq_arr[action_index-window_size+1:action_index]
                target_sub_seq = action_seq_arr[action_index]
                self.m_input_action_seq_list.append(input_sub_seq)
                self.m_target_action_seq_list.append(target_sub_seq)
                self.m_input_seq_len_list.append(action_index)
                self.m_time_action_seq_list.append(  np.asarray(time_seq_arr[action_index] - time_seq_arr[action_index-window_size+1:action_index]) )

    def __len__(self):
        return len(self.m_input_action_seq_list)

    def __getitem__(self, index):
        x = self.m_input_action_seq_list[index]
        y = self.m_target_action_seq_list[index]
        t = self.m_time_action_seq_list[index]

        x_tensor = torch.LongTensor(np.asarray(x))
        y_tensor = torch.LongTensor(np.asarray(y))
        t_tensor = torch.FloatTensor(t)
        
        return x_tensor, y_tensor, t_tensor

    @property
    def items(self):
        print("first item", self.m_itemmap['<PAD>'])
        return self.m_itemmap

class DataLoader():
    def __init__(self, dataset, batch_size):
        self.m_dataset = dataset
        self.m_batch_size = batch_size

    def __iter__(self):

        batch_size = self.m_batch_size
        input_action_seq_list = self.m_dataset.m_input_action_seq_list
        target_action_seq_list = self.m_dataset.m_target_action_seq_list
        input_time_seq_list = self.m_dataset.m_time_action_seq_list
        
        input_num = len(input_action_seq_list)
        batch_num = int(input_num/batch_size)

        for batch_index in range(batch_num):
            x_batch = []
            y_batch = []
            t_batch = []

            for seq_index_batch in range(batch_size):
                seq_index = batch_index*batch_size+seq_index_batch
                x = input_action_seq_list[seq_index]
                y = target_action_seq_list[seq_index]
                t = input_time_seq_list[seq_index]

                x_batch.append(x)
                y_batch.append(y)
                t_batch.append(t)
            
            
            x_batch, y_batch, t_batch, x_len_batch = self.batchifyData(x_batch, y_batch, t_batch)

            x_batch_tensor = torch.LongTensor(x_batch)
            y_batch_tensor = torch.LongTensor(y_batch)
            t_batch_tensor = torch.FloatTensor(t_batch)
            
            yield x_batch_tensor, y_batch_tensor, t_batch_tensor, x_len_batch

    def batchifyData(self, input_action_seq_batch, target_action_seq_batch, input_time_seq_batch):
        seq_len_batch = [len(seq_i) for seq_i in input_action_seq_batch]

        longest_len_batch = max(seq_len_batch)
        batch_size = len(input_action_seq_batch)

        pad_input_action_seq_batch = np.zeros((batch_size, longest_len_batch))
        pad_target_action_seq_batch = np.zeros(batch_size)
        pad_input_time_seq_batch = np.zeros((batch_size, longest_len_batch))
        pad_seq_len_batch = np.zeros(batch_size)

        zip_batch = sorted(zip(seq_len_batch, input_action_seq_batch, target_action_seq_batch, input_time_seq_batch), key=lambda x: x[0], reverse=True)

#         zip_batch = zip(seq_len_batch, input_action_seq_batch, target_action_seq_batch, input_time_seq_batch)

        for seq_i, (seq_len_i, input_action_seq_i, target_action_seq_i, input_time_seq_i) in enumerate(zip_batch):
            pad_input_action_seq_batch[seq_i, 0:seq_len_i] = input_action_seq_i
            pad_input_time_seq_batch[seq_i, 0:seq_len_i] = input_time_seq_i
            pad_target_action_seq_batch[seq_i] = target_action_seq_i
            pad_seq_len_batch[seq_i] = seq_len_i
        ### map item id back to start from 0
        # target_action_seq_batch = [target_i-1 for target_i in target_action_seq_batch]

        return pad_input_action_seq_batch, pad_target_action_seq_batch, pad_input_time_seq_batch, pad_seq_len_batch

