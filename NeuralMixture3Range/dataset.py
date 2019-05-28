"""
use mini batch for more than one steps back
"""

import pandas as pd
import numpy as np
import torch

import pickle

class Dataset(object):
    def __init__(self, itemFile, sep='\t', session_key='SessionID', item_key='ItemId', time_key='timestamp', n_sample=-1, itemmap=None, itemstamp=None, time_sort=False):

        data_file = open(itemFile, "rb")

        data_sess_arr = pickle.load(data_file)

        item_sess_arr = data_sess_arr

        sess_num = len(item_sess_arr)
        print("session num", sess_num)

        sess_len_list = []

        self.itemmap = itemmap

        item_id_arr = []

        for sess_index in range(sess_num):
            item_sess_unit_list = item_sess_arr[sess_index]
            # print("item_sess_unit_list", item_sess_unit_list)
            
            sess_len = len(item_sess_unit_list)

            # sess_len_list.append(sess_len)
            sess_action_num = 0

            # item_id_sess_arr = []

            for action_index in range(sess_len):
                item = item_sess_unit_list[action_index]
                if itemmap is None:
                    self.addItem(item, itemmap)

                if item not in self.itemmap:
                    continue
                
                item_id = self.itemmap[item]

                sess_action_num += 1
                # if itemmap is not None:
                #     print(item_id, item)
                # item_id_sess_arr.append(item_id)
                item_id_arr.append(item_id)

            if sess_action_num != 0:
            #     print("error action num zero")
            # else:
                sess_len_list.append(sess_action_num)

        self.click_offsets = self.getClickOffset(sess_num, sess_len_list)
        self.item_arr = np.array(item_id_arr)
        self.sess_num = len(sess_len_list)

        print("sess num", len(self.item_arr))
        
    def addItem(self, item, itemmap=None):
        if itemmap is None:
            if self.itemmap is None:
                self.itemmap = {}

            if item not in self.itemmap:
                item_id = len(self.itemmap)
                self.itemmap[item] = item_id

    def getClickOffset(self, sess_num, sess_len_list):

        if sess_num != len(sess_len_list):
            print("error sess num")

        offsets = np.zeros(len(sess_len_list)+1, dtype=np.int32)
        offsets[1:] = np.array(sess_len_list).cumsum()

        # offsets = np.zeros(sess_num+1, dtype=np.int32)
        # offsets = np.array(sess_len_list, dtype=np.int32)

        return offsets

    @property
    def items(self):
        # print("first item", self.itemmap)
        return self.itemmap

class DataLoader():
    def __init__(self, dataset, BPTT, batch_size=50, onehot_flag=-1):
        if onehot_flag == -1:
            onehot_flag = True
        else:
            onehot_flag = False
        self.dataset = dataset
        self.m_batch_size = batch_size
        self.m_onehot_flag = onehot_flag
        self.m_onehot_buffer = None
        self.m_output_size = len(dataset.itemmap)
       
        self.m_window_size = BPTT
        # self.m_device = torch.device('cuda' if use_cuda else 'cpu')
        
        if self.m_onehot_flag:
            self.m_onehot_buffer = self.initOneHot()

    def initOneHot(self):

        if self.m_window_size > 1:
            onehot_buffer = torch.FloatTensor(self.m_window_size, self.m_batch_size, self.m_output_size)
            # onehot_buffer = onehot_buffer.to(self.m_device)
        else:
            onehot_buffer = torch.FloatTensor(self.m_batch_size, self.m_output_size)
            # onehot_buffer = onehot_buffer.to(self.m_device)

        return onehot_buffer
        # print("self.batch_size", self.batch_size)

    def __iter__(self):
        click_offsets = self.dataset.click_offsets
        item_arr = self.dataset.item_arr

        sess_num = self.dataset.sess_num

        iters = np.arange(self.m_batch_size)
        maxiter = iters.max()
        start = click_offsets[iters]
        end = click_offsets[iters+1]

        mask_sess_arr = []
        finished = False

        idx_input_cum = []

        window_size = self.m_window_size

        for i in range(window_size-1):
            idx_input_sample = item_arr[start+i]
            if len(idx_input_cum) == 0:
                idx_input_cum.append(idx_input_sample)
                idx_input_cum = np.array(idx_input_cum)
            else:
                idx_input_cum = np.vstack((idx_input_cum, idx_input_sample))
                # print("size", idx_input_cum.shape, idx_input_sample.shape)
        start = start + window_size-1

        min_len = int((end-start).min())
        if min_len <= window_size:
            print("error window size for min lens")

        while not finished:
            min_len = int((end-start).min())

            if min_len <= 0:
                print("error window size for min lens")

            for i in range(min_len-1):
                # print("iters", iters)
                # print("start+i", start+i)
                idx_input_sample = item_arr[start+i]
                idx_target_sample = item_arr[start+i+1]
                
                if window_size > 1:
                    if i != 0:
                        idx_input_cum = idx_input_cum[1:]
              
                    idx_input_cum = np.vstack((idx_input_cum, idx_input_sample))
                else:
                    idx_input_cum = idx_input_sample
                
                ### idx_input_cum size:  seq_len*batch_size
                idx_input = idx_input_cum

                ### idx_input size: seq_len*batch_size
                # idx_input = np.transpose(idx_input_cum)

                idx_target = idx_target_sample

                input_tensor = torch.LongTensor(idx_input)
                target_tensor = torch.LongTensor(idx_target)

                if self.m_onehot_flag:
                    self.m_onehot_buffer.zero_()
                    if window_size > 1:
                        # input_ = torch.unsqueeze(input_tensor, 2).to(self.m_device)
                        input_ = torch.unsqueeze(input_tensor, 2)
                        input_tensor = self.m_onehot_buffer.scatter_(2, input_, 1)
                    else:
                        # index = input_tensor.view(-1, 1).to(self.m_device)
                        index = input_tensor.view(-1, 1)
                        input_tensor = self.m_onehot_buffer.scatter_(1, index, 1)

                yield idx_input, input_tensor, target_tensor, mask_sess_arr

            start = start + min_len - 1
            # maxiter = maxiter + 1

            mask_sess_arr = np.arange(self.m_batch_size)[(end-start) <= 1]
            idx_input_cum = idx_input_cum[1:] 
            for mask_sess in mask_sess_arr:
                maxiter = maxiter+1
                if maxiter >= sess_num:
                    finished = True
                    break

                start[mask_sess] = click_offsets[maxiter]+window_size-1
                end[mask_sess] = click_offsets[maxiter+1]

                iters[mask_sess] = maxiter

                # print(idx_input_cum[:, mask_sess])
                # print("idx input cum", idx_input_cum)
                # print(click_offsets[maxiter], start[mask_sess])
                if window_size > 1:
                    # print("shape", idx_input_cum.shape)
                    idx_input_cum[:, mask_sess] = item_arr[click_offsets[maxiter]: start[mask_sess]]
                



