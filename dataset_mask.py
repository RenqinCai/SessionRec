import pandas as pd
import numpy as np
import torch

import pickle

class Dataset(object):
    def __init__(self, itemFile, sep='\t', session_key='SessionID', item_key='ItemId', time_key='timestamp', n_sample=-1, itemmap=None, itemstamp=None, time_sort=False):

        data_file = open(itemFile, "rb")

        data_sess_arr = pickle.load(data_file)

        item_sess_arr = data_sess_arr[1]

        sess_num = len(item_sess_arr)
        print("session num", sess_num)

        sess_len_list = []

        self.itemmap = itemmap

        for sess_index in range(sess_num):
            item_sess_unit_list = item_sess_arr[sess_index]

            sess_len = len(item_sess_unit_list)

            sess_len_list.append(sess_len)

            for action_index in range(sess_len):
                item = item_sess_unit_list[action_index]
                if itemmap is None:
                    self.addItem(item, itemmap)

                item_id = self.itemmap[item]

                # if itemmap is not None:
                #     print(item_id, item)
                # item_id_sess_arr.append(item_id)

        # print("item sess arr", item_sess_arr[:10])
        # print("item id sess arr", item_id_sess_arr[:100])

        # self.click_offsets = self.getClickOffset(sess_num, sess_len_list)
        # self.item_arr = np.array(item_id_sess_arr)
        self.sess_num = sess_num

        sorted_session_idex_arr = np.argsort(sess_len_list)
        self.item_arr = item_sess_arr[sorted_session_idex_arr]
        self.sess_len = sess_len_list

    def addItem(self, item, itemmap=None):
        if itemmap is None:
            if self.itemmap is None:
                self.itemmap = {}

            if item not in self.itemmap:
                item_id = len(self.itemmap)
                self.itemmap[item] = item_id

    @property
    def items(self):
        return self.itemmap
        # return len(self.itemmap)
        # return self.itemmap.ItemId.unique()

class DataLoader():
    def __init__(self, dataset, batch_size=50):
        self.dataset = dataset
        self.batch_size = batch_size
        # print("self.batch_size", self.batch_size)

    def __iter__(self):
        item_arr = self.dataset.item_arr

        sess_num = self.dataset.sess_num

        # batch_iter = 0

        batch_num = int(sess_num / self.batch_size)

        finished = False

        sess_len = self.dataset.sess_len

        while not finished:
            for batch_iter in range(batch_num):
                idx_input = []
                idx_target = []
                mask = []
                # idx_input = item_arr[batch_iter*batch_size: (batch_iter+1)*batch_size]
                sess_len_batch = sess_len[batch_iter*batch_size: (batch_iter+1)*batch_size]

                max_len = sess_len_batch.max()-1

                for sample_iter in range(batch_size):
                    cur_sample = item_arr[batch_iter*batch_size+sample_iter][:-1]
                    idx_input_sample = np.concatenate(cur_sample, np.zeros(max_len-len(cur_sample)))
                    idx_input.append(idx_input_sample)

                    idx_target_sample = np.concatenate(cur_sample[1:], np.zeros(max_len-len(cur_sample)))
                    idx_target.append(idx_target_sample)

                    mask_sample = np.concatenate(np.ones(len(cur_sample)), np.zeros(max_len-len(cur_sample)))

                    mask.append(mask_sample)

                input_tensor = torch.LongTensor(idx_input)
                target_tensor = torch.LongTensor(idx_target)
                mask_tensor = torch.LongTensor(mask)

                yield input_tensor, target_tensor, mask_tensor
