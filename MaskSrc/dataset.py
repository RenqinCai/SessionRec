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
        item_sess_arr = np.array(item_sess_arr)
        sess_len_list = np.array(sess_len_list)

        sorted_session_idex_arr = np.argsort(sess_len_list)
        self.item_arr = item_sess_arr[sorted_session_idex_arr]
        self.sess_len = sess_len_list[sorted_session_idex_arr]

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
        print("item_arr", item_arr[:10])
        sess_num = self.dataset.sess_num

        # batch_iter = 0
        batch_size = self.batch_size
        batch_num = int(sess_num / batch_size)

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
                    cur_sample = np.array(item_arr[batch_iter*batch_size+sample_iter])
                    # print("cur_sample", cur_sample)
                    cur_sample_num = len(cur_sample)-1

                    # input_sample = np.array(cur_sample[:-1])
                    # print("input_sample", input_sample)
                    input_sample = cur_sample[:-1]
                    # idx_input_sample = input_sample+[0 for i in range(max_len-cur_sample_num)]
                    idx_input_sample = np.concatenate([input_sample, np.zeros(max_len-cur_sample_num)])
                    # print("idx_input_sample", idx_input_sample)
                    
                    idx_input.append(idx_input_sample)

                    # target_sample = np.array(cur_sample[1:])
                    # idx_target_sample = np.concatenate(target_sample, np.zeros(max_len-cur_sample_num))
                    target_sample = cur_sample[1:]
                    idx_target_sample = np.concatenate([target_sample, np.zeros(max_len-cur_sample_num)])
                    # idx_target_sample = target_sample+[0 for i in range(max_len-cur_sample_num)]
                    idx_target.append(idx_target_sample)

                    mask_sample = np.concatenate([np.ones(cur_sample_num), np.zeros(max_len-cur_sample_num)])

                    mask.append(mask_sample)


                input_tensor = torch.LongTensor(idx_input)
                # print("input_tensor size", input_tensor.size())
                target_tensor = torch.LongTensor(idx_target)
                mask_tensor = torch.LongTensor(mask)

                yield input_tensor, target_tensor, mask_tensor
