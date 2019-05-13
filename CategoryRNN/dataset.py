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

        item_id_sess_arr = []

        for sess_index in range(sess_num):
            item_sess_unit_list = item_sess_arr[sess_index]
            # print("item_sess_unit_list", item_sess_unit_list)
            
            sess_len = len(item_sess_unit_list)

            # sess_len_list.append(sess_len)
            sess_action_num = 0

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
                item_id_sess_arr.append(item_id)

            if sess_action_num == 0:
                print("error action num zero")
            sess_len_list.append(sess_action_num)

        print("item sess arr", item_sess_arr[:10])
        print("item id sess arr", item_id_sess_arr[:100])
        self.click_offsets = self.getClickOffset(sess_num, sess_len_list)
        self.item_arr = np.array(item_id_sess_arr)
        self.sess_num = sess_num
        # print(self.itemmap)
        # self.df = pd.read_csv(path, sep=sep, names=[session_key, item_key, time_key])
        # self.session_key = session_key
        # self.item_key = item_key
        # self.time_key = time_key
        # self.time_sort = time_sort

        # if n_sample > 0:
        #   self.df = self.df[:n_sample]

        # self.add_item_indices(itemmap=itemmap)

        # self.df.sort_values([session_key, time_key], inplace=True)
        # self.click_offsets = self.get_click_offset()
        # self.session_idx_arr = self.order_session_idx()

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
        offsets = np.zeros(sess_num+1, dtype=np.int32)
        offsets[1:] = np.array(sess_len_list).cumsum()

        return offsets

    # def add_item_indices(self, itemmap=None):
    #   if itemmap is None:
    #       item_ids = self.df[self.item_key].unique()
    #       item2idx = pd.Series(data=np.arange(len(item_ids)), index=item_ids)

    #       itemmap = pd.DataFrame({self.item_key: item_ids, 'item_idx':item2idx[item_ids].values})

    #   self.itemmap = itemmap

    #   self.df = pd.merge(self.df, self.itemmap, on=self.item_key, how='inner')

    # def get_click_offset(self):
    #   offsets = np.zeros(self.df[self.session_key].nunique()+1, dtype=np.int32)
    #   offsets[1:] = self.df.groupby(self.session_key).size().cumsum()

    #   return offsets

    # def order_session_idx(self):
    #   if self.time_sort:
    #       sessions_start_time = self.df.groupby(self.session_key)[self.time_key].min().values
    #       session_idx_arr = np.argsort(sessions_start_time)
    #   else:
    #       session_idx_arr = np.arange(self.df[self.session_key].nunique())

    #   return session_idx_arr
    @property
    def items(self):
        print("first item", self.itemmap[0])
        return self.itemmap
        # return len(self.itemmap)
        # return self.itemmap.ItemId.unique()

class DataLoader():
    def __init__(self, dataset, batch_size=50):
        self.dataset = dataset
        self.batch_size = batch_size
        # print("self.batch_size", self.batch_size)

    def __iter__(self):
        click_offsets = self.dataset.click_offsets
        item_arr = self.dataset.item_arr

        sess_num = self.dataset.sess_num

        iters = np.arange(self.batch_size)
        maxiter = iters.max()
        start = click_offsets[iters]
        end = click_offsets[iters+1]

        mask_sess_arr = []
        finished = False

        while not finished:
            minlen = (end-start).min()

            for i in range(minlen-1):
                idx_input = item_arr[start+i]
                idx_target = item_arr[start+i+1]
                # print(idx_input)
                # print(idx_target)
                input_tensor = torch.LongTensor(idx_input)
                target_tensor = torch.LongTensor(idx_target)

                yield input_tensor, target_tensor, mask_sess_arr


            start = start + minlen - 1
            maxiter = maxiter + 1

            mask_sess_arr = np.arange(self.batch_size)[(end-start) <= 1]
            for mask_sess in mask_sess_arr:
                maxiter = maxiter+1
                if maxiter >= sess_num:
                    finished = True
                    break

                start[mask_sess] = click_offsets[maxiter]
                end[mask_sess] = click_offsets[maxiter+1]

                iters[mask_sess] = maxiter
