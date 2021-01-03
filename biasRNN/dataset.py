import pandas as pd
import numpy as np
import torch

import pickle
import random
# import sys

class Dataset(object):

	def __init__(self, action_file, observed_threshold, window_size, itemmap=None):
		action_f = open(action_file, "rb")

		self.m_itemmap = {}

		action_seq_arr_total = None
		action_seq_arr_total = pickle.load(action_f)

		seq_num = len(action_seq_arr_total)
		print("seq num", seq_num)

		self.m_x_short_action_list = []
		self.m_x_short_actionNum_list = []
		self.m_y_action = []
		self.m_y_action_idx = []

		print("loading item map")

		print("finish loading item map")
		print("observed_threshold", observed_threshold, window_size)
		print("loading data")
		for seq_index in range(seq_num):
			action_seq_arr = action_seq_arr_total[seq_index]

			action_num_seq = len(action_seq_arr)

			if action_num_seq < window_size :
				window_size = action_num_seq

			for action_index in range(action_num_seq):
				item = action_seq_arr[action_index]
				if item not in self.m_itemmap:
					item_id = len(self.m_itemmap)
					self.m_itemmap[item] = item_id

				if action_index < observed_threshold:
					continue

				if action_index <= window_size:
					input_sub_seq = action_seq_arr[:action_index]
					
					target_sub_seq = action_seq_arr[action_index]
					self.m_x_short_action_list.append(input_sub_seq)
					self.m_x_short_actionNum_list.append(len(input_sub_seq))
					self.m_y_action.append(target_sub_seq)
					self.m_y_action_idx.append(action_index)

				if action_index > window_size:
					input_sub_seq = action_seq_arr[action_index-window_size:action_index]
				
					target_sub_seq = action_seq_arr[action_index]
					self.m_x_short_action_list.append(input_sub_seq)
					self.m_x_short_actionNum_list.append(len(input_sub_seq))
					self.m_y_action.append(target_sub_seq)
					self.m_y_action_idx.append(action_index)

	@property
	def items(self):
		# print("first item", self.m_itemmap['<PAD>'])
		return self.m_itemmap

class DataLoader():
	def __init__(self, dataset, batch_size):
		self.m_dataset = dataset
		self.m_batch_size = batch_size

		sorted_data = sorted(zip(self.m_dataset.m_x_short_action_list, self.m_dataset.m_x_short_actionNum_list, self.m_dataset.m_y_action, self.m_dataset.m_y_action_idx), reverse=True)

		self.m_dataset.m_x_short_action_list,self.m_dataset.m_x_short_actionNum_list , self.m_dataset.m_y_action, self.m_dataset.m_y_action_idx = zip(*sorted_data)

		input_seq_num = len(self.m_dataset.m_x_short_action_list)
		batch_num = int(input_seq_num/batch_size)
		print("seq num", input_seq_num)
		print("batch size", self.m_batch_size)
		print("batch_num", batch_num)

		x_short_action_list = [self.m_dataset.m_x_short_action_list[i*batch_size:(i+1)*batch_size] for i in range(batch_num)]
		
		x_short_actionNum_list = [self.m_dataset.m_x_short_actionNum_list[i*batch_size:(i+1)*batch_size] for i in range(batch_num)]
		y_action = [self.m_dataset.m_y_action[i*batch_size:(i+1)*batch_size] for i in range(batch_num)]
	
		y_action_idx = [self.m_dataset.m_y_action_idx[i*batch_size:(i+1)*batch_size] for i in range(batch_num)]

		temp = list(zip(x_short_action_list, x_short_actionNum_list, y_action, y_action_idx))

		self.m_temp = temp
	
	def __iter__(self):
		print("shuffling")

		temp = self.m_temp
		random.shuffle(temp)

		x_short_action_list, x_short_actionNum_list, y_action, y_action_idx = zip(*temp)

		batch_size = self.m_batch_size
		
		batch_num = len(x_short_action_list)

		for batch_index in range(batch_num):

			x_short_action_list_batch = x_short_action_list[batch_index]
			# x_short_cate_list_batch = x_short_cate_list[batch_index]
			x_short_actionNum_list_batch = x_short_actionNum_list[batch_index]

			y_action_batch = y_action[batch_index]
			# y_cate_batch = y_cate[batch_index]
			y_action_idx_batch = y_action_idx[batch_index]

			x_short_action_batch = []
			# x_short_cate_batch = []
			x_short_actionNum_batch = []

			mask_short_action_batch = []

			max_short_actionNum_batch = max(x_short_actionNum_list_batch)
			
			for seq_index_batch in range(batch_size):
				x_short_actionNum_seq = x_short_actionNum_list_batch[seq_index_batch]
				
				pad_x_short_action_seq = x_short_action_list_batch[seq_index_batch]+[0]*(max_short_actionNum_batch-x_short_actionNum_seq)
				x_short_action_batch.append(pad_x_short_action_seq)

			x_short_action_batch = np.array(x_short_action_batch)
			# x_short_cate_batch = np.array(x_short_cate_batch)

			x_short_actionNum_batch = np.array(x_short_actionNum_list_batch)
			mask_short_action_batch = np.arange(max_short_actionNum_batch)[None, :] < x_short_actionNum_batch[:, None]

			y_action_batch = np.array(y_action_batch)
			# y_cate_batch = np.array(y_cate_batch)
			y_action_idx_batch = np.array(y_action_idx_batch)

			x_short_action_batch_tensor = torch.from_numpy(x_short_action_batch)
			# x_short_cate_batch_tensor = torch.from_numpy(x_short_cate_batch)
			mask_short_action_batch_tensor = torch.from_numpy(mask_short_action_batch*1).float()

			y_action_batch_tensor = torch.from_numpy(y_action_batch)
			# y_cate_batch_tensor = torch.from_numpy(y_cate_batch)

			y_action_idx_batch_tensor = torch.from_numpy(y_action_idx_batch)

	
			pad_x_short_actionNum_batch = np.array([i-1 if i > 0 else 0 for i in x_short_actionNum_batch])

			yield x_short_action_batch_tensor, mask_short_action_batch_tensor, pad_x_short_actionNum_batch, y_action_batch_tensor, y_action_idx_batch_tensor
