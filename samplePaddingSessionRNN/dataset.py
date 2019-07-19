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

		self.m_input_seq_list = []
		self.m_target_action_seq_list = []
		self.m_input_seq_idx_list = []

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
					self.m_input_seq_list.append(input_sub_seq)
					self.m_target_action_seq_list.append(target_sub_seq)
					self.m_input_seq_idx_list.append(action_index)

				if action_index > window_size:
					input_sub_seq = action_seq_arr[action_index-window_size:action_index]
				
					target_sub_seq = action_seq_arr[action_index]
					self.m_input_seq_list.append(input_sub_seq)
					self.m_target_action_seq_list.append(target_sub_seq)
					self.m_input_seq_idx_list.append(action_index)

	
	def __len__(self):
		return len(self.m_input_seq_list)

	def __getitem__(self, index):
		x = self.m_input_seq_list[index]
		y = self.m_target_action_seq_list[index]

		x = np.array(x)
		y = np.array(y)

		x_tensor = torch.LongTensor(x)
		y_tensor = torch.LongTensor(y)

		return x_tensor, y_tensor

	@property
	def items(self):
		# print("first item", self.m_itemmap['<PAD>'])
		return self.m_itemmap

class DataLoader():
	def __init__(self, dataset, batch_size):
		self.m_dataset = dataset
		self.m_batch_size = batch_size
	
	def __iter__(self):
		
		print("shuffling")
		temp = list(zip(self.m_dataset.m_input_seq_list, self.m_dataset.m_target_action_seq_list, self.m_dataset.m_input_seq_idx_list))
		random.shuffle(temp)
		
		input_action_seq_list, target_action_seq_list, input_seq_idx_list = zip(*temp)

		batch_size = self.m_batch_size
        
		input_num = len(input_action_seq_list)
		batch_num = int(input_num/batch_size)
		print("batch num", batch_num)
		for batch_index in range(batch_num):
			x_batch = []
			y_batch = []
			idx_batch = []

			for seq_index_batch in range(batch_size):
				seq_index = batch_index*batch_size+seq_index_batch
				x = list(input_action_seq_list[seq_index])
				y = target_action_seq_list[seq_index]
                
				x_batch.append(x)
				y_batch.append(y)
				idx_batch.append(input_seq_idx_list[seq_index])
                
			x_batch, y_batch, x_len_batch, idx_batch = self.batchifyData(x_batch, y_batch, idx_batch)

			# y_neg_batch = self.generateNegSample(y_batch, pop, self.m_negative_num)

			x_batch_tensor = torch.LongTensor(x_batch)
			y_batch_tensor = torch.LongTensor(y_batch)
			idx_batch_tensor = torch.LongTensor(idx_batch)

			# y_neg_batch_tensor = torch.LongTensor(y_neg_batch)
            
			yield x_batch_tensor, y_batch_tensor, x_len_batch, idx_batch_tensor

	def generateNegSample(self, y_batch, pop, negative_sample_num):

		# candidate_itemid = all_itemid-y_batch
		neg_sample_batch = np.searchsorted(pop, np.random.rand(negative_sample_num*y_batch.shape[0]))
		
		neg_sample_batch = neg_sample_batch.reshape(-1, negative_sample_num)
		
		return neg_sample_batch

	def batchifyData(self, input_action_seq_batch, target_action_seq_batch, idx_batch):
		seq_len_batch = [len(seq_i) for seq_i in input_action_seq_batch]

		longest_len_batch = max(seq_len_batch)
		batch_size = len(input_action_seq_batch)

		pad_input_action_seq_batch = np.zeros((batch_size, longest_len_batch))
		pad_target_action_seq_batch = np.zeros(batch_size)
		pad_seq_len_batch = np.zeros(batch_size)
		pad_idx_batch = np.zeros(batch_size)

		# print(len(seq_len_batch))
		# print(len(input_action_seq_batch))
		# print(len(target_action_seq_batch))
		# print(len(idx_batch))

		zip_batch = sorted(zip(seq_len_batch, input_action_seq_batch, target_action_seq_batch, idx_batch), reverse=True)

		for seq_i, (seq_len_i, input_action_seq_i, target_action_seq_i, seq_idx) in enumerate(zip_batch):

			pad_input_action_seq_batch[seq_i, 0:seq_len_i] = input_action_seq_i
			pad_target_action_seq_batch[seq_i] = target_action_seq_i
			pad_seq_len_batch[seq_i] = seq_len_i
			pad_idx_batch[seq_i] = seq_idx
            
		return pad_input_action_seq_batch, pad_target_action_seq_batch, pad_seq_len_batch, pad_idx_batch
