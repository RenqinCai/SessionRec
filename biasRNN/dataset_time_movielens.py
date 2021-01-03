"""
clean code
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
		self.m_target_pos = -1
		
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

class MYDATA(object):

	def __init__(self, train_seq_corpus, test_seq_corpus, window_size):
		# train_action_f = open(train_action_file, "rb")
		# train_seq_corpus = pickle.load(train_action_f)
		# train_action_seq_num = len(train_seq_corpus.m_seq_list)
		train_action_seq_num = len(train_seq_corpus)
		print("train action seq num", train_action_seq_num)

		# test_action_f = open(valid_action_file, "rb")
		# test_seq_corpus = pickle.load(test_action_f)
		test_action_seq_num = len(test_seq_corpus)
		print("test action seq num", test_action_seq_num)

		self.m_itemmap = {}

		self.m_itemmap['<PAD>'] = 0

		### train
		self.m_x_short_action_list_train = []
		self.m_x_short_actionNum_list_train = []
		
		self.m_y_action_train = []

		self.m_y_action_idx_train = []

		### test
		self.m_x_short_action_list_test = []
		self.m_x_short_actionNum_list_test = []

		self.m_y_action_test = []
		
		self.m_y_action_idx_test = []

		for seq_index in range(train_action_seq_num):
			seq_obj = train_seq_corpus[seq_index]

			item_list = seq_obj.m_item_list
			target = seq_obj.m_target

			self.m_x_short_action_list_train.append(item_list)
			self.m_x_short_actionNum_list_train.append(len(item_list))
			self.m_y_action_train.append(target)
			self.m_y_action_idx_train.append(10)

			for item in item_list:
				if item in self.m_itemmap:
					continue
				else:
					item_id = len(self.m_itemmap)
					self.m_itemmap[item] = item_id

		for seq_index in range(test_action_seq_num):
			seq_obj = test_seq_corpus[seq_index]

			item_list = seq_obj.m_item_list
			target = seq_obj.m_target

			self.m_x_short_action_list_test.append(item_list)
			self.m_x_short_actionNum_list_test.append(len(item_list))
			self.m_y_action_test.append(target)
			self.m_y_action_idx_test.append(10)

		print("seq num for training", len(self.m_x_short_action_list_train))
		print("seq num of actions for training", len(self.m_x_short_actionNum_list_train))

		print("seq num for testing", len(self.m_x_short_action_list_test))
		print("seq num of actions for testing", len(self.m_x_short_actionNum_list_test))

		self.train_dataset = MYDATASET(self.m_x_short_action_list_train, self.m_x_short_actionNum_list_train, self.m_y_action_train, self.m_y_action_idx_train)

		self.test_dataset = MYDATASET(self.m_x_short_action_list_test, self.m_x_short_actionNum_list_test, self.m_y_action_test, self.m_y_action_idx_test)

	def addItem2train(self, action_index, window_size, action_seq_arr):
		
		short_seq = None
		
		if action_index <= window_size:
			short_seq = action_seq_arr[:action_index]
		else:
			short_seq = action_seq_arr[action_index-window_size: action_index]
			
		self.m_x_short_action_list_train.append(short_seq)

		short_actionNum_seq = len(short_seq)
		self.m_x_short_actionNum_list_train.append(short_actionNum_seq)

		y_action = action_seq_arr[action_index]
		self.m_y_action_train.append(y_action)
		self.m_y_action_idx_train.append(action_index)

	def addItem2test(self, action_index, window_size, action_seq_arr):
		short_seq = None

		if action_index <= window_size:
			short_seq = action_seq_arr[:action_index]
		else:
			short_seq = action_seq_arr[action_index-window_size: action_index]

		self.m_x_short_action_list_test.append(short_seq)

		short_actionNum_seq = len(short_seq)
		self.m_x_short_actionNum_list_test.append(short_actionNum_seq)

		y_action = action_seq_arr[action_index]
		self.m_y_action_test.append(y_action)
		self.m_y_action_idx_test.append(action_index)

	def items(self):
		print("item num", len(self.m_itemmap))
		return len(self.m_itemmap)

class MYDATASET(object):
	def __init__(self, x_short_action_list, x_short_actionNum_list, y_action, y_action_idx):

		self.m_x_short_action_list = x_short_action_list
		self.m_x_short_actionNum_list = x_short_actionNum_list

		self.m_y_action = y_action

		self.m_y_action_idx = y_action_idx

class MYDATALOADER(object):
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
			x_short_actionNum_list_batch = x_short_actionNum_list[batch_index]

			y_action_batch = y_action[batch_index]
			y_action_idx_batch = y_action_idx[batch_index]

			x_short_action_batch = []
			x_short_actionNum_batch = []

			mask_short_action_batch = []

			max_short_actionNum_batch = max(x_short_actionNum_list_batch)
			
			for seq_index_batch in range(batch_size):
				x_short_actionNum_seq = x_short_actionNum_list_batch[seq_index_batch]
				
				pad_x_short_action_seq = x_short_action_list_batch[seq_index_batch]+[0]*(max_short_actionNum_batch-x_short_actionNum_seq)
				x_short_action_batch.append(pad_x_short_action_seq)

			x_short_action_batch = np.array(x_short_action_batch)

			x_short_actionNum_batch = np.array(x_short_actionNum_list_batch)
			mask_short_action_batch = np.arange(max_short_actionNum_batch)[None, :] < x_short_actionNum_batch[:, None]

			y_action_batch = np.array(y_action_batch)
			y_action_idx_batch = np.array(y_action_idx_batch)

			x_short_action_batch_tensor = torch.from_numpy(x_short_action_batch)
			mask_short_action_batch_tensor = torch.from_numpy(mask_short_action_batch*1).float()

			y_action_batch_tensor = torch.from_numpy(y_action_batch)

			y_action_idx_batch_tensor = torch.from_numpy(y_action_idx_batch)

			pad_x_short_actionNum_batch = np.array([i-1 if i > 0 else 0 for i in x_short_actionNum_batch])

			yield x_short_action_batch_tensor, mask_short_action_batch_tensor, pad_x_short_actionNum_batch, y_action_batch_tensor, y_action_idx_batch_tensor
