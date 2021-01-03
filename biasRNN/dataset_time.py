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

class MYDATA(object):

	def __init__(self, action_file, cate_file, time_file, valid_start_time, test_start_time, observed_thresh, window_size):
		action_f = open(action_file, "rb")
		action_total = pickle.load(action_f)
		action_seq_num = len(action_total)
		print("action seq num", action_seq_num)

		self.m_itemmap = {}

		self.m_itemmap['<PAD>'] = 0

		time_f = open(time_file, "rb")
		time_total = pickle.load(time_f)
		time_seq_num = len(time_total)
		print("time seq num", time_seq_num)

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

		print("loading item map")

		print("loading item map")
		print("observed_threshold", observed_thresh, window_size)
		print("loading data")

		print("valid_start_time", valid_start_time)
		print("test start time", test_start_time)

		for seq_index in range(action_seq_num):
			# print("*"*10, "seq index", seq_index, "*"*10)
			action_seq_arr = action_total[seq_index]
			time_seq_arr = time_total[seq_index]

			actionNum_seq = len(action_seq_arr)
			actionNum_seq_time = len(time_seq_arr)

			if actionNum_seq < window_size :
				window_size = actionNum_seq
			
			for action_index in range(actionNum_seq):
				item_cur = action_seq_arr[action_index]
				time_cur = time_seq_arr[action_index]

				if time_cur <= valid_start_time:
					if item_cur not in self.m_itemmap:
						self.m_itemmap[item_cur] = len(self.m_itemmap)

				if action_index < observed_thresh:
					continue

				if time_cur <= valid_start_time:
					self.addItem2train(action_index, window_size, action_seq_arr)

				if time_cur > valid_start_time:
					if time_cur <= test_start_time:
						self.addItem2test(action_index, window_size, action_seq_arr)

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