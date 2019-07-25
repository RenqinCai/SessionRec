import pandas as pd
import numpy as np
import torch
import datetime
import pickle
import random
# import sys
import collections

class Data(object):

	def __init__(self, _action_file, _cate_file, _time_file, _valid_start_time, _test_start_time, _observed_threshold, _window_size):
		action_f = open(_action_file, "rb")

		self.m_itemmap = {}
		self.m_itemFreq_map = {}

		self.m_catemap = {}

		self.m_itemmap['<PAD>'] = 0
		self.m_catemap['<PAD>'] = 0

		action_seq_arr_total = pickle.load(action_f)

		cate_f = open(_cate_file, "rb")
		cate_seq_arr_total = pickle.load(cate_f)

		time_f = open(_time_file, "rb")
		time_seq_arr_totoal = pickle.load(time_f)

		action_seq_num = len(action_seq_arr_total)
		print("action seq num", action_seq_num)

		cate_seq_num = len(cate_seq_arr_total)
		print("cate seq num", cate_seq_num)

		time_seq_num = len(time_seq_arr_totoal)
		print("time seq num", time_seq_num)

		# self.m_seq_list = []
		
		self.m_input_seq_list_train = []
		self.m_input_seqLen_list_train = []

		self.m_input_cate_seq_list_train = []
		# self.m_input_subseqNum_seq_list = []

		self.m_input_subseq_list_cate_list_train = []
		self.m_input_cate_subseq_list_cate_list_train = []

		self.m_input_subseqLen_list_cate_list_train = []
		self.m_input_subseqNum_seq_cate_list_train = []

		self.m_target_action_seq_list_train = []
		self.m_target_cate_seq_list_train = []

		self.m_input_seq_list_test = []
		self.m_input_seqLen_list_test = []

		self.m_input_cate_seq_list_test = []

		self.m_input_subseq_list_cate_list_test = []
		self.m_input_cate_subseq_list_cate_list_test = []

		self.m_input_subseqLen_list_cate_list_test = []
		self.m_input_subseqNum_seq_cate_list_test = []

		self.m_target_action_seq_list_test = []
		self.m_target_cate_seq_list_test = []

		self.m_input_seq_idx_list_train = []
		self.m_input_seq_idx_list_test = []

		print("loading item map")

		print("finish loading item map")
		print("observed_threshold", _observed_threshold, _window_size)
		print("loading data")

		# time_threshold = 1512000000
		valid_start_time = _valid_start_time
		test_start_time = _test_start_time
		print("valid_start_time", valid_start_time)
		print("test start time", test_start_time)

		# seq_num = 1
		for seq_index in range(action_seq_num):
			# print("*"*10, "seq index", seq_index, "*"*10)
			action_seq_arr = action_seq_arr_total[seq_index]
			cate_seq_arr = cate_seq_arr_total[seq_index]
			time_seq_arr = time_seq_arr_totoal[seq_index]

			actionNum_seq = len(action_seq_arr)
			actionNum_seq_cate = len(cate_seq_arr)
			actionNum_seq_time = len(time_seq_arr)

			if actionNum_seq != actionNum_seq_cate:
				assert("!= seq len action cate")
			
			if actionNum_seq_cate != actionNum_seq_time:
				assert("!= seq len cate time")

			if actionNum_seq < _window_size :
				_window_size = actionNum_seq

			cate_action_list_map_user = {}
			for action_index in range(actionNum_seq):
				item_cur = action_seq_arr[action_index]
				cate_cur = cate_seq_arr[action_index]
				time_cur = time_seq_arr[action_index]

				if time_cur <= valid_start_time:
					if item_cur not in self.m_itemmap:
						self.m_itemmap[item_cur] = len(self.m_itemmap)
					
					if item_cur not in self.m_itemFreq_map:
						self.m_itemFreq_map[item_cur] = 0.0
					
					self.m_itemFreq_map[item_cur] += 1.0
								
				if action_index < _observed_threshold:

					if cate_cur not in self.m_catemap:
						cate_id_cur = len(self.m_catemap)
						self.m_catemap[cate_cur] = cate_id_cur

					if cate_cur not in cate_action_list_map_user:
						cate_action_list_map_user[cate_cur] = []

					cate_action_list_map_user[cate_cur].append(item_cur)

					continue

				### train data
				if time_cur <= valid_start_time:
					self.addItem2train(action_index, _window_size, cate_cur, action_seq_arr, cate_seq_arr, cate_action_list_map_user)

				if time_cur > valid_start_time:
					if time_cur <= test_start_time:
						self.addItem2test(action_index, _window_size, cate_cur, action_seq_arr, cate_seq_arr, cate_action_list_map_user)
						# print("test subseq num", self.m_input_subseqNum_seq_cate_list_test)
				### test data

				cate_cur = cate_seq_arr[action_index]
				if cate_cur not in self.m_catemap:
					cate_id_cur = len(self.m_catemap)
					self.m_catemap[cate_cur] = cate_id_cur

				if cate_cur not in cate_action_list_map_user:
					cate_action_list_map_user[cate_cur] = []

				cate_action_list_map_user[cate_cur].append(item_cur)

		# print("debug", self.m_input_subseq_list_seq_list[:10])
		print("subseq num for training", len(self.m_input_seq_list_train))
		print("subseq len num for training", len(self.m_input_seqLen_list_train))
		print("seq idx num for training", len(self.m_input_seq_idx_list_train))

		print("subseq num for testing", len(self.m_input_seq_list_test))
		print("subseq len num for testing", len(self.m_input_seqLen_list_test))

		self.train_dataset = Dataset(self.m_input_subseq_list_cate_list_train, self.m_input_subseqLen_list_cate_list_train, self.m_input_subseqNum_seq_cate_list_train, self.m_input_cate_subseq_list_cate_list_train, self.m_input_seq_list_train, self.m_input_cate_seq_list_train, self.m_input_seqLen_list_train, self.m_target_action_seq_list_train, self.m_target_cate_seq_list_train, self.m_input_seq_idx_list_train, self.m_itemFreq_map)

		self.test_dataset = Dataset(self.m_input_subseq_list_cate_list_test, self.m_input_subseqLen_list_cate_list_test, self.m_input_subseqNum_seq_cate_list_test, self.m_input_cate_subseq_list_cate_list_test, self.m_input_seq_list_test, self.m_input_cate_seq_list_test,self.m_input_seqLen_list_test, self.m_target_action_seq_list_test, self.m_target_cate_seq_list_test, self.m_input_seq_idx_list_test, self.m_itemFreq_map)
		
	def save2pickle(self, folder):
		
		dataset_pickle = {'train': self.train_dataset, 'test':self.test_dataset}
		dataset_pickle_file = folder+"dataset.pickle"
		print("dataset_pickle_file", dataset_pickle_file)
		f = open(dataset_pickle_file, 'wb')
		pickle.dump(dataset_pickle, f)

	def addItem2train(self, action_index, _window_size, _cate_cur, action_seq_arr, cate_seq_arr, cate_action_list_map_user):
		
		subseq_num = 0

		seq = None
		cate_seq = None
		if action_index <= _window_size:
			seq = action_seq_arr[:action_index]
			cate_seq = cate_seq_arr[:action_index]
		else:
			seq = action_seq_arr[action_index-_window_size:action_index]
			cate_seq = cate_seq_arr[action_index-_window_size:action_index]

		self.m_input_seq_list_train.append(seq)
		self.m_input_cate_seq_list_train.append(cate_seq)

		actionNum_seq = len(seq)
		self.m_input_seqLen_list_train.append(actionNum_seq)

		action_list_sub_seq = []
		actionNum_list_sub_seq = []

		cate_subseq_list = []
		
		for cate in cate_action_list_map_user:
			subseq_cate = cate_action_list_map_user[cate].copy()[-_window_size:]
			actionNum_subseq_cate = len(subseq_cate)

			action_list_sub_seq.append(subseq_cate)
			actionNum_list_sub_seq.append(actionNum_subseq_cate)
			
			subseq_num += 1

			cate_subseq_list.append(cate)

		self.m_input_cate_subseq_list_cate_list_train.append(cate_subseq_list)
		self.m_target_cate_seq_list_train.append(_cate_cur)

		self.m_input_subseq_list_cate_list_train.append(action_list_sub_seq)
		self.m_input_subseqLen_list_cate_list_train.append(actionNum_list_sub_seq)
		self.m_input_subseqNum_seq_cate_list_train.append(subseq_num)

		target_subseq = action_seq_arr[action_index]
		self.m_target_action_seq_list_train.append(target_subseq)
		self.m_input_seq_idx_list_train.append(action_index)

	def addItem2test(self, action_index, _window_size, _cate_cur, action_seq_arr, cate_seq_arr, cate_action_list_map_user):
		subseq_num = 0

		seq = None
		cate_seq = None
		if action_index <= _window_size:
			seq = action_seq_arr[:action_index]
			cate_seq = cate_seq_arr[:action_index]
		else:
			seq = action_seq_arr[action_index-_window_size:action_index]
			cate_seq = cate_seq_arr[action_index-_window_size:action_index]

		self.m_input_seq_list_test.append(seq)
		self.m_input_cate_seq_list_test.append(cate_seq)

		actionNum_seq = len(seq)
		self.m_input_seqLen_list_test.append(actionNum_seq)

		action_list_sub_seq = []
		actionNum_list_sub_seq = []

		cate_subseq_list = []
		
		for cate in cate_action_list_map_user:
			subseq_cate = cate_action_list_map_user[cate].copy()[-_window_size:]
			actionNum_subseq_cate = len(subseq_cate)

			action_list_sub_seq.append(subseq_cate)
			actionNum_list_sub_seq.append(actionNum_subseq_cate)
			
			subseq_num += 1
			
			cate_subseq_list.append(cate)

		self.m_input_cate_subseq_list_cate_list_test.append(cate_subseq_list)
		self.m_target_cate_seq_list_test.append(_cate_cur)

		self.m_input_subseq_list_cate_list_test.append(action_list_sub_seq)
		self.m_input_subseqLen_list_cate_list_test.append(actionNum_list_sub_seq)
		self.m_input_subseqNum_seq_cate_list_test.append(subseq_num)

		target_subseq = action_seq_arr[action_index]
		self.m_target_action_seq_list_test.append(target_subseq)
		self.m_input_seq_idx_list_test.append(action_index)

	@property
	def items(self):
		print("item num", len(self.m_itemmap))
		# print("first item", self.m_itemmap['<PAD>'])
		return self.m_itemmap
	
	@property
	def cates(self):
		print("cate num", len(self.m_catemap))
		
		return self.m_catemap

class Dataset(object):
	def __init__(self, input_subseq_list_cate_list, input_subseqLen_list_cate_list, input_subseqNum_seq_cate_list, input_cate_subseq_list_cate_list, input_seq_list, input_cate_seq_list, input_seqLen_list, target_action_seq_list, target_cate_seq_list, input_seq_idx_list, itemFreq_map):
		self.m_input_subseq_list_cate_list = input_subseq_list_cate_list
		self.m_input_subseqLen_list_cate_list = input_subseqLen_list_cate_list
		self.m_input_subseqNum_seq_cate_list = input_subseqNum_seq_cate_list

		self.m_input_seq_list = input_seq_list
		self.m_input_seqLen_list = input_seqLen_list

		self.m_input_cate_seq_list = input_cate_seq_list
		
		self.m_target_action_seq_list = target_action_seq_list
		self.m_input_seq_idx_list = input_seq_idx_list

		self.m_itemFreq_map = itemFreq_map

		self.m_input_cate_subseq_list_cate_list = input_cate_subseq_list_cate_list
		self.m_target_cate_seq_list = target_cate_seq_list
