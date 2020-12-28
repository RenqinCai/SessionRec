import pandas as pd
import numpy as np
import torch
import datetime
import pickle

import random

class MYDATA(object):
	def __init__(self, action_file, cate_file, time_file, valid_start_time, test_start_time):
		action_f = open(action_file, "rb")
		action_total = pickle.load(action_f)
		action_seq_num = len(action_total)
		print("action seq num", action_seq_num)

		self.m_itemmap = {}
		self.m_catemap = {}

		self.m_itemmap['<PAD>'] = 0
		self.m_catemap['<PAD>'] = 0

		cate_f = open(cate_file, "rb")
		cate_total = pickle.load(cate_f)
		cate_seq_num = len(cate_total)
		print("cate seq num", cate_seq_num)

		time_f = open(time_file, "rb")
		time_total = pickle.load(time_f)
		time_seq_num = len(time_total)
		print("time seq num", time_seq_num)

			### train
		self.m_x_action_list_train = []
		self.m_x_cate_list_train = []
		self.m_x_time_list_train = []

		### test
		self.m_x_action_list_test = []
		self.m_x_cate_list_test = []
		self.m_x_time_list_test = []

		print("loading item map")
		print("loading data")

		print("valid_start_time", valid_start_time)
		print("test start time", test_start_time)

		for seq_index in range(action_seq_num):
			# print("*"*10, "seq index", seq_index, "*"*10)
			action_seq_arr = action_total[seq_index]
			cate_seq_arr = cate_total[seq_index]
			time_seq_arr = time_total[seq_index]

			actionNum_seq = len(action_seq_arr)
			actionNum_seq_cate = len(cate_seq_arr)
			actionNum_seq_time = len(time_seq_arr)

			if actionNum_seq != actionNum_seq_cate:
				assert("!= seq len action cate")
			
			if actionNum_seq_cate != actionNum_seq_time:
				assert("!= seq len cate time")

			train_start_index_seq = 0
			valid_start_index_seq = 0
			test_start_index_seq = 0

			for action_index in range(actionNum_seq):
				item_cur = action_seq_arr[action_index]
				cate_cur = cate_seq_arr[action_index]
				time_cur = time_seq_arr[action_index]

				if time_cur <= valid_start_time:
					if item_cur not in self.m_itemmap:
						self.m_itemmap[item_cur] = len(self.m_itemmap)
				
				if cate_cur not in self.m_catemap:
					cate_id_cur = len(self.m_catemap)
					self.m_catemap[cate_cur] = cate_id_cur

				if time_cur <= valid_start_time:
					valid_start_index_seq = action_index

				if time_cur > valid_start_time:
					if time_cur <= test_start_time:
						test_start_index_seq = action_index
					else:
						break
			# print("train_start_index_seq", train_start_index_seq, valid_start_index_seq, test_start_index_seq)

			self.addItem2train(train_start_index_seq, valid_start_index_seq, action_seq_arr, cate_seq_arr, time_seq_arr)
					
			self.addItem2test(valid_start_index_seq, test_start_index_seq, action_seq_arr, cate_seq_arr, time_seq_arr)

		print("seq num for training", len(self.m_x_action_list_train))
		print("seq num of actions for training", len(self.m_x_cate_list_train))

		print("seq num for testing", len(self.m_x_action_list_test))
		print("seq num of actions for testing", len(self.m_x_cate_list_test))

		self.train_dataset = MYDATASET(self.m_x_action_list_train, self.m_x_cate_list_train, self.m_x_time_list_train)

		self.test_dataset = MYDATASET(self.m_x_action_list_test, self.m_x_cate_list_test, self.m_x_time_list_test)

	def addItem2train(self, start_index, end_index, item_seq, cate_seq, time_seq):
		if start_index >= end_index-1:
			return

		self.m_x_action_list_train.append(item_seq[start_index:end_index])
		self.m_x_cate_list_train.append(cate_seq[start_index:end_index])
		self.m_x_time_list_train.append(time_seq[start_index:end_index])

	def addItem2test(self, start_index, end_index, item_seq, cate_seq, time_seq):
		# print(end_index-start_index)
		if start_index >= end_index-1:
			return

		self.m_x_action_list_test.append(item_seq[start_index:end_index])
		self.m_x_cate_list_test.append(cate_seq[start_index:end_index])
		self.m_x_time_list_test.append(time_seq[start_index:end_index])

	def items(self):
		print("item num", len(self.m_itemmap))
		return len(self.m_itemmap)

	def cates(self):
		print("cate num", len(self.m_catemap))
		return len(self.m_catemap)

class MYDATASET(object):
	def __init__(self, x_action_list, x_cate_list, x_time_list):
		self.m_x_action_list = x_action_list
		self.m_x_cate_list = x_cate_list
		self.m_x_time_list = x_time_list

	def segment2Session(self):
		total_action_pickle = self.m_x_action_list
		total_time_pickle = self.m_x_time_list

		user_num = len(total_action_pickle)
		print("user num", user_num)
		
		total_action_num = 0

		### preprocess user action pickle into user sess action arr
		session_thresh = 30*60 ## 30 minutes
		# user_sess_action_arr = user_action_pickle

		user_sess_action_arr = []
		user_sess_action_num_List = []

		action_arr = []

		itemmap = {}

		for user_index in range(user_num):
			# print("user index", user_index)
			if user_index %5000 == 0:
				print("user index", user_index)
			user_action_list = total_action_pickle[user_index]
			user_time_list = total_time_pickle[user_index]
			
			user_action_num = len(user_action_list)
			total_action_num += user_action_num
			# print("user_action_num", user_action_num)
			
			user_padding_action_num = 0

			user_sess_action_list = []

			sess_start_index = 0

			item_id = len(itemmap)
			item = user_action_list[0]
			if item not in itemmap:
				itemmap[item] = item_id

			user_action_list = np.array(user_action_list)
			user_action_list = user_action_list.tolist()
			# action_arr += user_action_list

			for action_index in range(1, user_action_num):
				last_action_time = user_time_list[action_index-1]
				cur_action_time = user_time_list[action_index]

				item_id = len(itemmap)
				item = user_action_list[action_index]
				if item not in itemmap:
					itemmap[item] = item_id

				delta_time =  cur_action_time-last_action_time
				if delta_time > session_thresh:
					sess_action_user = None
					if sess_start_index == 0:
						sess_action_user = user_action_list[sess_start_index:action_index]
					else:
						sess_action_user = user_action_list[sess_start_index:action_index]
					user_sess_action_list.append(sess_action_user)
					action_arr += sess_action_user
					
					sess_start_index = action_index

					user_sess_action_num = len(sess_action_user)
					# print("user_sess_action_num", user_sess_action_num, end=" ")
					user_padding_action_num += user_sess_action_num

					user_sess_action_num_List.append(user_sess_action_num)
					 
					# if action_index == user_action_num-1:
			sess_action_user = None
			if sess_start_index == 0:	
				sess_action_user = user_action_list[sess_start_index:]
			else:
				sess_action_user = user_action_list[sess_start_index:]

			user_sess_action_list.append(sess_action_user)
			action_arr += sess_action_user

			user_sess_action_num = len(sess_action_user)
			# print("user_sess_action_num", user_sess_action_num)

			user_sess_action_num_List.append(user_sess_action_num)
			user_padding_action_num += user_sess_action_num

			user_sess_action_arr.append(user_sess_action_list)

		
		### user_sess_num_list: sess_num per user
		user_sess_num_list = [len(user_sess_list) for user_sess_list in user_sess_action_arr]

		self.m_sess_offsets = self.setSessOffset(user_sess_num_list)

		# user_sess_action_num_List = [len(sess_action_user) for sess_action_user in sess_action_user_list for sess_action_user_list in user_sess_action_arr]

		self.m_action_offsets = self.setActionOffset(user_sess_action_num_List)
	
		### a list of sessions
		self.m_user_sess_action_arr = user_sess_action_arr

		self.m_item_map = itemmap

		self.m_action_arr = np.array(action_arr)

		print("padding user action num", len(self.m_action_arr))
		print("total user_action_num", total_action_num)

	def setActionOffset(self, user_sess_action_num_List):
		sess_num = len(user_sess_action_num_List)

		offsets = np.zeros(sess_num+1, dtype=np.int32)

		offsets[1:] = np.array(user_sess_action_num_List).cumsum()

		return offsets

	def setSessOffset(self, user_sess_num_list):
		user_num = len(user_sess_num_list)

		offsets = np.zeros(user_num+1, dtype=np.int32)
		offsets[1:] = np.array(user_sess_num_list).cumsum()

		return offsets

class DataLoader():
	def __init__(self, dataset, batch_size=50):
	
		self.dataset = dataset
		self.m_batch_size = batch_size
		# self.m_output_size = len(dataset.m_item_map)
	
	def __iter__(self):
		### sess_offsets: 
		sess_offsets = self.dataset.m_sess_offsets

		action_offsets = self.dataset.m_action_offsets
		
		action_arr = self.dataset.m_action_arr

		print("action num", len(action_arr))

		user_num = len(self.dataset.m_sess_offsets) - 1
		# print("user num", user_num)

		sess_num = len(self.dataset.m_action_offsets) - 1
		# print("sess num", sess_num)

		# sess_iters = np.arange(self.m_batch_size)
		# max_sess_iter = sess_iters.max()

		user_iters = np.arange(self.m_batch_size)
		max_user_iter = user_iters.max()

		action_user_offsets = action_offsets[sess_offsets]
	
		sess_iters = sess_offsets[user_iters]
		sess_start = action_offsets[sess_iters]
		sess_end = action_offsets[sess_iters+1]
			
		user_start = action_user_offsets[user_iters]
		user_end = action_user_offsets[user_iters+1]

		mask_sess_arr_start = []
		mask_sess_arr_end = []

		mask_user_arr_start = []
		mask_user_arr_end = []

		finished = False

		# window_size = self.m_bptt

		total_action_num = 0

		start_user_mask = np.zeros(self.m_batch_size)
		
		while not finished:
			user_min_len = int((user_end-user_start).min())

			for i in range(user_min_len-1):
				x_input = action_arr[user_start+i]
				y_target = action_arr[user_start+i+1]
				idx_sample = user_start+i+1

				input_tensor = torch.LongTensor(x_input)
				target_tensor = torch.LongTensor(y_target)
				idx_tensor = torch.LongTensor(idx_sample)

				if i > 0:
					mask_user_arr_start = []
				
				mask_sess_arr_start = np.arange(self.m_batch_size)[(sess_end-user_start-i-1) <= 1]
				if len(mask_sess_arr_start):
					for mask_sess in mask_sess_arr_start:
						sess_iters[mask_sess] += 1

						if sess_iters[mask_sess] >= sess_num:
							finished = True
							break

						sess_end[mask_sess] = action_offsets[sess_iters[mask_sess]+1]
	
				yield input_tensor, target_tensor, mask_sess_arr_start, mask_user_arr_start, start_user_mask, idx_tensor
			
			user_start = user_start + user_min_len - 1
			mask_user_arr_start = np.arange(self.m_batch_size)[(user_end-user_start) <= 1]

			for mask_user in mask_user_arr_start:
				max_user_iter = max_user_iter + 1
				if max_user_iter >= user_num:
					finished = True
					break

				user_start[mask_user] = action_user_offsets[max_user_iter]
				user_end[mask_user] = action_user_offsets[max_user_iter+1]

				sess_iters[mask_user] = sess_offsets[max_user_iter]
				# sess_start[mask_user] = action_offsets[sess_iters[mask_user]]
				sess_end[mask_user] = action_offsets[sess_iters[mask_user]+1]

	