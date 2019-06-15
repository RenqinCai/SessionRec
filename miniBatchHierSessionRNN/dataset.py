import pandas as pd
import numpy as np
import torch

import pickle

class Dataset(object):
	def __init__(self, itemFile, data_name, sep='\t', session_key='SessionID', item_key='ItemId', time_key='timestamp', n_sample=-1, itemmap=None, itemstamp=None, time_sort=False):

		data_file = open(itemFile, "rb")
		data_sess_arr = pickle.load(data_file)
		
		total_action_pickle = None
		total_time_pickle = None
	   
		if data_name == "movielen_itemmap":
			total_action_pickle = data_sess_arr['action']
			itemmap = data_sess_arr['itemmap']
		
		if data_name == "movielen":
			total_action_pickle = data_sess_arr

		if data_name == "xing":
			total_action_pickle = data_sess_arr['item']
			total_time_pickle = data_sess_arr['time']

		if data_name == "taobao":
			total_action_pickle = data_sess_arr['item']
			total_time_pickle = data_sess_arr['time']

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

			# print("user_padding_action_num", user_padding_action_num)

		### split a list of ations per user into a list of session per user

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
		# exit()

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

	@property
	def items(self):
		# print("first item", self.itemmap[0])
		return self.m_item_map
		# return len(self.itemmap)
		# return self.itemmap.ItemId.unique()

class DataLoader():
	def __init__(self, dataset, BPTT, batch_size=50):
	
		self.dataset = dataset
		self.m_batch_size = batch_size
		# self.m_output_size = len(dataset.m_item_map)
	   
		self.m_bptt = BPTT        

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

	