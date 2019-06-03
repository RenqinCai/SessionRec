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

		user_num = len(total_action_pickle)
		print("user num", user_num)

		### preprocess user action pickle into user sess action arr
		session_thresh = 30*60 ## 30 minutes
		# user_sess_action_arr = user_action_pickle

		user_sess_action_arr = []
		user_sess_action_num_List = []

		action_arr = []

		itemmap = {}

		for user_index in range(user_num):
			if user_index %5000 == 0:
				print("user index", user_index)
			user_action_list = total_action_pickle[user_index]
			user_time_list = total_time_pickle[user_index]
			
			user_action_num = len(user_action_list)

			# print("user_action_num", user_action_num, user_action_list)
			
			user_sess_action_list = []

			sess_start_index = 0

			item_id = len(itemmap)
			item = user_action_list[0]
			if item not in itemmap:
				itemmap[item] = item_id

			user_action_list = user_action_list.tolist()
			action_arr += user_action_list

			for action_index in range(1, user_action_num):
				last_action_time = user_time_list[action_index-1]
				cur_action_time = user_time_list[action_index]

				item_id = len(itemmap)
				item = user_action_list[action_index]
				if item not in itemmap:
					itemmap[item] = item_id

				delta_time =  cur_action_time-last_action_time
				if delta_time > session_thresh:
					sess_action_user = user_action_list[sess_start_index:action_index]
					user_sess_action_list.append(sess_action_user)
					sess_start_index = action_index

					user_sess_action_num = len(sess_action_user)
					user_sess_action_num_List.append(user_sess_action_num)
				
					# if action_index == user_action_num-1:
			sess_action_user = user_action_list[sess_start_index:]
			user_sess_action_list.append(sess_action_user)
			user_sess_action_num_List.append(len(sess_action_user))

			user_sess_action_arr.append(user_sess_action_list)

		### split a list of ations per user into a list of session per user

		user_sess_num_list = [len(user_sess_list) for user_sess_list in user_sess_action_arr]

		self.m_sess_offsets = self.setSessOffset(user_sess_num_list)

		# user_sess_action_num_List = [len(sess_action_user) for sess_action_user in sess_action_user_list for sess_action_user_list in user_sess_action_arr]

		self.m_action_offsets = self.setActionOffset(user_sess_action_num_List)
	
		### a list of sessions
		self.m_user_sess_action_arr = user_sess_action_arr

		self.m_item_map = itemmap

		self.m_action_arr = np.array(action_arr)

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
		self.m_output_size = len(dataset.m_item_map)
	   
		self.m_bptt = BPTT        

	def __iter__(self):
		sess_offsets = self.dataset.m_sess_offsets

		action_offsets = self.dataset.m_action_offsets
		print("action num", len(action_offsets))

		action_arr = self.dataset.m_action_arr

		user_num = len(self.dataset.m_sess_offsets) - 1
		sess_num = len(self.dataset.m_action_offsets) - 1

		sess_iters = np.arange(self.m_batch_size)
		max_sess_iter = sess_iters.max()

		user_iters = np.arange(self.m_batch_size)
		max_user_iter = user_iters.max()

		sess_start = action_offsets[sess_iters]

		# print("sess start", sess_start, sess_start.dtype)
		sess_end = action_offsets[sess_iters+1]

		action_user_offsets = action_offsets[sess_offsets]

		user_start = action_user_offsets[user_iters]
		user_end = action_user_offsets[user_iters+1]
		
		# start = click_offsets[iters]
		# end = click_offsets[iters+1]

		mask_sess_arr = []
		finished = False

		# window_size = self.m_bptt

		total_action_num = 0

		# start = start + window_size-1

		# sess_min_len = int((sess_end-sess_start).min())
		# if sess_min_len <= window_size:
		#     print("error window size for min lens")

		while not finished:
			sess_min_len = int((sess_end-sess_start).min())
			print("sess_min_len", sess_min_len)
### iterate through sessions per user
			for i in range(sess_min_len-1):
				
				idx_input = action_arr[sess_start+i]
				idx_target = action_arr[sess_start+i+1]

				input_tensor = torch.LongTensor(idx_input)
				target_tensor = torch.LongTensor(idx_target)

				yield input_tensor, target_tensor, mask_sess_arr, mask_user_arr

			sess_start = sess_start + sess_min_len - 1

			mask_sess_arr = np.arange(self.m_batch_size)[(sess_end-sess_start) <= 1]
			print("mask sess arr", mask_sess_arr)
			for mask_sess in mask_sess_arr:
				max_sess_iter = max_sess_iter+1
				if max_sess_iter >= sess_num:
					finished = True
					# total_action_num += np.sum(end-start-1)
					# print(mask_sess, "mask_sess_arr", mask_sess_arr)
					# print("total_action_num", total_action_num)
					break

				sess_start[mask_sess] = action_offsets[max_sess_iter]
				sess_end[mask_sess] = action_offsets[max_sess_iter+1]

				sess_iters[mask_sess] = max_sess_iter

			mask_user_arr = np.arange(self.m_batch_size)[(user_end-sess_start) <= 1]
			print("mask user arr", mask_user_arr)
			for mask_user in mask_user_arr:
				max_user_iter = max_user_iter + 1
				if max_user_iter >= user_num:
					finished = True
					break
				
				user_start[mask_user] = action_user_offsets[max_user_iter]
				user_end[mask_user] = action_user_offsets[max_user_iter+1]
