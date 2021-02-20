import pandas as pd
import numpy as np
import torch

import pickle

class Dataset(object):
	def __init__(self, itemFile, data_name, sep='\t', session_key='SessionID', item_key='ItemId', time_key='timestamp', n_sample=-1, itemmap=None, itemstamp=None, time_sort=False):
		item_map = {}
		data_file = open(itemFile, "rb")

		item_sess_arr = None
		data_sess_arr = pickle.load(data_file)
		if data_name == "movielen_itemmap":
			item_sess_arr = data_sess_arr['action_list']
			itemmap = data_sess_arr['itemmap']
		
		if data_name == "movielen":
			item_sess_arr = data_sess_arr

		if data_name == "xing":
			item_sess_arr = data_sess_arr

		if data_name == "taobao":
			item_sess_arr = data_sess_arr

		sess_num = len(item_sess_arr)
		print("session num", sess_num)

		sess_len_list = []

		item_id_sess_arr = []

		for sess_index in range(sess_num):
			item_sess_unit_list = item_sess_arr[sess_index]

			sess_len = len(item_sess_unit_list)

			sess_action_num = 0

			for action_index in range(sess_len):
				item = item_sess_unit_list[action_index]

				sess_action_num += 1
				
				item_id_sess_arr.append(item)

				if item not in item_map:
					item_map[item] = len(item_map)
			
			if sess_action_num != 0:
				sess_len_list.append(sess_action_num)

		self.click_offsets = self.getClickOffset(sess_num, sess_len_list)
		self.item_arr = np.array(item_id_sess_arr)
		self.sess_num = len(sess_len_list)

		self.m_itemmap = item_map

		print("session num", self.sess_num)
		print("action num", len(self.item_arr))

	def getClickOffset(self, sess_num, sess_len_list):

		if sess_num != len(sess_len_list):
			print("error sess num")

		valid_sess_num = len(sess_len_list)

		offsets = np.zeros(valid_sess_num+1, dtype=np.int32)
		offsets[1:] = np.array(sess_len_list).cumsum()

		return offsets

	@property
	def items(self):
		# print("first item", self.itemmap[0])
		return self.m_itemmap
		# return len(self.itemmap)
		# return self.itemmap.ItemId.unique()

class DataLoader():
	def __init__(self, dataset, BPTT, batch_size=50):

		self.dataset = dataset
		self.m_batch_size = batch_size
	   
		self.m_bptt = BPTT

	def __iter__(self):
		click_offsets = self.dataset.click_offsets
		item_arr = self.dataset.item_arr

		print("action num", len(item_arr))
		sess_num = self.dataset.sess_num

		user_num = len(click_offsets)-1
		print("user num", user_num)

		iters = np.arange(self.m_batch_size)
		maxiter = iters.max()
		start = click_offsets[iters]
		end = click_offsets[iters+1]

		mask_sess_arr = []
		finished = False

		window_size = self.m_bptt

		total_action_num = 0

		start = start + window_size-1

		print("window_size", window_size)

		min_len = int((end-start).min())
		if min_len <= window_size:
			print("error window size for min lens")

		while not finished:
			minlen = (end-start).min()

			for i in range(minlen-1):
				x_input_sample = item_arr[start+i]
				y_target_sample = item_arr[start+i+1]
				idx_sample = start+i+1

				x_input = x_input_sample
				y_target = y_target_sample

				input_tensor = torch.LongTensor(x_input)
				target_tensor = torch.LongTensor(y_target)
				idx_tensor = torch.LongTensor(idx_sample)

				total_action_num += self.m_batch_size
				
				if i > 0:
					mask_sess_arr = []

				# print("mask_sess_arr", mask_sess_arr)
				yield input_tensor, target_tensor, mask_sess_arr, idx_tensor
				# if self.m_onehot_flag:
				#     self.m_onehot_buffer.zero_()

				#     index = input_tensor.view(-1, 1)
				#     input_tensor = self.m_onehot_buffer.scatter_(1, index, 1)

				# yield idx_input, input_tensor, target_tensor, mask_sess_arr

			start = start + minlen - 1

			mask_sess_arr = np.arange(self.m_batch_size)[(end-start) <= 1]
			for mask_sess in mask_sess_arr:
				# print("sess start", start[mask_sess])
				# print("sess end", end[mask_sess])
				# print("... masking ...")
				maxiter = maxiter+1
				if maxiter >= sess_num:
					finished = True
					total_action_num += np.sum(end-start-1)
					# print(mask_sess, "mask_sess_arr", mask_sess_arr)
					# print("total_action_num", total_action_num)
					break

				start[mask_sess] = click_offsets[maxiter]+window_size-1
				end[mask_sess] = click_offsets[maxiter+1]

				iters[mask_sess] = maxiter
