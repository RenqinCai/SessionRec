import pandas as pd
import numpy as np
import torch

import pickle

class Dataset(object):
	def __init__(self, itemFile, data_name, sep='\t', session_key='SessionID', item_key='ItemId', time_key='timestamp', n_sample=-1, itemmap=None, itemstamp=None, time_sort=False):

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

		sess_num = len(item_sess_arr)
		print("session num", sess_num)

		sess_len_list = []

		self.itemmap = itemmap

		item_id_sess_arr = []

		for sess_index in range(sess_num):
			item_sess_unit_list = item_sess_arr[sess_index]

			sess_len = len(item_sess_unit_list)

			sess_action_num = 0
			# sess_len_list.append(sess_len)

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
			# print("sess_action_num", sess_action_num)
			if sess_action_num != 0:
			#     print("error action num zero")
			# else:
				sess_len_list.append(sess_action_num)

		# print("item sess arr", item_sess_arr[:10])
		# print("item id sess arr", item_id_sess_arr[:100])
		self.click_offsets = self.getClickOffset(sess_num, sess_len_list)
		self.item_arr = np.array(item_id_sess_arr)
		self.sess_num = len(sess_len_list)

		print("session num", self.sess_num)
		print("action num", len(self.item_arr))
		
		# exit()

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

		valid_sess_num = len(sess_len_list)

		offsets = np.zeros(valid_sess_num+1, dtype=np.int32)
		offsets[1:] = np.array(sess_len_list).cumsum()

		return offsets

	@property
	def items(self):
		# print("first item", self.itemmap[0])
		return self.itemmap
		# return len(self.itemmap)
		# return self.itemmap.ItemId.unique()

class DataLoader():
	def __init__(self, dataset, BPTT, batch_size=50):

		self.dataset = dataset
		self.m_batch_size = batch_size
		self.m_output_size = len(dataset.itemmap)
	   
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
			# print("*"*10, "min len", "*"*10, minlen)
			# print("mask_sess_arr", mask_sess_arr)
			# idx_input = item_arr[start]
			# idx_target = item_arr[start+1]

			# input_tensor = torch.LongTensor(idx_input)
			# target_tensor = torch.LongTensor(idx_target)

			# yield input_tensor, target_tensor, mask_sess_arr

			# mask_sess_arr = []

			for i in range(minlen-1):
				idx_input_sample = item_arr[start+i]
				idx_target_sample = item_arr[start+i+1]
				
				idx_input = idx_input_sample

				idx_target = idx_target_sample

				input_tensor = torch.LongTensor(idx_input)
				target_tensor = torch.LongTensor(idx_target)

				total_action_num += self.m_batch_size
				
				if i > 0:
					mask_sess_arr = []

				# print("mask_sess_arr", mask_sess_arr)
				yield input_tensor, target_tensor, mask_sess_arr
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
