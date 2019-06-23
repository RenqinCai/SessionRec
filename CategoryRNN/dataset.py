import pandas as pd
import numpy as np
import torch
import datetime
import pickle
import random
# import sys

class Dataset(object):

	def __init__(self, action_file, cate_file, data_name, observed_threshold, window_size, itemmap=None):
		action_f = open(action_file, "rb")

		self.m_itemmap = {}
		self.m_catemap = {}

		action_seq_arr_total = pickle.load(action_f)

		cate_f = open(cate_file, "rb")
		cate_seq_arr_total = pickle.load(cate_f)

		seq_num = len(action_seq_arr_total)
		print("seq num", seq_num)

		# self.m_seq_list = []

		### each user's sequence is composed of multiple sub sequences
		### sub sequence is composed of actions	
		
		self.m_input_seq_list = []
		self.m_input_seqLen_list = []
		# self.m_input_subseqNum_seq_list = []

		# self.m_input_subseq_list_cate_list = []
		# self.m_input_subseqLen_list_cate_list = []
		# self.m_input_subseqNum_seq_cate_list = []

		self.m_input_seq_list_cate_list = []
		self.m_input_seqLen_list_cate_list = []

		self.m_target_action_seq_list = []
		self.m_target_cate_seq_list = []

		self.m_input_seq_idx_list = []

		print("loading item map")

		print("finish loading item map")
		print("observed_threshold", observed_threshold, window_size)
		print("loading data")
		# seq_num = 1
		for seq_index in range(seq_num):
			# print("*"*10, "seq index", seq_index, "*"*10)
			action_seq_arr = action_seq_arr_total[seq_index]
			cate_seq_arr = cate_seq_arr_total[seq_index]

			actionNum_seq = len(action_seq_arr)

			if actionNum_seq < window_size :
				window_size = actionNum_seq

			cate_action_list_map_user = {}
			for action_index in range(actionNum_seq):
				item_cur = action_seq_arr[action_index]
				if item_cur not in self.m_itemmap:
					item_id_cur = len(self.m_itemmap)
					self.m_itemmap[item_cur] = item_id_cur

				subseq_num = 0
				action_list_sub_seq = []
				actionNum_list_sub_seq = []

				cate_cur = cate_seq_arr[action_index]
				if action_index < observed_threshold:

					if cate_cur not in self.m_catemap:
						cate_id_cur = len(self.m_catemap)
						self.m_catemap[cate_cur] = cate_id_cur

					if cate_cur not in cate_action_list_map_user:
						cate_action_list_map_user[cate_cur] = []

					cate_action_list_map_user[cate_cur].append(item_cur)

					continue

				if action_index <= window_size:
					subseq = action_seq_arr[:action_index]

					# action_list_sub_seq.append(subseq)
					# actionNum_list_sub_seq.append(action_index)
					
					self.m_input_seq_list.append(subseq)
					self.m_input_seqLen_list.append(action_index)

					# subseq_num += 1
					# print("cate action map", cate_action_list_map_user)
					# for cate in cate_action_list_map_user:
					# 	subseq_cate = cate_action_list_map_user[cate].copy()[:5]
					# 	actionNum_subseq_cate = len(subseq_cate)

					# 	action_list_sub_seq.append(subseq_cate)
					# 	actionNum_list_sub_seq.append(actionNum_subseq_cate)
						
					# 	subseq_num += 1

					if cate_cur in cate_action_list_map_user:
						subseq_cate = cate_action_list_map_user[cate_cur].copy()[:5]
						actionNum_subseq_cate = len(subseq_cate)

						# action_list_sub_seq.append(subseq_cate)
						# actionNum_list_sub_seq.append(actionNum_subseq_cate)
						subseq_num += 1

						self.m_input_seq_list_cate_list.append(subseq_cate)
						self.m_input_seqLen_list_cate_list.append(actionNum_subseq_cate)
					else:
						self.m_input_seq_list_cate_list.append([])
						self.m_input_seqLen_list_cate_list.append(0)
					# self.m_input_subseqNum_seq_cate_list.append(subseq_num)
					
					target_subseq = action_seq_arr[action_index]
					self.m_target_action_seq_list.append(target_subseq)

					self.m_input_seq_idx_list.append(action_index)

				if action_index > window_size:
					subseq = action_seq_arr[action_index-window_size:action_index]
					# action_list_sub_seq.append(subseq)
					# actionNum_list_sub_seq.append(window_size)

					self.m_input_seq_list.append(subseq)
					self.m_input_seqLen_list.append(window_size)

					# print("cate action map", cate_action_list_map_user)
					if cate_cur in cate_action_list_map_user:
						subseq_cate = cate_action_list_map_user[cate_cur].copy()[:5]
						actionNum_subseq_cate = len(subseq_cate)

						# action_list_sub_seq.append(subseq_cate)
						# actionNum_list_sub_seq.append(actionNum_subseq_cate)
						subseq_num += 1

						self.m_input_seq_list_cate_list.append(subseq_cate)
						self.m_input_seqLen_list_cate_list.append(actionNum_subseq_cate)
					else:
						self.m_input_seq_list_cate_list.append([])
						self.m_input_seqLen_list_cate_list.append(0)

					target_subseq = action_seq_arr[action_index]
					self.m_target_action_seq_list.append(target_subseq)

					self.m_input_seq_idx_list.append(action_index)
		
				if cate_cur not in self.m_catemap:
					cate_id_cur = len(self.m_catemap)
					self.m_catemap[cate_cur] = cate_id_cur

				if cate_cur not in cate_action_list_map_user:
					cate_action_list_map_user[cate_cur] = []

				cate_action_list_map_user[cate_cur].append(item_cur)

		# print("debug", self.m_input_subseq_list_seq_list[:10])
		print("subseq num", len(self.m_input_seq_list))
		print("subseq len num", len(self.m_input_seqLen_list))
		print("seq idx num", len(self.m_input_seq_idx_list))

	@property
	def items(self):
		# print("first item", self.m_itemmap['<PAD>'])
		return self.m_itemmap

class DataLoader():
	def __init__(self, dataset, batch_size):
		self.m_dataset = dataset
		self.m_batch_size = batch_size

		"""
		sort subsequences 
		"""

		sorted_data = sorted(zip(self.m_dataset.m_input_seqLen_list_cate_list, self.m_dataset.m_input_seq_list_cate_list, self.m_dataset.m_input_seq_list, self.m_dataset.m_input_seqLen_list , self.m_dataset.m_target_action_seq_list, self.m_dataset.m_input_seq_idx_list), reverse=True)
		
		self.m_dataset.m_input_seqLen_list_cate_list, self.m_dataset.m_input_seq_list_cate_list, self.m_dataset.m_input_seq_list, self.m_dataset.m_input_seqLen_list , self.m_dataset.m_target_action_seq_list, self.m_dataset.m_input_seq_idx_list = zip(*sorted_data)

	def __iter__(self):
		
		input_seqLen_list_cate_list = self.m_dataset.m_input_seqLen_list_cate_list
		input_seq_list_cate_list = self.m_dataset.m_input_seq_list_cate_list
		input_seq_list = self.m_dataset.m_input_seq_list
		input_seqLen_list = self.m_dataset.m_input_seqLen_list
		target_action_seq_list = self.m_dataset.m_target_action_seq_list
		input_seq_idx_list = self.m_dataset.m_input_seq_idx_list

		## batchify, subsequences from the same user should be put in the same batch

		batch_size = self.m_batch_size
		
		input_seq_num = len(input_seqLen_list_cate_list)
		batch_num = int(input_seq_num/batch_size)

		print("batch_num", batch_num)

		for batch_index in range(batch_num):
			
			# print("batch index", batch_index)

			x_cate_batch = []

			y_batch = []

			idx_batch = []

			max_actionNum_cate_batch = 0
	
			actionNum_cate_batch = []
			
			for seq_index_batch in range(batch_size):
				seq_index = batch_index*batch_size+seq_index_batch

				seqlen_list_cate_user = input_seqLen_list_cate_list[seq_index]

				actionNum_cate_batch.append(seqlen_list_cate_user)

			max_actionNum_cate_batch = max(actionNum_cate_batch)
			if max_actionNum_cate_batch == 0:
				max_actionNum_cate_batch = 1
			# print("max_subseqNum_cate_batch", max_subseqNum_cate_batch)

			seqLen_cate_batch = []
			mask_cate_batch = None

			# x_subseq_index_batch = []
			for seq_index_batch in range(batch_size):
				seq_index = batch_index*batch_size+seq_index_batch

				seq_list_cate_user = input_seq_list_cate_list[seq_index]

				pad_seq_list_cate_user = seq_list_cate_user+[0]*(max_actionNum_cate_batch-len(seq_list_cate_user))

				x_cate_batch.append(pad_seq_list_cate_user)

				# print("pad_subseq_list_user", pad_subseq_list_user)

				seqLen_cate_user = input_seqLen_list_cate_list[seq_index]
				seqLen_cate_batch.append(seqLen_cate_user)
				
				y = target_action_seq_list[seq_index]

				y_batch.append(y)
				idx_batch.append(input_seq_idx_list[seq_index])

			seqLen_cate_batch = np.array(seqLen_cate_batch)
			mask_cate_batch = np.arange(max_actionNum_cate_batch)[None,:] < seqLen_cate_batch[:, None]

			x_cate_batch = np.array(x_cate_batch)

			x_batch = []
			mask_batch = []
			seqLen_batch = []
			
			for seq_index_batch in range(batch_size):
				seq_index = batch_index*batch_size+seq_index_batch

				seq_user = input_seq_list[seq_index]
				seqLen_user = input_seqLen_list[seq_index]
			
				seqLen_batch.append(seqLen_user)

			max_seqLen_batch = max(seqLen_batch)

			for seq_index_batch in range(batch_size):
				seq_index = batch_index*batch_size+seq_index_batch

				seq_user = input_seq_list[seq_index]

				pad_seq_user = seq_user+[0]*(max_seqLen_batch-len(seq_user))
			
				x_batch.append(pad_seq_user)
			
			seqLen_batch = np.array(seqLen_batch)
			# print("seqLen_batch", seqLen_batch)
			mask_batch = np.arange(max_seqLen_batch)[None,:] < seqLen_batch[:, None]
			
			x_batch = np.array(x_batch)

			y_batch = np.array(y_batch)
			idx_batch = np.array(idx_batch)

			x_batch_tensor = torch.from_numpy(x_batch)
			mask_batch_tensor = torch.from_numpy(mask_batch*1)

			# print("x_cate_batch_tensor", x_cate_batch.shape)
			x_cate_batch_tensor = torch.from_numpy(x_cate_batch)
			y_batch_tensor = torch.from_numpy(y_batch)
			mask_cate_batch_tensor = torch.from_numpy(mask_cate_batch*1).float()
			# print("x batch size", x_cate_batch_tensor.size())
			# print("mask batch size", mask_batch_tensor.size())

			idx_batch_tensor = torch.LongTensor(idx_batch)
			
			yield x_cate_batch_tensor, mask_cate_batch_tensor, max_actionNum_cate_batch, seqLen_cate_batch, x_batch_tensor, mask_batch_tensor, seqLen_batch, y_batch_tensor, idx_batch_tensor
