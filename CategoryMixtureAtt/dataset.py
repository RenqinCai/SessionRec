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
		
		self.m_input_subseq_list_seq_list = []
		self.m_input_subseq_cate_list = []

		self.m_input_subseqLen_list_seq_list = []
		self.m_input_subseqNum_seq_list = []

		self.m_target_action_seq_list = []
		self.m_target_cate_seq_list = []

		self.m_input_seq_idx_list = []

		print("loading item map")

		print("finish loading item map")
		print("observed_threshold", observed_threshold, window_size)
		print("loading data")
		# seq_num = 200
		for seq_index in range(seq_num):
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
				if action_index < observed_threshold:

					cate_cur = cate_seq_arr[action_index]
					if cate_cur not in self.m_catemap:
						cate_id_cur = len(self.m_catemap)
						self.m_catemap[cate_cur] = cate_id_cur

					if cate_cur not in cate_action_list_map_user:
						cate_action_list_map_user[cate_cur] = []

					cate_action_list_map_user[cate_cur].append(item_cur)

					continue

				if action_index <= window_size:
					# input_sub_seq = action_seq_arr[:action_index-1]
					subseq = action_seq_arr[:action_index]

					action_list_sub_seq.append(subseq)
					actionNum_list_sub_seq.append(action_index)
					# self.m_input_subseq_action_list.append(input_sub_seq)
					# self.m_input_subseq_seqlen_list.append(action_index)
					
					subseq_num += 1
					for cate in cate_action_list_map_user:
						subseq_cate = cate_action_list_map_user[cate]
						actionNum_subseq_cate = len(subseq_cate)

						action_list_sub_seq.append(subseq_cate)
						actionNum_list_sub_seq.append(actionNum_subseq_cate)
						# self.m_input_subseq_action_list.append(input_sub_seq_cate)
						# self.m_input_subseq_seqlen_list.append(cate_action_user_num)
						
						subseq_num += 1

					self.m_input_subseq_list_seq_list.append(action_list_sub_seq)
					self.m_input_subseqLen_list_seq_list.append(actionNum_list_sub_seq)

					self.m_input_subseqNum_seq_list.append(subseq_num)
					
					target_subseq = action_seq_arr[action_index]
					self.m_target_action_seq_list.append(target_subseq)

					self.m_input_seq_idx_list.append(action_index)

				if action_index > window_size:
					subseq = action_seq_arr[action_index-window_size:action_index]
					action_list_sub_seq.append(subseq)
					actionNum_list_sub_seq.append(window_size)
					# self.m_input_subseq_action_list.append(input_sub_seq)
					# self.m_input_subseq_seqlen_list.append(action_index)
					subseq_num += 1

					for cate in cate_action_list_map_user:
						subseq_cate = cate_action_list_map_user[cate]
						actionNum_subseq_cate = len(subseq_cate)

						action_list_sub_seq.append(subseq_cate)
						actionNum_list_sub_seq.append(actionNum_subseq_cate)
						# self.m_input_subseq_action_list.append(input_sub_seq_cate)
						# self.m_input_subseq_seqlen_list.append(cate_action_user_num)
						subseq_num += 1

					self.m_input_subseq_list_seq_list.append(action_list_sub_seq)
					self.m_input_subseqLen_list_seq_list.append(actionNum_list_sub_seq)

					self.m_input_subseqNum_seq_list.append(subseq_num)

					target_subseq = action_seq_arr[action_index]
					self.m_target_action_seq_list.append(target_subseq)

					self.m_input_seq_idx_list.append(action_index)
		
				cate_cur = cate_seq_arr[action_index]
				if cate_cur not in self.m_catemap:
					cate_id_cur = len(self.m_catemap)
					self.m_catemap[cate_cur] = cate_id_cur

				if cate_cur not in cate_action_list_map_user:
					cate_action_list_map_user[cate_cur] = []

				cate_action_list_map_user[cate_cur].append(item_cur)

		print("subseq num", len(self.m_input_subseq_list_seq_list))
		print("seq idx num", len(self.m_input_seq_idx_list))

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
		temp = list(zip(self.m_dataset.m_input_subseq_list_seq_list, self.m_dataset.m_input_subseqLen_list_seq_list, self.m_dataset.m_input_subseqNum_seq_list, self.m_dataset.m_target_action_seq_list, self.m_dataset.m_input_seq_idx_list))
		random.shuffle(temp)
		
		input_subseq_list_seq_list, input_subseqLen_list_seq_list, input_subseqNum_seq_list, target_action_seq_list, input_seq_idx_list = zip(*temp)

		## batchify, subsequences from the same user should be put in the same batch

		batch_size = self.m_batch_size
        
		input_seq_num = len(input_subseqNum_seq_list)
		batch_num = int(input_seq_num/batch_size)

		print("batch_num", batch_num)

		for batch_index in range(batch_num):
			
			print("batch index", batch_index)

			x_batch = []
			y_batch = []
			x_subseq_len_batch = []
			x_subseq_index_batch = []
			x_subseqIndex_seq_batch = []
			idx_batch = []

			subseq_index_acc = 0
			for seq_index_batch in range(batch_size):
				seq_index = batch_index*batch_size+seq_index_batch

				subseq_list_user = input_subseq_list_seq_list[seq_index]
				subseqlen_list_user = input_subseqLen_list_seq_list[seq_index]
				subseqNum_user = input_subseqNum_seq_list[seq_index]

				x = subseq_list_user

				### x_subseqLen: list of len of sub sequence
				x_subseqLen = subseqlen_list_user

				### x_subseqIndex: subsequence index in a batch of subsequence
				x_subseqIndex = [subseq_index_acc+i for i in range(subseqNum_user)]
				y = target_action_seq_list[seq_index]
                
				x_batch += x
				x_subseq_len_batch += x_subseqLen

				x_subseq_index_batch += x_subseqIndex

				#### x_subseqIndex_seq_batch: list of [subsequence index in a batch per sequence]
				x_subseqIndex_seq_batch.append(x_subseqIndex)
				
				y_batch.append(y)
				idx_batch.append(input_seq_idx_list[seq_index])

				subseq_index_acc += subseqNum_user
            
			y_batch = np.array(y_batch)
			x_batch, x_subseq_len_batch, x_subseq_index_batch = self.batchifyData(x_batch, x_subseq_index_batch)

			if len(x_subseq_index_batch) != len(x_batch):
				assert("error batchify subseq")

			x_batch_tensor = torch.LongTensor(x_batch)
			y_batch_tensor = torch.LongTensor(y_batch)
			idx_batch_tensor = torch.LongTensor(idx_batch)
            
			yield x_batch_tensor, y_batch_tensor, x_subseq_len_batch, x_subseq_index_batch, x_subseqIndex_seq_batch, idx_batch_tensor

	def batchifyData(self, input_action_seq_batch, input_seq_index_batch):
		seq_len_batch = [len(seq_i) for seq_i in input_action_seq_batch]
		
		if seq_len_batch != len(input_action_seq_batch):
			assert("error batchify 1")
		
		if seq_len_batch != len(input_seq_index_batch):
			assert("error batchify 2")

		longest_len_batch = max(seq_len_batch)
		batch_size = len(input_action_seq_batch)

		pad_input_action_seq_batch = np.zeros((batch_size, longest_len_batch))
		pad_seq_len_batch = np.zeros(batch_size)

		## key is the original index, value is the sorted index
		pad_input_seq_index_batch = np.zeros(batch_size)

		zip_batch = sorted(zip(seq_len_batch, input_action_seq_batch, input_seq_index_batch), reverse=True)

		# st = datetime.datetime.now()

		for seq_i, (seq_len_i, input_action_seq_i, input_seq_index_i) in enumerate(zip_batch):
			pad_input_action_seq_batch[seq_i, 0:seq_len_i] = input_action_seq_i
			pad_seq_len_batch[seq_i] = seq_len_i

			## this is to recover index of subsequence for next RNN
			pad_input_seq_index_batch[input_seq_index_i] = seq_i
		# print("pad_seq_len_batch", pad_seq_len_batch) 
		
		# et = datetime.datetime.now()
		# print("batchify duration", et-st)

		return pad_input_action_seq_batch, pad_seq_len_batch, pad_input_seq_index_batch
