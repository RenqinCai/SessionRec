import pandas as pd
import numpy as np
import torch
import datetime
import pickle
import random
from torch.utils import data
import os
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

class MYDATA(object):
	def __init__(self, action_file, cate_file, time_file, valid_start_time, test_start_time, observed_thresh, window_size):
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
		self.m_x_short_action_list_train = []
		self.m_x_short_actionNum_list_train = []
		self.m_x_short_cate_list_train = []

		self.m_x_long_cate_action_list_train = []
		self.m_x_long_cate_list_train = []

		self.m_x_long_cate_actionNum_list_train = []
		self.m_x_long_cateNum_list_train = []

		self.m_y_action_train = []
		self.m_y_cate_train = []

		self.m_y_action_idx_train = []
		
		### test
		self.m_x_short_action_list_test = []
		self.m_x_short_actionNum_list_test = []
		self.m_x_short_cate_list_test = []

		self.m_x_long_cate_action_list_test = []
		self.m_x_long_cate_list_test = []

		self.m_x_long_cate_actionNum_list_test = []
		self.m_x_long_cateNum_list_test = []

		self.m_y_action_test = []
		self.m_y_cate_test = []

		self.m_y_action_idx_test = []

		print("loading item map")
		print("observed_threshold", observed_thresh, window_size)
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

			if actionNum_seq < window_size :
				window_size = actionNum_seq
			
			###{cate:[action list]}
			cate_action_list_map_user = {}

			for action_index in range(actionNum_seq):
				item_cur = action_seq_arr[action_index]
				cate_cur = cate_seq_arr[action_index]
				time_cur = time_seq_arr[action_index]

				if time_cur <= valid_start_time:
					if item_cur not in self.m_itemmap:
						self.m_itemmap[item_cur] = len(self.m_itemmap)
				continue
				if action_index < observed_thresh:
					if cate_cur not in self.m_catemap:
						cate_id_cur = len(self.m_catemap)
						self.m_catemap[cate_cur] = cate_id_cur

					if cate_cur not in cate_action_list_map_user:
						cate_action_list_map_user[cate_cur] = []
					
					cate_action_list_map_user[cate_cur].append(item_cur)
				
					continue

				if time_cur <= valid_start_time:
					self.addItem2train(action_index, window_size, cate_cur, action_seq_arr, cate_seq_arr, cate_action_list_map_user)

				if time_cur > valid_start_time:
					if time_cur <= test_start_time:
						self.addItem2test(action_index, window_size, cate_cur, action_seq_arr, cate_seq_arr, cate_action_list_map_user)

				cate_cur = cate_seq_arr[action_index]
				if cate_cur not in self.m_catemap:
					cate_id_cur = len(self.m_catemap)
					self.m_catemap[cate_cur] = cate_id_cur

				if cate_cur not in cate_action_list_map_user:
					cate_action_list_map_user[cate_cur] = []

				cate_action_list_map_user[cate_cur].append(item_cur)

		print("seq num for training", len(self.m_x_short_action_list_train))
		print("seq num of actions for training", len(self.m_x_short_actionNum_list_train))

		print("seq num for testing", len(self.m_x_short_action_list_test))
		print("seq num of actions for testing", len(self.m_x_short_actionNum_list_test))

		self.train_dataset = MYDATASET(self.m_x_long_cate_action_list_train, self.m_x_long_cate_actionNum_list_train, self.m_x_long_cateNum_list_train, self.m_x_long_cate_list_train, self.m_x_short_action_list_train, self.m_x_short_cate_list_train, self.m_x_short_actionNum_list_train, self.m_y_action_train, self.m_y_cate_train, self.m_y_action_idx_train)

		self.test_dataset = MYDATASET(self.m_x_long_cate_action_list_test, self.m_x_long_cate_actionNum_list_test, self.m_x_long_cateNum_list_test, self.m_x_long_cate_list_test, self.m_x_short_action_list_test, self.m_x_short_cate_list_test, self.m_x_short_actionNum_list_test, self.m_y_action_test, self.m_y_cate_test, self.m_y_action_idx_test)

	def addItem2train(self, action_index, window_size, cate_cur, action_seq_arr, cate_seq_arr, cate_action_list_map_user):
		
		short_seq = None
		short_cate_seq = None

		if action_index <= window_size:
			short_seq = action_seq_arr[:action_index]
			short_cate_seq = cate_seq_arr[:action_index]
		else:
			short_seq = action_seq_arr[action_index-window_size: action_index]
			short_cate_seq = cate_seq_arr[action_index-window_size: action_index]

		self.m_x_short_action_list_train.append(short_seq)
		self.m_x_short_cate_list_train.append(short_cate_seq)

		short_actionNum_seq = len(short_seq)
		self.m_x_short_actionNum_list_train.append(short_actionNum_seq)

		long_cate_num = 0
		long_cate_action_list = []
		long_cate_actionNum_list = []
		long_cate_list = []

		for cate in cate_action_list_map_user:
			long_cate_subseq = cate_action_list_map_user[cate].copy()[-window_size:]
			long_cate_actionNum_subseq = len(long_cate_subseq)

			long_cate_action_list.append(long_cate_subseq)
			long_cate_actionNum_list.append(long_cate_actionNum_subseq)

			long_cate_num += 1

			long_cate_list.append(cate)

		# self.m_x_long_cate_action_list_train.append(long_cate_action_list)
		self.m_x_long_cate_list_train.append(long_cate_list)
		self.m_y_cate_train.append(cate_cur)

		self.m_x_long_cate_action_list_train.append(long_cate_action_list)
		self.m_x_long_cate_actionNum_list_train.append(long_cate_actionNum_list)
		self.m_x_long_cateNum_list_train.append(long_cate_num)

		y_action = action_seq_arr[action_index]
		self.m_y_action_train.append(y_action)
		self.m_y_action_idx_train.append(action_index)

	def addItem2test(self, action_index, window_size, cate_cur, action_seq_arr, cate_seq_arr, cate_action_list_map_user):
		short_seq = None
		short_cate_seq = None

		if action_index <= window_size:
			short_seq = action_seq_arr[:action_index]
			short_cate_seq = cate_seq_arr[:action_index]
		else:
			short_seq = action_seq_arr[action_index-window_size: action_index]
			short_cate_seq = cate_seq_arr[action_index-window_size: action_index]

		self.m_x_short_action_list_test.append(short_seq)
		self.m_x_short_cate_list_test.append(short_cate_seq)

		short_actionNum_seq = len(short_seq)
		self.m_x_short_actionNum_list_test.append(short_actionNum_seq)

		long_cate_num = 0
		long_cate_action_list = []
		long_cate_actionNum_list = []
		long_cate_list = []

		for cate in cate_action_list_map_user:
			long_cate_subseq = cate_action_list_map_user[cate].copy()[-window_size:]
			long_cate_actionNum_subseq = len(long_cate_subseq)

			long_cate_action_list.append(long_cate_subseq)
			long_cate_actionNum_list.append(long_cate_actionNum_subseq)

			long_cate_num += 1

			long_cate_list.append(cate)

		# self.m_x_long_cate_action_list_train.append(long_cate_action_list)
		self.m_x_long_cate_list_test.append(long_cate_list)
		self.m_y_cate_test.append(cate_cur)

		self.m_x_long_cate_action_list_test.append(long_cate_action_list)
		self.m_x_long_cate_actionNum_list_test.append(long_cate_actionNum_list)
		self.m_x_long_cateNum_list_test.append(long_cate_num)

		y_action = action_seq_arr[action_index]
		self.m_y_action_test.append(y_action)
		self.m_y_action_idx_test.append(action_index)

	@property
	def items(self):
		print("item num", len(self.m_itemmap))
		return self.m_itemmap

	@property
	def cates(self):
		print("cate num", len(self.m_catemap))
		return self.m_catemap

	def save2pickle(self, train_test_dataset, batch_size, folder):
		sorted_data = sorted(zip(train_test_dataset.m_x_long_cateNum_list, train_test_dataset.m_x_long_cate_action_list,  train_test_dataset.m_x_long_cate_actionNum_list, train_test_dataset.m_x_long_cate_list, train_test_dataset.m_x_short_action_list, train_test_dataset.m_x_short_cate_list, train_test_dataset.m_x_short_actionNum_list, train_test_dataset.m_y_action, train_test_dataset.m_y_cate, train_test_dataset.m_y_action_idx), reverse=True)

		train_test_dataset.m_x_long_cateNum_list, train_test_dataset.m_x_long_cate_action_list,  train_test_dataset.m_x_long_cate_actionNum_list, train_test_dataset.m_x_long_cate_list, train_test_dataset.m_x_short_action_list, train_test_dataset.m_x_short_cate_list, train_test_dataset.m_x_short_actionNum_list, train_test_dataset.m_y_action, train_test_dataset.m_y_cate, train_test_dataset.m_y_action_idx = zip(*sorted_data)

		x_seq_num = len(train_test_dataset.m_x_long_cateNum_list)
		batch_num = int(x_seq_num/batch_size)

		print("batch num", batch_num)
		file_num = 20
		step_size = int(batch_num/file_num)

		print("file num", file_num)
		print("step size", step_size)

		iter_index = 0
		for step_index in range(0, batch_num, step_size):
			start_batch_index_step = step_index

			x_long_cateNum_list_step = [train_test_dataset.m_x_long_cateNum_list[(i+start_batch_index_step)*batch_size:(i+start_batch_index_step+1)*batch_size] for i in range(step_size)]
			x_long_cate_action_list_step = [train_test_dataset.m_x_long_cate_action_list[(i+start_batch_index_step)*batch_size:(i+start_batch_index_step+1)*batch_size] for i in range(step_size)]
			x_long_cate_actionNum_list_step = [train_test_dataset.m_x_long_cate_actionNum_list[(i+start_batch_index_step)*batch_size:(i+start_batch_index_step+1)*batch_size] for i in range(step_size)]

			x_long_cate_list_step = [train_test_dataset.m_x_long_cate_list[(i+start_batch_index_step)*batch_size:(i+start_batch_index_step+1)*batch_size] for i in range(step_size)]

			x_short_action_list_step = [train_test_dataset.m_x_short_action_list[(i+start_batch_index_step)*batch_size:(i+start_batch_index_step+1)*batch_size] for i in range(step_size)]

			x_short_cate_list_step = [train_test_dataset.m_x_short_cate_list[(i+start_batch_index_step)*batch_size:(i+start_batch_index_step+1)*batch_size] for i in range(step_size)]

			x_short_actionNum_list_step = [train_test_dataset.m_x_short_actionNum_list[(i+start_batch_index_step)*batch_size:(i+start_batch_index_step+1)*batch_size] for i in range(step_size)]

			y_action_step = [train_test_dataset.m_y_action[(i+start_batch_index_step)*batch_size:(i+start_batch_index_step+1)*batch_size] for i in range(step_size)]  

			y_cate_step = [train_test_dataset.m_y_cate[(i+start_batch_index_step)*batch_size:(i+start_batch_index_step+1)*batch_size] for i in range(step_size)]

			y_action_idx_step = [train_test_dataset.m_y_action_idx[(i+start_batch_index_step)*batch_size:(i+start_batch_index_step+1)*batch_size] for i in range(step_size)]

			step_data_map = {"long_cateNum": x_long_cateNum_list_step, "long_action_cate":x_long_cate_action_list_step, "long_actionNum_cate": x_long_cate_actionNum_list_step, "long_cate": x_long_cate_list_step, "short_action": x_short_action_list_step, "short_cate": x_short_cate_list_step, "short_actionNum": x_short_actionNum_list_step, "target_action": y_action_step, "target_cate": y_cate_step, "target_id":y_action_idx_step}

			f_name = "/"+str(iter_index)+".pickle"
			f_name = folder+f_name
			f_step = open(f_name, "wb")

			pickle.dump(step_data_map, f_step)
			f_step.close()

			iter_index += 1

class MYDATASET(object):
	def __init__(self, x_long_cate_action_list, x_long_cate_actionNum_list, x_long_cateNum_list, x_long_cate_list, x_short_action_list, x_short_cate_list, x_short_actionNum_list, y_action, y_cate, y_action_idx):

		self.m_x_long_cate_action_list = x_long_cate_action_list
		self.m_x_long_cate_actionNum_list = x_long_cate_actionNum_list
		self.m_x_long_cateNum_list = x_long_cateNum_list
		self.m_x_long_cate_list = x_long_cate_list

		self.m_x_short_action_list = x_short_action_list
		self.m_x_short_cate_list = x_short_cate_list

		self.m_x_short_actionNum_list = x_short_actionNum_list
		self.m_y_action = y_action
		self.m_y_cate = y_cate

		self.m_y_action_idx = y_action_idx

class Dataset(data.Dataset):
	def __init__(self, folder):
		self.m_folder = folder
		
		self.m_file_num = 0
		for file in os.listdir(self.m_folder):
			if file.endswith(".pickle"):
				self.m_file_num += 1

		print("file num", self.m_file_num)

	def __len__(self):
		return self.m_file_num

	def __getitem__(self, index):
		file_name = self.m_folder+"/"+str(index)+".pickle"
		print("file name", file_name)
		f_step = open(file_name, "rb")

		step_data_map = pickle.load(f_step)
		f_step.close()

		x_long_cate_action_list_step = step_data_map["long_action_cate"]
		x_long_cateNum_list_step = step_data_map["long_cateNum"]
		
		x_long_cate_actionNum_list_step = step_data_map["long_actionNum_cate"]
		x_long_cate_list_step = step_data_map["long_cate"]
		x_short_action_list_step = step_data_map["short_action"]
		x_short_cate_list_step = step_data_map["short_cate"]
		x_short_actionNum_list_step = step_data_map["short_actionNum"]
		y_action_step = step_data_map["target_action"]
		y_cate_step = step_data_map["target_cate"]
		y_action_idx_step = step_data_map["target_id"]

		print("step size", len(x_long_cate_action_list_step))

		return x_long_cate_action_list_step, x_long_cateNum_list_step, x_long_cate_actionNum_list_step, x_long_cate_list_step, x_short_action_list_step, x_short_cate_list_step, x_short_actionNum_list_step, y_action_step, y_cate_step, y_action_idx_step

class MYDATALOADER():
	def __init__(self, train_test_dataset, batch_size):
		params = {'batch_size': 1, "shuffle": True, "num_workers": 5}
		self.m_dataloader = data.DataLoader(train_test_dataset, **params)
		self.m_batch_size = batch_size

	def __iter__(self):
		print("iter")
		for x_long_cate_action_list_step, x_long_cateNum_list_step, x_long_cate_actionNum_list_step, x_long_cate_list_step, x_short_action_list_step, x_short_cate_list_step, x_short_actionNum_list_step, y_action_step, y_cate_step, y_action_idx_step in self.m_dataloader:
			### shuffle
			print("shuffling")
			temp = list(zip(x_short_action_list_step, y_action_step, y_action_idx_step))
			random.shuffle(temp)
			
			x_short_action_list_step, y_action_step, y_action_idx_step = zip(*temp)

			batch_size = self.m_batch_size
			
			input_num = len(x_short_action_list_step)
			print(input_num)
			batch_num = input_num
			# batch_num = int(input_num/batch_size)
			print("batch num in a step", batch_num)
			for batch_index in range(batch_num):
				x_batch = []
				y_batch = []
				idx_batch = []

				for seq_index_batch in range(batch_size):
					seq_index = batch_index*batch_size+seq_index_batch
					x = list(x_short_action_list_step[seq_index])
					y = y_action_step[seq_index]
					
					x_batch.append(x)
					y_batch.append(y)
					idx_batch.append(y_action_idx_step[seq_index])
					print("y batch", y)
				x_batch, y_batch, x_len_batch, idx_batch = self.batchifyData(x_batch, y_batch, idx_batch)

				x_batch_tensor = torch.LongTensor(x_batch)
				y_batch_tensor = torch.LongTensor(y_batch)
				idx_batch_tensor = torch.LongTensor(idx_batch)

				yield x_batch_tensor, y_batch_tensor, idx_batch_tensor
				
	def batchifyData(self, input_action_seq_batch, target_action_seq_batch, idx_batch):
		seq_len_batch = [len(seq_i) for seq_i in input_action_seq_batch]

		longest_len_batch = max(seq_len_batch)
		batch_size = len(input_action_seq_batch)

		pad_input_action_seq_batch = np.zeros((batch_size, longest_len_batch))
		pad_target_action_seq_batch = np.zeros(batch_size)
		pad_seq_len_batch = np.zeros(batch_size)
		pad_idx_batch = np.zeros(batch_size)

		zip_batch = sorted(zip(seq_len_batch, input_action_seq_batch, target_action_seq_batch, idx_batch), reverse=True)

		for seq_i, (seq_len_i, input_action_seq_i, target_action_seq_i, seq_idx) in enumerate(zip_batch):

			pad_input_action_seq_batch[seq_i, 0:seq_len_i] = input_action_seq_i
			pad_target_action_seq_batch[seq_i] = target_action_seq_i
			pad_seq_len_batch[seq_i] = seq_len_i
			pad_idx_batch[seq_i] = seq_idx
            
		return pad_input_action_seq_batch, pad_target_action_seq_batch, pad_seq_len_batch, pad_idx_batch



