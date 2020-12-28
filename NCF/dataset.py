import pandas as pd
import numpy as np
import torch
import random
import pickle

from torch.utils import data
from copy import deepcopy


def preprocess(data_file, num_negatives, batch_size):
	### binarize

	data_pd = data_file 
	data_pd = pd.read_csv(data_file, sep='::', header=None, skiprows=1, names = ["uid", "iid", "rating", "timestamp"])

	user_id = data_pd[['uid']].drop_duplicates().reindex()
	user_id['userId'] = np.arange(len(user_id))

	data_pd = pd.merge(user_id, data_pd, on=['uid'], how='left')

	item_id = data_pd[['iid']].drop_duplicates().reindex()
	item_id['itemId'] = np.arange(len(item_id))

	data_pd = pd.merge(item_id, data_pd, on=['iid'], how='left')

	data_pd = data_pd[['userId', 'itemId', 'rating', 'timestamp']]
	### create negative item samples
	binarized_data_pd = binarize(data_pd)

	user_pool = set(data_pd['userId'].unique())
	item_pool = set(data_pd['itemId'].unique())

	negatives = sampleNegative(data_pd, item_pool)

	train_data, test_data = splitLOO(binarized_data_pd)
	### split train_test

	users, items, ratings = [], [], []

	train_data = pd.merge(train_data, negatives[['userId', 'negative_items']], on='userId')
	train_data['negatives'] = train_data['negative_items'].apply(lambda x: random.sample(x, num_negatives))

	for row in train_data.itertuples():
		users.append(int(row.userId))
		items.append(int(row.itemId))
		ratings.append(float(row.rating))

		for i in range(num_negatives):
			users.append(int(row.userId))
			items.append(int(row.negatives[i]))
			ratings.append(float(0))

	train_dataset = UserItemDataset(user_tensor=torch.LongTensor(users), item_tensor=torch.LongTensor(items), target_tensor=torch.FloatTensor(ratings))

	test_data = pd.merge(test_data, negatives[['userId', 'negative_samples']], on='userId')

	test_dataset = EvalDataset(test_data)

	return train_dataset, test_dataset, user_pool, item_pool

	# test_data = pd.merge(test_data, negatives[['userId', 'negative_samples']], on='userId')
	# users, items, ratings = [], [], []

	# for row in test_data.itertuples():
	# 	users.append(int(row.userId))
	# 	items.append(int(row.itemId))

	# 	ratings.append(float(row.rating))

	# 	for i in range(len(row.negative_samples)):
	# 		users.append(int(row.userId))
	# 		items.append(int(row.negative_samples[i]))
	# 		ratings.append(float(0))

	# test_dataset = UserItemDataset(user_tensor=torch.LongTensor(users), item_tensor=torch.LongTensor(items), target_tensor=torch.FloatTensor(ratings))

	# return DataLoader(train_dataset, batch_size=batch_size, shuffle=True), DataLoader(test_dataset, batch_size=user_tensor.size(0), shuffle=True)
	### return item dataset

def binarize(data_pd):
	data_pd_copy = deepcopy(data_pd)

	data_pd_copy['rating'][data_pd_copy['rating'] > 0] = 1.0

	return data_pd_copy

def sampleNegative(data_pd, item_pool):
	interact_status = data_pd.groupby('userId')['itemId'].apply(set).reset_index().rename(columns={'itemId':'interacted_items'})

	# print("item pool", item_pool) 
	# print("interacted_items", interact_status['interacted_items'])
	interact_status['negative_items'] = interact_status['interacted_items'].apply(lambda x: item_pool-x)
	interact_status['negative_samples'] = interact_status['negative_items'].apply(lambda x: random.sample(x, 99))

	return interact_status[['userId', 'negative_items', 'negative_samples']]

def splitLOO(data_pd):
	data_pd['rank_latest'] = data_pd.groupby(['userId'])['timestamp'].rank(method='first', ascending=False)
	test = data_pd[data_pd['rank_latest'] == 1]
	train = data_pd[data_pd['rank_latest'] > 1]

	return train[['userId', 'itemId', 'rating']], test[['userId', 'itemId', 'rating']]

class EvalDataset(data.Dataset):
	def __init__(self, eval_data):
		self.m_eval_data = eval_data

	def __len__(self):
		return len(self.m_eval_data)

	def __getitem__(self, index):
		eval_user = []
		eval_item = []
		eval_target = []

		row = self.m_eval_data.iloc[index]

		eval_user.append(row.userId)
		eval_item.append(row.itemId)
		eval_target.append(float(1))

		for i in range(len(row.negative_samples)):
			eval_user.append(row.userId)
			eval_item.append(row.negative_samples[i])
			eval_target.append(float(0))

		return torch.LongTensor(eval_user), torch.LongTensor(eval_item), torch.LongTensor(eval_target)

class UserItemDataset(data.Dataset):
	def __init__(self, user_tensor, item_tensor, target_tensor):

		self.user_tensor = user_tensor
		self.item_tensor = item_tensor
		self.target_tensor = target_tensor

	def __getitem__(self, index):
		return self.user_tensor[index], self.item_tensor[index], self.target_tensor[index]

	def __len__(self):
		return self.user_tensor.size(0)





