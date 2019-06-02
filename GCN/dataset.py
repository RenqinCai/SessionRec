import numpy as np
import pickle as pkl

import networkx as nx
import scipy.sparse as sp

import sys

def parse_index_file(filename):
	index = []
	for line in open(filename):
		index.append(int(line.strip()))

	return index

def sample_mask(idx, l):
	mask = np.zeros(l)

	mask[idx] = 1

	return np.array(mask, dtype=np.bool)
	

def load_data(dataset_str):
	names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
	objects = []

	for i in range(len(names)):
		with open('data/ind.{}.{}'.format(dataset_str, names[i]), 'rb') as f:
			if sys.version_info > (3, 0):
				objects.append(pkl.load(f, encoding='latin1'))
			else:
				objects.append(pkl.load(f))

	x, y, tx, ty, allx, ally, graph = tuple(objects)

	test_idx_reorder = parse_index_file('data/ind.{}.test.index'.format(dataset_str))
	test_idx_range = np.sort(test_idx_reorder)

	if dataset_str == 'citeseer':
		test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)

		tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))

		tx_extended[test_idx_range-min(test_idx_range), :] = tx

		tx = tx_extended

		ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
		ty_extended[test_idx_range-min(test_idx_range), :] = ty
		ty = ty_extended

	features = sp.vstack((allx, tx)).tolil()
	features[test_idx_reorder, :] = features[test_idx_range, :]
	adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
	labels = np.vstack((ally, ty))
	labels[test_idx_reorder, :] = labels[test_idx_range, :]

	idx_test = test_idx_range.tolist()
	idx_train = range(len(y))
	idx_val = range(len(y), len(y)+500)

	train_mask = sample_mask(idx_train, labels.shape[0])
	val_mask = sample_mask(idx_val, labels.shape[0])
	test_mask = sample_mask(idx_test, labels.shape[0])

	y_train = np.zeros(labels.shape)
	y_val = np.zeros(labels.shape)
	y_test = np.zeros(labels.shape)
	y_train[train_mask, :] = labels[train_mask, :]
	y_val[val_mask, :] = labels[val_mask, :]
	y_test[test_mask, :] = labels[test_mask, :]

	return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask
