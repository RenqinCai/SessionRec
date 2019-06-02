import numpy as np
import pickle as pkl

import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys


"""
construct feed dict
"""
def construct_feed_dict(features, support, labels, labels_mask, placeholders):
	feed_dict = dict()

	feed_dict.update({placeholders['labels']: labels})
	feed_dict.update({placeholders['labels_mask']:labels_mask})
	feed_dict.update({placeholders['features']:features})

	feed_dict.update({placeholders['support'][i]: support[i] for i in range(len(support))})
	feed_dict.update({placeholders['num_features_nonzero']:features[1].shape})

	return feed_dict

