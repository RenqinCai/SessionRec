import torch
import numpy as np

def evaluate(preds, targets, k=20):
	# print("preds size", preds.size())
	preds = preds.view(-1)
	# print("view preds size", preds.size())
	_, indices = torch.topk(preds, k, 0)

	recall = get_recall(indices, targets)
	mrr = get_mrr(indices, targets)

	return recall, mrr

def get_recall(topk_indices, targets):
	targets = np.zeros(topk_indices.size(0))

	topk_indices = topk_indices.cpu().data.numpy()
	hits = (targets == topk_indices).nonzero()[0]
	# print(hits[0], len(hits[0]))
	return len(hits)

def get_mrr(topk_indices, targets):
	# targets = np.zeros(topk_indices.size(0))

	targets = targets.cpu().data.numpy()

	targets = targets.nonzero()[0]
	topk_indices = topk_indices.cpu().data.numpy()
	hits = (targets == topk_indices).nonzero()[0]

	mrr = 0.0
	# print("hits num", hits)
	if len(hits):
		ranks = hits[0] + 1
		rranks = np.reciprocal(ranks)
		# print("rranks", rranks)
		mrr = rranks/len(targets)
	else:
		mrr = 0.0
	# print("mrr", mrr)
	return mrr