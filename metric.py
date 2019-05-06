import torch

def get_recall(indices, targets):

	targets = targets.view(-1, 1).expand_as(indices)
	hits = (targets == indices).nonzero()

	if len(hits) == 0:
		# print(hits)
		return 0

	n_hits = (targets==indices).nonzero()[:, :-1].size(0)
	recall = float(n_hits)/targets.size(0)
	return recall

def get_mrr(indices, targets):

	tmp = targets.view(-1, 1)
	targets = tmp.expand_as(indices)
	hits = (targets==indices).nonzero()
	# print("hits")
	ranks = hits[:, -1] + 1
	ranks = ranks.float()

	# print(ranks)

	rranks = torch.reciprocal(ranks)
	# print("rranks", rranks)
	mrr = torch.sum(rranks).data/targets.size(0)
	# print("mrr", mrr)
	return mrr

def evaluate(indices, targets, k=20):
	_, indices = torch.topk(indices, k, -1)
	# print(indices.size())
	recall = get_recall(indices, targets)
	mrr = get_mrr(indices, targets)

	return recall, mrr
