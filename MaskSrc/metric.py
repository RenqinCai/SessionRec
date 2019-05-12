import torch


def get_recall(indices, targets, mask):
    """
    Calculates the recall score for the given predictions and targets

    Args:
        indices (Bxk): torch.LongTensor. top-k indices predicted by the model.
        targets (B): torch.LongTensor. actual target indices.

    Returns:
        recall (float): the recall score
    """

    targets = targets.view(-1, 1).expand_as(indices)
    hits = (targets == indices)

    mask = mask.unsqueeze(1)
    masked_hits = hits*mask.float()
    masked_hits = masked_hits.nonzero()

    if len(masked_hits) == 0:
        return 0
    n_hits = masked_hits[:, :-1].size(0)
    recall = float(n_hits) / mask.sum().item()
    return recall


def get_mrr(indices, targets, mask):
    """
    Calculates the MRR score for the given predictions and targets
    Args:
        indices (Bxk): torch.LongTensor. top-k indices predicted by the model.
        targets (B): torch.LongTensor. actual target indices.

    Returns:
        mrr (float): the mrr score
    """

    tmp = targets.view(-1, 1)
    targets = tmp.expand_as(indices)
    hits = (targets == indices)

    mask = mask.unsqueeze(1)

    masked_hits = hits*mask.float()

    masked_hits = masked_hits.nonzero()
    ranks = masked_hits[:, -1] + 1
    ranks = ranks.float()

    rranks = torch.reciprocal(ranks)
    mrr = torch.sum(rranks).data / mask.sum().item()
    return mrr


def evaluate(indices, targets, mask, k=20):
    """
    Evaluates the model using Recall@K, MRR@K scores.

    Args:
        logits (B,C): torch.LongTensor. The predicted logit for the next items.
        targets (B): torch.LongTensor. actual target indices.

    Returns:
        recall (float): the recall score
        mrr (float): the mrr score
    """

    _, indices = torch.topk(indices, k, -1)

    # print("topK", _)
    # print("predict top k", indices)
    # print("true target", targets)
    recall = get_recall(indices, targets, mask)
    mrr = get_mrr(indices, targets, mask)
    return recall, mrr
