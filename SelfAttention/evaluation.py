import numpy as np
import torch
import dataset
from metric import *
from collections import defaultdict
import matplotlib
import matplotlib.pyplot as plt

class Evaluation(object):
    def __init__(self, model, loss_func, use_cuda, k=20, warm_start=5):
        self.model = model
        self.loss_func = loss_func

        self.warm_start = warm_start
        self.topk = k
        self.device = torch.device('cuda' if use_cuda else 'cpu')

    def eval(self, eval_data, batch_size, popularity=None, debug=False):
        self.model.eval()

        losses = []
        recalls = []
        mrrs = []
        weights = []

        if popularity == None:
            item_popularity = defaultdict(int)  
        else:
            item_popularity = popularity
        item_recalls = defaultdict(list)
        item_mrrs = defaultdict(list)

        dataloader = eval_data

        eval_iter = 0

        with torch.no_grad():
            total_test_num = []
            for input_x_batch, target_y_batch, idx_batch in dataloader:
                input_x_batch = input_x_batch.to(self.device)
                target_y_batch = target_y_batch.to(self.device)
                warm_start_mask = (idx_batch>=self.warm_start).to(self.device)

                # Find popularity based on number of interactions in data
                if popularity == None: 
                    for seq in input_x_batch:
                        for item in seq:
                            item_popularity[item.item()] += 1
                    for item in target_y_batch:
                        item_popularity[item.item()] += 1

                logit_batch = self.model(input_x_batch)
                logit_sampled_batch = logit_batch[:, target_y_batch.view(-1)]

                loss_batch = self.loss_func(logit_sampled_batch, target_y_batch)

                losses.append(loss_batch.item())

                recall_batch, item_rec, mrr_batch, item_mrr = evaluate(logit_batch, target_y_batch, warm_start_mask, k=self.topk, debug=debug)

                for k, v in item_rec.items():
                    item_recalls[k].append(v)

                for k, v in item_mrr.items():
                    item_mrrs[k].append(v)

                weights.append( int( warm_start_mask.int().sum() ) )
                recalls.append(recall_batch)
                mrrs.append(mrr_batch)

                total_test_num.append(target_y_batch.view(-1).size(0))

        for k in item_popularity.keys():
            if k not in item_recalls:
                item_recalls[k].append(0)
            if k not in item_mrrs:
                item_mrrs[k].append(0)

        for k, v in item_recalls.items():
            item_recalls[k] = np.mean(v)

        for k, v in item_mrrs.items():
            item_mrrs[k] = np.mean(v)

        recall_popularities = defaultdict(list)
        mrr_popularities = defaultdict(list)

        for k, v in item_popularity.items():
            if v < 100:
                recall_popularities[v].append(item_recalls[k])
                mrr_popularities[v].append(item_mrrs[k])

        for v in recall_popularities.keys():
            recall_vals = recall_popularities[v]
            mrr_vals = mrr_popularities[v]
            recall_popularities[v] = np.mean(recall_vals)
            mrr_popularities[v] = np.mean(mrr_vals)

        recall_fig, ax = plt.subplots()
        ax.bar(list(recall_popularities.keys()), list(recall_popularities.values()))
        ax.set(xlabel='popularity', ylabel='Recall@5',
            title='Recall of items with varying popularity (Xing Self-Attention)')
        ax.grid()
        recall_fig.savefig("recall_popularity_train.png") if popularity == None else recall_fig.savefig("recall_popularity_test.png")
        plt.show()

        mrr_fig, axm = plt.subplots()
        axm.bar(list(mrr_popularities.keys()), list(mrr_popularities.values()))
        axm.set(xlabel='popularity', ylabel='MRR@5',
            title='MRR of items with varying popularity (Xing Self-Attention)')
        axm.grid()
        mrr_fig.savefig("mrr_popularity_train.png") if popularity == None else mrr_fig.savefig("mrr_popularity_test.png")
        plt.show()

        mean_losses = np.mean(losses)
        mean_recall = np.average(recalls, weights=weights)
        mean_mrr = np.average(mrrs, weights=weights)
#         print(recalls, mrrs, weights)
        return mean_losses, mean_recall, mean_mrr, item_popularity
