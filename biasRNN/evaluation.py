import numpy as np
import torch
import dataset
from metric import *
import datetime
import torch.nn.functional as F
import random
from collections import Counter

class Evaluation(object):
    def __init__(self, log, model, loss_func, device, k=20, warm_start=5):
        self.model = model
        self.loss_func = loss_func

        self.topk = k
        self.warm_start = warm_start
        self.m_device = device
        # self.device = torch.device('cuda' if use_cuda else 'cpu')

        self.m_log = log

    def eval(self, eval_data, train_test_flag):
        self.model.eval()

        losses = []
        recalls = []
        mrrs = []
        weights = []

        # losses = None
        # recalls = None
        # mrrs = None

        dataloader = eval_data

        with torch.no_grad():
            total_test_num = []

            for x_short_action_batch, mask_short_action_batch, pad_x_short_actionNum_batch, y_action_batch, y_action_idx_batch, t_y_batch in dataloader:

                if train_test_flag == "train":
                    eval_flag = random.randint(1,101)
                    if eval_flag != 10:
                        continue

                x_short_action_batch = x_short_action_batch.to(self.m_device)
                mask_short_action_batch = mask_short_action_batch.to(self.m_device)
                y_action_batch = y_action_batch.to(self.m_device)
            
                warm_start_mask = (y_action_idx_batch>=self.warm_start)
    
                output_batch = self.model(x_short_action_batch, mask_short_action_batch, pad_x_short_actionNum_batch)

                sampled_logit_batch, sampled_target_batch = self.model.m_ss(output_batch, y_action_batch, None, None, None, None, None, None, "full")

                loss_batch = self.loss_func(sampled_logit_batch, sampled_target_batch)
                losses.append(loss_batch.item())

                # et_2 = datetime.datetime.now()
                # print("duration 2", et_2-et_1)

                # logit_batch = self.model.m_ss.params(output_batch)
                recall_batch, mrr_batch = evaluate(sampled_logit_batch, sampled_target_batch, warm_start_mask, k=self.topk)

                weights.append( int( warm_start_mask.int().sum() ) )
                recalls.append(recall_batch)
                mrrs.append(mrr_batch)

                # et_3 = datetime.datetime.now()
                # print("duration 3", et_3-et_2)

                total_test_num.append(y_action_batch.view(-1).size(0))

        mean_losses = np.mean(losses)
        mean_recall = np.average(recalls, weights=weights)
        mean_mrr = np.average(mrrs, weights=weights)

        if self.m_log is not None:
            msg = "total_test_num"+str(np.sum(total_test_num))
            self.m_log.addOutput2IO(msg)

        return mean_losses, mean_recall, mean_mrr

    # def set_bucket4item(self, data):
    #     item_freq_dict = dict(Counter(data.m_y_action))
    #     print(len(item_freq_dict))
    #     sorted_item_freq_dict = {k:v for k, v in sorted(item_freq_dict.items(), key=lambda x: x[1], reverse=True)}
    #     # ratio_list = [0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
    #     ratio_list = [0, 0.1, 0.25, 0.4, 0.55, 0.7, 0.85, 1.0]
    #     print("ratio num", len(ratio_list))
    #     item_num = len(sorted_item_freq_dict)
    #     id_threshold_list = item_num*np.array(ratio_list)
    #     print("id_threshold_list", id_threshold_list)
    #     itemid_bucketid_dict = {}
    #     bucketid_itemidlist_dict = {}
    #     itemid_list = list(sorted_item_freq_dict.keys())
    #     for i, itemid in enumerate(itemid_list):
    #     #     print(i)
    #         bucketid = 0
    #         if i <= id_threshold_list[1]:
    #             bucketid = 1
    #         elif i <= id_threshold_list[2]:
    #             bucketid = 2
    #         elif i <= id_threshold_list[3]:
    #             bucketid = 3
    #         elif i <= id_threshold_list[4]:
    #             bucketid = 4
    #         elif i <= id_threshold_list[5]:
    #             bucketid = 5
    #         elif i <= id_threshold_list[6]:
    #             bucketid = 6
    #         # elif i <= id_threshold_list[7]:
    #         #     bucketid = 7
    #         else:
    #             bucketid = 7
    #         itemid_bucketid_dict[itemid] = bucketid
    #         if bucketid not in bucketid_itemidlist_dict:
    #             bucketid_itemidlist_dict[bucketid] = []
    #         bucketid_itemidlist_dict[bucketid].append(itemid)
            
    #     print("bucket", len(bucketid_itemidlist_dict), bucketid_itemidlist_dict.keys())
    #     for bucketid in bucketid_itemidlist_dict:
    #         itemid_list_bucket = bucketid_itemidlist_dict[bucketid]
    #         freq_bucket = 0
    #         for itemid in itemid_list_bucket:
    #             freq_bucket += sorted_item_freq_dict[itemid]
    #         print("bucket %d:%d"%(bucketid, freq_bucket))
            
    #     return item_freq_dict, itemid_bucketid_dict, bucketid_itemidlist_dict

    def set_bucket4item(self, data):
        item_freq_dict = dict(Counter(data.m_y_action))
        print(len(item_freq_dict))
        # freq_threshold_list = [0, 50, 100, 150, 200, 250, 300, 350]
        freq_threshold_list = [0, 20, 30, 40, 80, 120, 240, 400]

        itemid_bucketid_dict = {}
        bucketid_itemidlist_dict = {}
        for itemid in item_freq_dict:
            i = item_freq_dict[itemid]
            bucketid = 0
            if i <= freq_threshold_list[1]:
                bucketid = 1
            elif i <= freq_threshold_list[2]:
                bucketid = 2
            elif i <= freq_threshold_list[3]:
                bucketid = 3
            elif i <= freq_threshold_list[4]:
                bucketid = 4
            elif i <= freq_threshold_list[5]:
                bucketid = 5
            elif i <= freq_threshold_list[6]:
                bucketid = 6
            elif i <= freq_threshold_list[7]:
                bucketid = 7
            else:
                bucketid = 8
            itemid_bucketid_dict[itemid] = bucketid
            if bucketid not in bucketid_itemidlist_dict:
                bucketid_itemidlist_dict[bucketid] = []
            bucketid_itemidlist_dict[bucketid].append(itemid)

        print("bucket", len(bucketid_itemidlist_dict), bucketid_itemidlist_dict.keys())
    #     for bucketid in bucketid_itemidlist_dict:
        for bucketid in range(1, len(bucketid_itemidlist_dict)+1):
            itemid_list_bucket = bucketid_itemidlist_dict[bucketid]
            freq_bucket = 0
            for itemid in itemid_list_bucket:
                freq_bucket += item_freq_dict[itemid]
            print("bucket %d, freq: %d, item num: %d"%(bucketid, freq_bucket, len(itemid_list_bucket)))

        return item_freq_dict, itemid_bucketid_dict, bucketid_itemidlist_dict

    def bias_eval(self, eval_data, itemid_bucketid_dict, train_test_flag):
        self.model.eval()

        losses = []
        recalls = []
        mrrs = []
        weights = []

        # losses = None
        # recalls = None
        # mrrs = None

        dataloader = eval_data
        topk = self.topk

        item_recall_dict = {}
        item_mrr_dict = {}

        with torch.no_grad():
            total_test_num = []

            for x_short_action_batch, mask_short_action_batch, pad_x_short_actionNum_batch, y_action_batch, y_action_idx_batch, t_y_batch in dataloader:

                # batch_item_recall_dict = {}
                # batch_item_mrr_dict = {}

                x_short_action_batch = x_short_action_batch.to(self.m_device)
                mask_short_action_batch = mask_short_action_batch.to(self.m_device)
                y_action_batch = y_action_batch.to(self.m_device)
            
                # warm_start_mask = (y_action_idx_batch>=self.warm_start)
    
                output_batch = self.model(x_short_action_batch, mask_short_action_batch, pad_x_short_actionNum_batch)

                sampled_logit_batch, sampled_target_batch = self.model.m_ss(output_batch, y_action_batch, None, None, None, None, None, None, "full")

                loss_batch = self.loss_func(sampled_logit_batch, sampled_target_batch)
                losses.append(loss_batch.item())

                _, preds = torch.topk(sampled_logit_batch, topk, -1)
                preds = preds.cpu()
                targets = sampled_target_batch.cpu()

                expand_targets = targets.view(-1, 1).expand_as(preds)
                hits = (preds == expand_targets)

                for i, hit in enumerate(hits):
                    target_i = targets[i]
                    itemid_i = target_i.item()
                    
                    rank = hit.nonzero()
                    if itemid_i not in item_recall_dict:
                        item_recall_dict[itemid_i] = []
                        item_mrr_dict[itemid_i] = []
                        
                    if len(rank) == 1:
                        item_recall_dict[itemid_i].append(1.0)
                        rank = rank[0]+1.0
                        rank = torch.reciprocal(rank.float())
                        item_mrr_dict[itemid_i].append(rank.item())
                    else:
                        item_recall_dict[itemid_i].append(0.0)
                        item_mrr_dict[itemid_i].append(0.0)

                total_test_num.append(y_action_batch.view(-1).size(0))
                
        bucket_recall_dict = {}
        bucket_mrr_dict = {}
        for item in item_recall_dict:
            bucketid = itemid_bucketid_dict[item]
            item_recall = np.mean(item_recall_dict[item])
            
            if bucketid not in bucket_recall_dict:
                bucket_recall_dict[bucketid] = []
                bucket_mrr_dict[bucketid] = []
            bucket_recall_dict[bucketid].append(item_recall)
            
            item_mrr = np.mean(item_mrr_dict[item])
            bucket_mrr_dict[bucketid].append(item_mrr)

        for bucketid in bucket_recall_dict:
            recall_list = bucket_recall_dict[bucketid]
            mean_recall = np.mean(recall_list)
            bucket_recall_dict[bucketid] = mean_recall

        for bucketid in bucket_mrr_dict:
            mrr_list = bucket_mrr_dict[bucketid]
            mean_mrr = np.mean(mrr_list)
            bucket_mrr_dict[bucketid] = mean_mrr  

        sorted_bucket_recall_dict = {k:v for k, v in sorted(bucket_recall_dict.items(), key=lambda x: x[0], reverse=True)}
        sorted_bucket_mrr_dict = {k:v for k, v in sorted(bucket_mrr_dict.items(), key=lambda x: x[0], reverse=True)}

        print("---"*10+"recall"+"---"*10)
        recall_list = []
        for k in sorted_bucket_recall_dict:
            recall = sorted_bucket_recall_dict[k]
            recall_list.append(recall)
            print("%d:%.4f"%(k, recall), end=", ")
        print()
        for i in recall_list:
            print("%.4f"%i, end=", ")
        print()
        
        print("---"*10+"mrr"+"---"*10)
        mrr_list = []
        for k in sorted_bucket_mrr_dict:
            mrr = sorted_bucket_mrr_dict[k]
            mrr_list.append(mrr)
            print("%d:%.4f"%(k, mrr), end=", ")
        print()
        for i in mrr_list:
            print("%.4f"%i, end=", ")
        print()


