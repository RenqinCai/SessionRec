def bias_eval(eval_data, itemid_bucketid_dict):
    network.eval()

    losses = []
    recalls = []
    mrrs = []
    weights = []

    dataloader = eval_data
    topk = args.topk
    
    item_recall_dict = {}
    item_mrr_dict = {}
#     # ### load 
    # item_freq_dict = {}

    with torch.no_grad():
        total_test_num = []

        for x_short_action_batch, mask_short_action_batch, pad_x_short_actionNum_batch, y_action_batch, y_action_idx_batch in dataloader:
            
#             eval_flag = random.randint(1,101)
#             if eval_flag != 10:
#                 continue

            batch_item_recall_dict = {}
            batch_item_mrr_dict = {}

            x_short_action_batch = x_short_action_batch.to(device)
            mask_short_action_batch = mask_short_action_batch.to(device)
            y_action_batch = y_action_batch.to(device)

            # warm_start_mask = (y_action_idx_batch>=self.warm_start)

            output_batch = network(x_short_action_batch, mask_short_action_batch, pad_x_short_actionNum_batch)

            sampled_logit_batch, sampled_target_batch = network.m_ss(output_batch, y_action_batch, \
                                                        None, None, None, None, None, None, "full")

            loss_batch = loss_function(sampled_logit_batch, sampled_target_batch)
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
                if itemid_i not in batch_item_recall_dict:
                    batch_item_recall_dict[itemid_i] = []
                    batch_item_mrr_dict[itemid_i] = []
                    
                if len(rank) == 1:
                    batch_item_recall_dict[itemid_i].append(1.0)
                    rank = rank[0]+1.0
                    rank = torch.reciprocal(rank.float())
                    batch_item_mrr_dict[itemid_i].append(rank.item())
                else:
                    batch_item_recall_dict[itemid_i].append(0.0)
                    batch_item_mrr_dict[itemid_i].append(0.0)
                
            for item in batch_item_recall_dict:
                batch_mean_recall = np.mean(batch_item_recall_dict[item])
                batch_mean_mrr = np.mean(batch_item_mrr_dict[item])
                
                if item not in item_recall_dict:
                    item_recall_dict[item] = []
                    item_mrr_dict[item] = []
                item_recall_dict[item].append(batch_mean_recall)
                item_mrr_dict[item].append(batch_mean_mrr)
                
            total_test_num.append(y_action_batch.view(-1).size(0))
    return item_recall_dict, item_mrr_dict
    

bucket_recall_dict = {}
bucket_mrr_dict = {}
for item in item_recall_dict:
    bucketid = train_itemid_bucketid_dict[item]
    item_recall = np.mean(item_recall_dict[item])
    
#     item_recall = item_recall[0]*1.0/item_recall[1]
    if bucketid not in bucket_recall_dict:
        bucket_recall_dict[bucketid] = []
        bucket_mrr_dict[bucketid] = []
    bucket_recall_dict[bucketid].append(item_recall)
    
    item_mrr = np.mean(item_mrr_dict[item])
#     item_mrr = item_mrr[0]*1.0/item_mrr[1]
    bucket_mrr_dict[bucketid].append(item_mrr)

for bucket in bucket_recall_dict:
    recall_list = bucket_recall_dict[bucket]
    mean_recall = np.mean(recall_list)
    bucket_recall_dict[bucket] = mean_recall

for bucket in bucket_mrr_dict:
    mrr_list = bucket_mrr_dict[bucket]
    mean_mrr = np.mean(mrr_list)
    bucket_mrr_dict[bucket] = mean_mrr  