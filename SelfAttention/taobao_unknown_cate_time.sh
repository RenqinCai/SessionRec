CUDA_VISIBLE_DEVICES=0 python main.py --data_folder ../../../Data/xing/ --data_name xing --num_heads 2 --embedding_dim 64 --hidden_size 64 --lr 0.001 --window_size 32 --test_observed 5 --n_epochs 1 --position_embedding 1 --shared_embedding 1 --batch_size 200 --optimizer_type Adam --loss_type 'BPR' --topk 1 --model_name 'self-attention' 