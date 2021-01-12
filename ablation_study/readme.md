This is an ablation study on using DHNE for location prediction.

The data is in the following order :- emb_size learning_rate hidden_size epochs batch_size n_sam acc
1. embedding size: 50, 100
2. learning_rate: 0.1, 0.001
2. hidden_size: 16, 64 
3. epochs: 10, 25, 50 
4. batch size: 64, 128
5. negative_samples: 5, 10
6. accuracy: 10, 20

The resultant folders are named as follows:
model_(embedding_size)\_(hidden_size)\_(epochs)\_(batch_size)\_(learning_rate)\_(negative_samples)
