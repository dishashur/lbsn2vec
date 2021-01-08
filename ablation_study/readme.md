This is an ablation study on using DHNE for location prediction.

The data is in the following order :-
1. embedding size: 10, 5, 23, 47
2. hidden_size: 8, 32, 16 
3. epochs: 10, 30, 50 
4. batch size: 16, 2, 64
5. learning_rate: 0.01, 0.0001, 0.2
6. negative_samples: 5, 2, 11

The resultant folders are named as follows:
model_(embedding_size)_(hidden_size)_(epochs)_(batch_size)_(learning_rate)_(negative_samples)
