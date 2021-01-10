IFS=","
while read acc embedding_size hidden_size epochs bs learning_rate n_sam
do
    echo Printing no comma $acc $embedding_size $hidden_size $epochs $bs $learning_rate $n_sam
    python hypergraph_embedding.py --data_path lbsndata --save_path lbsn_ablation -s $embedding_size --hidden_size $hidden_size -e $epochs -b $bs -lr $learning_rate -neg $n_sam -acnum $acc
done < dhneparams.csv

