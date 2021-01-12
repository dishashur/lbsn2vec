IFS=","
while read emb_size learning_rate hidden_size epochs batch_size n_sam acc
do
    echo Printing details $emb_size $learning_rate $hidden_size $epochs $batch_size $n_sam $acc
    python hypergraph_embedding.py --data_path lbsndata --save_path lbsn_ablation -s $emb_size --hidden_size $hidden_size -e $epochs -b $batch_size -lr $learning_rate -neg $n_sam -acnum $acc
done < dhneparams.csv

