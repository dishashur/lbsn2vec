#change the name of the file here
import numpy as np
data_train = np.load('train_data/train_data.npy')
data_train_type = np.load('train_data/nums_type.npy')
data_test = np.load('test_data/test_data.npy')
data_test_type = np.load('test_data/nums_type.npy')
print("data_train, ", np.shape(data_train))
#print("data_train, ", data_train)
print("data_train_type, ", data_train_type)
print("data_test, ", np.shape(data_test))
#print("data_test, ", data_test)
print("data_test_type, ", data_test_type)

embed = np.load('../../result/GPS/model_2/embeddings.npy')
print("embeddings developed shape, ", np.shape(embed))
print("shapeof each element in embed, ", embed[0].shape, embed[1].shape, embed[2].shape)
print("unique elements in col 1 of test_data, ", np.unique(data_test[:,0]))
print("unique elements in col 2 of test_data, ", np.unique(data_test[:,1]))
print("unique elements in col 3 of test_data, ", np.unique(data_test[:,2]))
print("max from each col of data_train", max(data_train[:,0]),max(data_train[:,1]),max(data_train[:,2]))
print("max from each col of data_test", max(data_test[:,0]),max(data_test[:,1]),max(data_test[:,2]))
#print("embeddings developed, ", embed)

