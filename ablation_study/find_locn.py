import time as tm
from scipy.spatial import distance
import numpy as np

def locn_acc(emb,acnum,test_data):
    count = 0
    start = tm.time()
    print(test_data.shape)
    for d in range(test_data.shape[0]):
        test = test_data[d]
        user = test[0]
        time = test[1]
        dist = []
    
        for i in range(min(test_data[:,2]),max(test_data[:,2])+1):
            summ = (1 - distance.cosine(emb[0][user],emb[2][i]))+(1 - distance.cosine(emb[1][time],emb[2][i]))
            dist.append((abs(summ),i))
        dist = sorted(dist, key=lambda student: student[0])
        if test[2] in (np.array(dist[0:acnum])[:,1]):
            count+=1
        print("d ",d)
    ending = tm.time()
    acc = count/test_data.shape[0]
    time_lapsed = ending-start
    return acc, time_lapsed


#location prediction on the learnt embeddings
#selected_checkins (4 columns): user_index, time (hour in a week), venue_index, venue_category_index
#import time as tm
#from scipy.spatial import distance

#embed = np.load('lbsnresult/model_100/embeddings.npy')
#count = 0

#start = tm.time()
#for d in range(test_data.shape[0]):
#    test = test_data[d]
#    user = test[0]
#    time = test[1]
#    dist = []
    
#    for i in range(min(test_data[:,2]),max(test_data[:,2])+1):
#        summ = (1 - distance.cosine(embed[0][user],embed[2][i]))+(1 - distance.cosine(embed[1][time],embed[2][i]))
#        dist.append((abs(summ),i))
        
#    dist = sorted(dist, key=lambda student: student[0])
#    if test[2] in (np.array(dist[0:20])[:,1]):
#        count+=1
        
#ending = tm.time()

#print("this is count ",count)
#print("this is count fraction ",count/test_data.shape[0])
#print("time lapsed ",ending-start)
