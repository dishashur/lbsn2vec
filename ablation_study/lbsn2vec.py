import numpy as np
import scipy.io
import random
from scipy.sparse import csr_matrix
from collections import Counter
from scipy.spatial import distance

import learn_LBSN2Vec_embedding

#collecting data

mat = scipy.io.loadmat('dataset_connected_NYC.mat') #data10while tryin in terminal, start path with Documents
#print(mat)


# ## Preprocessing data


#friendship_new = mat['friendship_new']
friendship_new = mat['friendship_new']
friendship_old = mat['friendship_old']#new_friendship_old
selected_checkins = mat['selected_checkins'] #new_checkins
selected_users_IDs = mat['selected_users_IDs'] #new_users_IDs



num_usrs = [10, 50, 100, 500, 1000]

for usr in num_usrs:
    # # Trimming data 
    # ## (Change qty for number of users )
    print("number of users for this run, ",usr)
    qty = usr

    new_users_IDs = selected_users_IDs[:qty];
    new_friendship_old = []
    for i in range(np.size(friendship_old,0)):
        if ((friendship_old[i,0]<=qty) or (friendship_old[i,1]<=qty)):
            new_friendship_old.append(friendship_old[i,:])
    new_friendship_old = np.array(new_friendship_old)
    
    #make sure all qty user ids are accounted for atleast
    #once in the friendship matrix
    #found = np.zeros((qty,))
    #for i in range(qty):
    #    found[i] = i in new_friendship_old
    #temp = np.where(found == 0)[0]; #ensure that all users are accounted for atleast once


    for i in range(np.size(new_friendship_old,0)):
        if new_friendship_old[i,0]>qty:
            new_friendship_old[i,0]=random.randint(1,qty)
        elif new_friendship_old[i,1]>qty:
            new_friendship_old[i,1]=random.randint(1,qty)

    new_checkins = []
    for i in range(np.size(selected_checkins,0)):
        if selected_checkins[i,0]<=qty:
            new_checkins.append(selected_checkins[i,:])

#assigning original variables

    selected_checkins = np.array(new_checkins)
    friendship_old = np.array(new_friendship_old)
    selected_users_IDs = new_users_IDs

#verify

    train_size = int((80/100)*np.size(selected_checkins,0))
    test_size = np.size(selected_checkins,0) - train_size
    train_checkins = selected_checkins[:train_size]
    test_checkins = selected_checkins[train_size:]
    print("selected_checkins.shape",selected_checkins.shape)
    print("np.size(train_checkins,0)",np.size(train_checkins,0))
    print("np.size(test_checkins,0)",np.size(test_checkins,0))

#training the system on train_checkins only
    orig_checkins = selected_checkins
    selected_checkins = train_checkins

# 1. rebuild node index
    offset1 = max(selected_checkins[:,0]);
    dumy, dumy, n = np.unique(selected_checkins[:,1],return_index=True,return_inverse=True, axis=0)


    selected_checkins[:,1] = n+offset1+1; #n is the indices of the unique values in selected_checkins[:,1]
    offset2 = max(selected_checkins[:,1]);
    dumy, dumy, n = np.unique(selected_checkins[:,2],return_index=True,return_inverse=True, axis=0)

    selected_checkins[:,2] = n+offset2+1;
    offset3 = max(selected_checkins[:,2]);
    dumy, dumy, n = np.unique(selected_checkins[:,3],return_index=True,return_inverse=True, axis=0)
    selected_checkins[:,3] = n+offset3+1;

    num_node_total = max(map(max, selected_checkins)) #max of the entire matrix = 8117
    
    
    print("the following should be consecutive")
    print(min(selected_checkins[:,0]))
    print(max(selected_checkins[:,0]))
    print(min(selected_checkins[:,1]))
    print(max(selected_checkins[:,1]))
    print(min(selected_checkins[:,2]))
    print(max(selected_checkins[:,2]))
    print(min(selected_checkins[:,3]))
    print(max(selected_checkins[:,3]))



# 2. prepare checkins per user (fast)
#user_chechkins is a cell in MATLAB; using Python List for the same
    user_checkins = [[] for _ in range(selected_users_IDs.shape[0])]
    user_checkins_counter=np.zeros((len(selected_users_IDs),1)).astype(np.int64)
    ind = selected_checkins[:,0].argsort(axis=0)
    temp_checkins = np.array([selected_checkins[i,:] for i in ind]) #will not exactly be similar to MATLAB op
    u,m,n = np.unique(temp_checkins[:,0],return_index=True,return_inverse=True, axis=0)
    m=m.reshape(-1,1)
    u=np.array(u)
    
    counters = np.vstack((m[1:],temp_checkins.shape[0]))-m #+1 is not there because indexing starts from 0
    ini_val=0
    for i in range(0,u.shape[0]):
        user_checkins[u[i]-1].append(temp_checkins[ini_val:ini_val+counters[i,0],:])
        user_checkins[u[i]-1]=np.array(user_checkins[u[i]-1])[0,:,:].T
        user_checkins[u[i]-1].astype(np.int64)
        ini_val=ini_val+counters[i,0]
        user_checkins_counter[u[i]-1]=counters[i].astype(np.int64)

   
# 3. random walk

    num_node = len(selected_users_IDs)
    
    network=csr_matrix((np.ones((len(friendship_old),)), (friendship_old[:,0]-1, friendship_old[:,1]-1)), shape=(num_node, num_node))
    network=network+network.T
    #print("network ",network) #should not be empty

    node_list=[[] for _ in range(num_node)] #num_node is number of nodes #500 in clipped case
    node_list_len = np.zeros((num_node,));
    num_walk = 10
    len_walk = 80
    (indx,indy) = network.T.nonzero()

    temp, m, n = np.unique(indx,return_index=True,return_inverse=True, axis=0) #check m for debugging
    m=np.array(m).reshape(-1,1)
    node_list_len=np.vstack((m[1:],len(indx))) - m #+1 is not there because indexing starts from 0
    ini_val=0
    temp=np.array(temp)
    node_list_len=node_list_len[:,0] #for indexing

    for i in range(0,temp.shape[0]):
        node_list[temp[i]].append(indy[ini_val:ini_val+node_list_len[i]])
        ini_val=ini_val+node_list_len[i]
    
#node_list_len.shape = np.count_nonzero(node_list_len) should be





# let's have a walk over social network (friendship)

    walks = np.zeros((num_walk*num_node,len_walk),dtype = np.int64);
    for ww in range(num_walk):
        for ii in range(num_node):
            seq = np.zeros((len_walk,),dtype=int)
            seq[0] = ii
            current_e = ii
            for jj in range(len_walk-1):
                rand_ind = random.randint(0, node_list_len[seq[jj]]-1);
                tempvar = node_list[seq[jj]][0] 
                seq[jj+1] = tempvar[rand_ind];
        
            walks[ii+(ww-1)*num_node,:] = seq
        
#preprocessing for removing 0 quantities
    for i in range(walks.shape[0]):
        for j in range(walks.shape[1]):
            walks[i,j] = walks[i,j] +1

# 4. prepare negative sample table in advance (fast)
# social relationship
 
    (dumy,r) = network.nonzero(); #MATLAB gives [y,x]; Python gives (x,y)
    temptab = Counter(r) #len(temptab) = size(tab_degree,1) #Counter gives unique elements in r
    tab2= [[] for _ in range(len(temptab))]

    for i in range(len(temptab)):
        tab2[i].append(temptab[i])
    tab2=np.array(tab2)[:,0] #when printing temptab, not in asc. order but here it becomes asc. wrt keys

    tab1=np.unique(r)
    tab3=[[] for _ in range(len(tab1))]
    tot=sum(tab2)
    for i in range(len(tab1)):
        tab3[i]=round((tab2[i]*100)/tot,4)
    tab_degree=np.stack((tab1,tab2,tab3),axis=1) # equivalent of tab_degree = tabulate(r) in MATLAB

    freq = np.array([round(np.power(i,0.75),4) for i in tab_degree[:,2]] )
    den=float(sum(freq))
    neg_sam_table_social = np.repeat((1+tab_degree[:,0]),np.around(1000000*freq/sum(freq)).astype(np.int64))
    neg_sam_table_social=neg_sam_table_social.astype(np.int64) # unigram with 0.75 power
    #neg_sam_table_social[740] check for debugging
    del temptab,tab1,tab2,tab3,tab_degree,freq,den
    #print("neg_sam_table_social ",neg_sam_table_social)

    neg_sam_table_mobility_norm = [[] for _ in range(4)]
    for ii in range(len(neg_sam_table_mobility_norm)):
        tab1=np.array([elem for elem in range(max(temp_checkins[:,ii]))])#tab1 has all elements;even with frequency 0
        temptab = Counter(temp_checkins[:,ii]) #correct
        tab2= np.zeros((len(tab1),))
        for i in tab1:
            tab2[i]= temptab[i+1] #because in python, index starts from 0
        tab2 = np.array(tab2)
        tot=np.sum(tab2)
        tab3=np.zeros((len(tab1),))
        for i in range(len(tab1)):
            tab3[i]= np.round((tab2[i]*100)/tot,4) if (tot) else 0
        tab3=np.array(tab3)
        tab_degree=np.stack((tab1,tab2,tab3),axis=1)
        freq = np.array([np.round(np.power(i,0.75),4) for i in tab_degree[:,2]])
        den=float(sum(freq))
        ingoes = np.repeat((tab_degree[:,0]+1),np.around(100000*freq/sum(freq)).astype(np.int64))
        neg_sam_table_mobility_norm[ii].append(ingoes)

        del tab1,tab2,i,tab3,tab_degree,freq,ingoes,tot

    neg_sam_table_mobility_norm = [np.array(x).astype(np.int64) for x in neg_sam_table_mobility_norm]





# LBSN2vec
    dim_emb = 2 #128
    num_epoch = 1
    num_threads =  4
    K_neg = 3 # 10
    win_size = 2 #10
    learning_rate = 0.001

    embs_ini = (np.random.uniform(size=(num_node_total,dim_emb))-0.5)/dim_emb 
    temp = np.sum(np.power(embs_ini,2),axis=1)
    embs_len = np.power(temp,0.5)
    den = embs_len
    for i in range(dim_emb-1):
        den=np.vstack((den,embs_len))
    den=den.T

    embs_ini = np.divide(embs_ini,den)

    mobility_ratio = 0.2

    embs = learn_LBSN2Vec_embedding.driver_fn(walks.T,user_checkins, user_checkins_counter,embs_ini.T,learning_rate, K_neg,neg_sam_table_social, win_size, neg_sam_table_mobility_norm, num_epoch, mobility_ratio)




    print("embs.shape",embs.shape)
  
#normalizing

    twodemb = np.zeros((embs.shape[1],2))
    for i in range(twodemb.shape[0]):
        twodemb[i,0] = embs[0,i]
        twodemb[i,1] = embs[1,i]
 
#location prediction on the learnt embeddings
    count = 0

    for d in range(test_checkins.shape[0]):
        test = test_checkins[d]
        user = test[0]
        time = test[1]
        dist = []
    
        for i in range(min(test_checkins[:,3]),max(test_checkins[:,3])+1):
            summ = (1 - distance.cosine(twodemb[user-1,:],twodemb[i-1,:]))+(1 - distance.cosine(twodemb[time-1,:],twodemb[i-1,:]))
            dist.append((abs(summ),i))
        
        dist = sorted(dist, key=lambda student: student[0])
        if test[3] in (np.array(dist[0:10])[:,1]):
            count+=1
    print("this is count "+str(count))
    print("this is count fraction "+str(count/test_checkins.shape[0]))







