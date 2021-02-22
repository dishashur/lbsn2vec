#!/usr/bin/env python
# coding: utf-8

# In[1]:


#collecting data
import numpy as np
import scipy.io
mat = scipy.io.loadmat('dataset_connected_NYC.mat') #data10while tryin in terminal, start path with Documents


# In[2]:


#friendship_new = mat['friendship_new']
friendship_new = mat['friendship_new']
friendship_old = mat['friendship_old']#new_friendship_old
selected_checkins = mat['selected_checkins'] #new_checkins
selected_users_IDs = mat['selected_users_IDs'] #new_users_IDs

#print(min(map(min, friendship_old)))

print(np.unique(selected_users_IDs).shape)
print(selected_users_IDs.shape)
print(np.unique(selected_checkins[:,0]).shape)
print(np.unique(selected_checkins[:,2]).shape)
print(selected_checkins[:,2].shape)


# In[3]:


qty = 10 #number of entities
num_ckn_wanted = 3 #number of checkin per user from the data


num_usr_chkn = np.zeros((len(selected_users_IDs),2)).astype(np.int64)
print(num_usr_chkn.shape)
for ele in selected_checkins[:,0]:
    num_usr_chkn[ele-1][0]+=1
    num_usr_chkn[ele-1][1]=ele
num_usr_chkn=num_usr_chkn[np.argsort(num_usr_chkn[:,0])]
#taking 50 IDs with num_ckn_wanted checkins
k=qty
new_users_IDs=[]
for i in num_usr_chkn:
    if i[0]==num_ckn_wanted and k>0:
        new_users_IDs.append(i[1])
        k-=1
print(new_users_IDs)
temp=[]
for i in range(len(new_users_IDs)):
    k = np.where(selected_checkins[:,0]==new_users_IDs[i]) 
    #check that the value is less than size of checkins and number of values is = num_ckn_wanted
    for j in k:
        temp.append(selected_checkins[j,:])
selected_checkins = np.array(temp)[:,0,:]
print(len(selected_checkins))


# In[4]:


train_size = int((80/100)*np.size(selected_checkins,0))
test_size = np.size(selected_checkins,0) - train_size
train_checkins = selected_checkins[:train_size]
test_checkins = selected_checkins[train_size:]
orig_checkins = selected_checkins
selected_checkins = train_checkins
print(len(selected_checkins))


# In[5]:


k=np.sort(selected_checkins[:,2])
print(k)
for i in range(len(k)):
    t=np.array(np.where(selected_checkins[:,2]==k[i]))
    for j in range(len(t[0])):
        selected_checkins[t[0][j],2]=i+j+1
print(selected_checkins[:,2])


# In[6]:


chk = np.sort(selected_checkins[:,2])
print(chk)


# In[7]:


print(selected_checkins)


# In[8]:


#swapping the col locations
selected_checkins[:,[0,2]]=selected_checkins[:,[2,0]]
#the configuration of selected_checkin now
#location, hours in the day, users_ID, location category

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


# In[9]:


print(selected_checkins.shape)
print(min(selected_checkins[:,0]))
print(max(selected_checkins[:,0]))
print(min(selected_checkins[:,1]))
print(max(selected_checkins[:,1]))
print(min(selected_checkins[:,2]))
print(max(selected_checkins[:,2]))
print(min(selected_checkins[:,3]))
print(max(selected_checkins[:,3]))


# In[10]:


import random
unique_checkins = np.unique(selected_checkins[:,0])
selected_location = random.sample(list(unique_checkins),len(unique_checkins))
#random.sample randomly selectes 2nd arg number of elements from 1st arg
#selecting all locations so that no loss of information occurs
selected_location = np.array(selected_location) #for function compatibility
print(len(selected_location))
print(len(selected_checkins[:,0]))


# In[11]:


#uncomment for debugging
#print(selected_location)
#print(selected_checkins[:,2])


# In[12]:


# 2. prepare checkins per location (fast)
#user_chechkins is a cell in MATLAB; using Python List for the same
user_checkins = [[] for _ in range(selected_location.shape[0])]
user_checkins_counter=np.zeros((len(selected_location),1)).astype(np.int64)
ind = selected_checkins[:,0].argsort(axis=0)
temp_checkins = np.array([selected_checkins[i,:] for i in ind]) #will not exactly be similar to MATLAB op
u,m,n = np.unique(temp_checkins[:,0],return_index=True,return_inverse=True, axis=0)
m=m.reshape(-1,1)
u=np.array(u)
u= u-min(u)
counters = np.vstack((m[1:],temp_checkins.shape[0]))-m #+1 is not there because indexing starts from 0
ini_val=0 #min(selected_location)
for i in range(0,u.shape[0]):
    user_checkins[u[i]-1].append(temp_checkins[ini_val:ini_val+counters[i,0],:])
    #print(i,"here1")
    user_checkins[u[i]-1]=np.array(user_checkins[u[i]-1])[0,:,:].T
    #print(i,"here2")
    user_checkins[u[i]-1].astype(np.int64)
    ini_val=ini_val+counters[i,0]
    user_checkins_counter[u[i]-1]=counters[i].astype(np.int64)
    #print(i,"here3")


# In[13]:


print(len(user_checkins))


# In[14]:


#print("user_checkins",user_checkins)
print("user_checkins_counter",user_checkins_counter)


# In[15]:


#trial for making walks

#selected_location = list(selected_location)
#print(selected_location)
#k = selected_location.pop(0)
#print(selected_location)
#selected_location.insert(0,k)
#print(selected_location)


# In[16]:


# let's have a walk over all th locations
import random

selected_location = list(selected_location)
num_node = len(selected_location)
num_walk = 10
len_walk = num_node  #has to be smaller than len(selected_location)
print("selected_location length",len(selected_location))
walks = np.zeros((num_walk*num_node,len_walk),dtype = np.int64);
for ww in range(num_walk):
    for ii in range(num_node):
        seq = np.zeros((len_walk,),dtype=int)
        seq[0] = selected_location[ii]
        ins_later = selected_location.pop(ii)
        seq[1:]= np.random.choice(selected_location,len_walk-1,replace=False)  #selected nodes from selected_location without repetition
        selected_location.insert(ii,ins_later)
        walks[ii+(ww-1)*num_node,:] = seq
        
#preprocessing for removing 0 quantities
for i in range(walks.shape[0]):
    for j in range(walks.shape[1]):
        walks[i,j] = walks[i,j] +1
#print(min(map(min, walks))) #should be 1 and 4024


#saving these numbers in walk are creating out of index problem for embedding.py 
walks = walks - min(map(min, walks))

print(walks)


# In[17]:


# 4. prepare negative sample table in advance (fast)
# location relationship
from collections import Counter

temptab = Counter(selected_checkins[:,0]) #len(temptab) = size(tab_degree,1) #Counter gives unique elements in r and their freq
print("temptab ",temptab,len(temptab))
tab2= [[] for _ in range(len(temptab))]

ind=0
for i in temptab:
    #print(i,temptab[i])
    tab2[ind].append(temptab[i])
tab2=np.array(tab2[:][0]) #when printing temptab, not in asc. order but here it becomes asc. wrt keys

tab1=np.unique(selected_checkins[:,0])
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


# In[18]:


#uncomment for debugging
#trii = Counter(neg_sam_table_social)
#print(trii)


# In[19]:


#selected_checkins (4 columns): user_index, time (hour in a week), venue_index, venue_category_index

neg_sam_table_mobility_norm = [[] for _ in range(4)]
for ii in range(len(neg_sam_table_mobility_norm)):
    tab1=np.array([elem for elem in range(max(temp_checkins[:,ii]))])#tab1 has all elements;even with frequency 0
    #temp_checkins is basically array version of selected_checkins
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

#neg_sam_table_mobility1 = neg_sam_table_mobility_norm[3][0]
#neg_sam_table_mobility1[5]
print(neg_sam_table_mobility_norm[0][0])
print(neg_sam_table_mobility_norm[1][0])
print(neg_sam_table_mobility_norm[2][0])
print(neg_sam_table_mobility_norm[3][0])


# In[20]:


# LBSN2vec
dim_emb = 2 #128
num_epoch = 1
num_threads =  4
K_neg = 3 #10
win_size = 2 #10
learning_rate = 0.001

embs_ini = (np.random.uniform(size=(num_node_total,dim_emb))-0.5)/dim_emb 
temp = np.sum(np.power(embs_ini,2),axis=1)
embs_len = np.power(temp,0.5)
den = embs_len
for i in range(dim_emb-1):
    den=np.vstack((den,embs_len))
den=den.T
#print(den.shape)
#print(embs_ini.shape)
embs_ini = np.divide(embs_ini,den)

mobility_ratio = 0.2


# In[21]:


import learn_LBSN2Vec_embedding

embs = learn_LBSN2Vec_embedding.driver_fn(walks.T,user_checkins, user_checkins_counter,embs_ini.T,learning_rate, 
                                K_neg,neg_sam_table_social, win_size, neg_sam_table_mobility_norm,
                                num_epoch, mobility_ratio);


# In[22]:


print(embs.shape)
print(np.min(temp_checkins))
print(np.max(temp_checkins))
print(temp_checkins.shape)
print("this is range of user "+ str(min(temp_checkins[:,0])))
print("this is range of user "+ str(max(temp_checkins[:,0])))
print("this is range of time "+ str(min(temp_checkins[:,1])))
print("this is range of time "+ str(max(temp_checkins[:,1])))
print("this is range of venue_index "+ str(min(temp_checkins[:,2])))
print("this is range of venue_index "+ str(max(temp_checkins[:,2])))
print("this is range of venue_category_index "+ str(min(temp_checkins[:,3])))
print("this is range of venue_category_index "+ str(max(temp_checkins[:,3])))


# In[23]:


#normalizing

twodemb = np.zeros((embs.shape[1],2))
for i in range(twodemb.shape[0]):
    twodemb[i,0] = embs[0,i]
    twodemb[i,1] = embs[1,i]

print(twodemb)


# In[24]:


#location prediction on the learnt embeddings
from scipy.spatial import distance
import time

count = 0

start_time = time.time()
for d in range(selected_checkins.shape[0]):
    test = selected_checkins[d]
    user = test[0]
    hour = test[1]
    dist = []
    
    for i in range(min(selected_checkins[:,2]),max(selected_checkins[:,2])+1):
        summ = (1 - distance.cosine(twodemb[user-1,:],twodemb[i-1,:]))+(1 - distance.cosine(twodemb[hour-1,:],twodemb[i-1,:]))
        dist.append((abs(summ),i))
        
    dist = sorted(dist, key=lambda student: student[0])
    if test[3] in (np.array(dist[0:10])[:,1]):
        count+=1
print("this is count "+str(count))
print("this is count fraction "+str(count/selected_checkins.shape[0]))
print("time taken is ",(time.time() - start_time))


# In[ ]:




