#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import scipy.io
mat = scipy.io.loadmat('dataset_connected_NYC.mat') 


# In[ ]:


friendship_new = mat['friendship_new']
friendship_old = mat['friendship_old']#new_friendship_old
selected_checkins = mat['selected_checkins'] #new_checkins
selected_users_IDs = mat['selected_users_IDs'] #new_users_IDs


# In[ ]:


#trimming dataset


import random

qty = 1000;

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

print(selected_checkins.shape)
print(friendship_old.shape)
print(selected_users_IDs.shape)


# In[ ]:


# 1. rebuild node index
offset1 = max(selected_checkins[:,0]);
dumy, dumy, n = np.unique(selected_checkins[:,1],return_index=True,return_inverse=True, axis=0)


selected_checkins[:,1] = n+1#+offset1; #n is the indices of the unique values in selected_checkins[:,1]
offset2 = max(selected_checkins[:,1]);
dumy, dumy, n = np.unique(selected_checkins[:,2],return_index=True,return_inverse=True, axis=0)

selected_checkins[:,2] = n+1#+offset2;
offset3 = max(selected_checkins[:,2]);
dumy, dumy, n = np.unique(selected_checkins[:,3],return_index=True,return_inverse=True, axis=0)
selected_checkins[:,3] = n+1#+offset3;

num_node_total = max(map(max, selected_checkins)) #max of the entire matrix = 8117


# In[ ]:


#verify sequence
selected_checkins-=1
print(selected_checkins.shape)
print(min(selected_checkins[:,0]))
print(max(selected_checkins[:,0]))
print(min(selected_checkins[:,1]))
print(max(selected_checkins[:,1]))
print(min(selected_checkins[:,2]))
print(max(selected_checkins[:,2]))
print(min(selected_checkins[:,3]))
print(max(selected_checkins[:,3]))


# In[ ]:


num_of_users = max(selected_checkins[:,0])
num_of_time = max(selected_checkins[:,1])
num_of_venues = max(selected_checkins[:,2])
num_of_categories = max(selected_checkins[:,3])


# In[ ]:


print(num_of_users)
print(num_of_time)
print(num_of_venues)
print(num_of_categories)


# In[ ]:


#separating training and testing data
train_size = int((80/100)*np.size(selected_checkins,0))
train_data = selected_checkins[:train_size]
nums_type = np.array([num_of_users+1, num_of_time+1, num_of_venues+1, num_of_categories+1])
test_data = selected_checkins[train_size:]
#data = {'train_data':train_data,'test_data':test_data,'num_types':num_types}


# In[ ]:


np.savez_compressed('lbsndata/train_data.npz', train_data = train_data, nums_type = nums_type)
np.savez_compressed('lbsndata/test_data.npz', test_data = test_data, nums_type = nums_type)


# In[ ]:


print(train_data.shape)


# In[ ]:


#selected_checkins format -
#selected_checkins (4 columns): user_index, time (hour in a week), venue_index, venue_category_index
#feeding it as it is to DHNE

#run hypergraph_embedding.py --data_path lbsndata --save_path lbsnresult -s '10 10 10 10' -e 2
get_ipython().system('bash run_dhne.sh')


# In[ ]:




