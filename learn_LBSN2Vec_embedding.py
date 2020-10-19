import numpy as np
from random import *
from decimal import *
getcontext().prec = 6

#defining constants
def ULONG_MAX():
	return(getrandbits(66))

def RAND_MULTIPLIER():
	return(25214903917)

def RAND_INCREMENT():
	return(11)

#def MAX_EXP():
#	return(6)

#def EXP_TABLE_SIZE():
#	return(1000)


################################

def getNextRand(next_random):
    next_random = next_random*RAND_MULTIPLIER() + RAND_INCREMENT();
    return(next_random)

def get_a_social_decision(): #next_random,mobility_ratio):
#we will randompy pick between 0 and 1
    ans = choice([1,0])
    return(ans)

def get_a_mobility_decision(): #next_random,mobility_ratio):
    ans = choice([1,0])
    return(ans)

def get_a_checkin_sample(next_random, table_size):
    return ((next_random >> 16) % table_size);

def get_a_neg_sample(next_random, neg_sam_table, table_size):
    ind = (next_random >> 16) % table_size
    target_n = neg_sam_table[ind]
    return(target_n)

def get_a_neg_sample_Kless1(next_random, num_neg):
    v_rand_uniform = next_random/ULONG_MAX()
    ans = 1
    if (v_rand_uniform>num_neg):
        ans = 0
    return(ans)


def learn_a_pair_loc_pr_cosine(flag, loc1, best_fit, dim_emb, emb_n, alpha):
    f=0
    a=0
    norm1 = np.linalg.norm(emb_n[:,(loc1-1)])
    for d in range(dim_emb):
        f += emb_n[d,(loc1-1)]*best_fit[d]
    a = alpha
    c1 = (1/norm1)*a
    c2 = (f/(norm1*norm1*norm1))*a
    #if norm1 == 0:
    #    print("this is c1 "+str(c1)+" this is c2 "+str(c2))
    if (flag==1):
        for d in range(dim_emb):
            emb_n[d,(loc1-1)] += c1*best_fit[d] - c2*emb_n[d,(loc1-1)]
    else:
        for d in range(dim_emb):
            emb_n[d,(loc1-1)] -= c1*best_fit[d] - c2*emb_n[d,(loc1-1)]
    return(emb_n)    



def learn_an_edge_with_BFT(word, target_e, dim_emb, next_random, best_fit, num_neg, emb_n, alpha, neg_sam_table_social, 								table_size_social):
    loc_w = word
    loc_e = target_e
    for d in range(dim_emb):
        best_fit[d] = emb_n[d,(loc_w-1)] + emb_n[d,(loc_e-1)]
    norm_b = np.linalg.norm(best_fit) 
    for d in range(dim_emb):
        best_fit[d] = best_fit[d]/norm_b
    
    #print("from learn_an_edge_with_BFT 1")
    emb_n = learn_a_pair_loc_pr_cosine(1, loc_w, best_fit, dim_emb, emb_n, alpha)
    #print("from learn_an_edge_with_BFT 2")
    emb_n = learn_a_pair_loc_pr_cosine(1, loc_e, best_fit, dim_emb, emb_n, alpha)
    #print("done bft2") 

    if (num_neg<1):
        next_random = getNextRand(next_random)
        if (get_a_neg_sample_Kless1(next_random, num_neg)==1):
            next_random = getNextRand(next_random)
            target_n = get_a_neg_sample(next_random, neg_sam_table_social, table_size_social)
            if ((target_n != target_e) and (target_n != word)):
                loc_neg = target_n
                #print("from learn_an_edge_with_BFT numneg if")
                emb_n = learn_a_pair_loc_pr_cosine(0, loc_neg, best_fit, dim_emb, emb_n, alpha)
    else:
        for n in range(num_neg):
            next_random = getNextRand(next_random)
            target_n = get_a_neg_sample(next_random, neg_sam_table_social, table_size_social)
            if ((target_n != target_e) and (target_n != word)):
                loc_neg = target_n
                #print("from learn_an_edge_with_BFT numneg else")
                emb_n = learn_a_pair_loc_pr_cosine(0, loc_neg, best_fit, dim_emb, emb_n, alpha)
    return(emb_n)



def learn_a_hyperedge(edge, edge_len, next_random, best_fit, dim_emb, emb_n, alpha, num_neg, neg_sam_table_mobility1, neg_sam_table_mobility2,
                      neg_sam_table_mobility3, neg_sam_table_mobility4, table_size_mobility1, table_size_mobility2, table_size_mobility3,
                      table_size_mobility4):
    #get best-fit-line dim_emb = 128
    for i in range(edge_len):
        loc_n = edge[i] #-1
        norm_e = np.linalg.norm(emb_n[:,(loc_n-1)])
        for d in range(dim_emb):
            best_fit[d] += (emb_n[d,(loc_n-1)]/norm_e)

    #normalize best fit line for fast computation
    norm_b = np.linalg.norm(best_fit)
    for d in range(0,dim_emb):
        best_fit[d] = best_fit[d]/norm_b


	#learn learn learn
    for i in range(edge_len):
        node = edge[i]
        loc_n = node #-1
        #print("from learn_a_hyperedge")
        emb_n = learn_a_pair_loc_pr_cosine(1, loc_n, best_fit, dim_emb, emb_n, alpha) 

        if (num_neg<1): 
            next_random = getNextRand(next_random)
            if (get_a_neg_sample_Kless1(next_random, num_neg)==1):
                next_random = getNextRand(next_random)
                if (i==0):
                    target_neg = get_a_neg_sample(next_random, neg_sam_table_mobility1, table_size_mobility1)
                elif (i==1):
                    target_neg = get_a_neg_sample(next_random, neg_sam_table_mobility2, table_size_mobility2) 
                elif (i==2):
                    target_neg = get_a_neg_sample(next_random, neg_sam_table_mobility3, table_size_mobility3)
                elif (i==3):
                    target_neg = get_a_neg_sample(next_random, neg_sam_table_mobility4, table_size_mobility4)
                if (target_neg != node):
                    loc_neg = target_neg
                    #print("from learn_a_hyperedge numneg if")
                    emb_n = learn_a_pair_loc_pr_cosine(0, loc_neg, best_fit, dim_emb, emb_n, alpha) 
                
            
        else:
            for n in range(num_neg): 
                next_random = getNextRand(next_random)
                if (i==0):
                    target_neg = get_a_neg_sample(next_random, neg_sam_table_mobility1, table_size_mobility1) 
                elif (i==1):
                    target_neg = get_a_neg_sample(next_random, neg_sam_table_mobility2, table_size_mobility2) 
                elif (i==2):
                    target_neg = get_a_neg_sample(next_random, neg_sam_table_mobility3, table_size_mobility3)
                elif (i==3):
                    target_neg = get_a_neg_sample(next_random, neg_sam_table_mobility4, table_size_mobility4)

                if (target_neg != node): 
                    loc_neg = target_neg
                    #print("from learn_a_hyperedge numneg else")
                    emb_n = learn_a_pair_loc_pr_cosine(0, loc_neg, best_fit, dim_emb, emb_n, alpha) 
	
    return(emb_n)             
            
        
def learn(walk,num_w,num_wl,user_checkins,user_checkins_count,emb_n,num_n,dim_emb,starting_alpha,num_neg,
			neg_sam_table_social,table_size_social, win_size, neg_sam_table_mobility1,table_size_mobility1,neg_sam_table_mobility2,
			table_size_mobility2,neg_sam_table_mobility3,table_size_mobility3,neg_sam_table_mobility4,table_size_mobility4,
			num_epoch, mobility_ratio):
    
    best_fit = np.zeros((dim_emb,))
    next_random = getrandbits(32) #equivalent to (long) rand() in C
    edge_len = 4 # here 4 is a checkin node number user-time-POI-category
    ind_start = 0
    ind_end = num_w
    ind_len = num_w
    progress=0
    progress_old=0
    alpha = starting_alpha

    for pp in range(num_epoch):
        counter = 0
        for w in range(0,num_w):
            progress = (pp*num_w + w)/(num_w*num_epoch)
            if (progress-progress_old > 0.001):
                alpha =  min(starting_alpha * (1 - progress),starting_alpha * 0.001)
                progress_old = progress

            for i in range(num_wl):
                word = walk[i,w] #walk.T is sent -> shape = (num_wl x num_w)

                for j in range(1,win_size+1):
                    next_random = getNextRand(next_random)
                    if (get_a_social_decision()==1):
                        if (i-j)>=0: 
                            target_e = walk[(i-j),w]
                            if (word!=target_e):
                                #print("inside learn in line 205:word "+str(word)+" target_e "+str(target_e))
                                emb_n = learn_an_edge_with_BFT(word, target_e, dim_emb, next_random, best_fit, num_neg, emb_n, alpha, 											neg_sam_table_social,table_size_social)                        
                        if (i+j)<num_wl :
                            target_e = walk[(i+j),w]
                            if (word!=target_e):
                                #print("inside learn in line 210:word "+str(word)+" target_e "+str(target_e))
                                emb_n = learn_an_edge_with_BFT(word, target_e, dim_emb, next_random, best_fit, num_neg, emb_n, alpha, 									neg_sam_table_social,table_size_social)
                    

                if user_checkins_count[word-1,0]>0:
                    for m in range(min(win_size*2,user_checkins_count[word-1,0])):     
                        next_random = getNextRand(next_random)
                        if (get_a_mobility_decision()==1):
                            a_user_checkins = user_checkins[word-1] 
                            next_random = getNextRand(next_random)
                            a_checkin_ind = get_a_checkin_sample(next_random, user_checkins_count[word-1,0])
                            edge = a_user_checkins[:,a_checkin_ind]
                            #print("inside learn in line 220:word "+str(word)+" target_e "+str(target_e))
                            emb_n = learn_a_hyperedge(edge, edge_len, next_random, best_fit, dim_emb, emb_n, alpha, num_neg, 
                                                      neg_sam_table_mobility1, neg_sam_table_mobility2, neg_sam_table_mobility3,
                                                      neg_sam_table_mobility4, table_size_mobility1, table_size_mobility2,
                                                      table_size_mobility3, table_size_mobility4)
                print("this is num_wl: "+str(i)+" and num_w "+str(w))
                print(emb_n)

    return(emb_n) 

# main function ---- equivalent to mexFunction

def driver_fn(walks,user_checkins, user_checkins_counter, embs_ini, learning_rate, K_neg,
	 neg_sam_table_social, win_size, neg_sam_table_mobility_norm, num_epoch, mobility_ratio):
    
    walk = walks
    num_w = walks.shape[1] #N = 4024*10 = 40240
    num_wl = walks.shape[0] #M = 80
    print("this is num_w " +str(num_w))
    print("this is num_wl " +str(num_wl))
    print(walk)
    user_checkins = user_checkins
    num_u = len(user_checkins)
    user_checkins_count = user_checkins_counter
    emb_n = embs_ini
    num_n = emb_n.shape[1] #N = 8117
    dim_emb = emb_n.shape[0] #M = 128
    starting_alpha = learning_rate
    num_neg = K_neg # = 10
    neg_sam_table_social = neg_sam_table_social
    table_size_social = neg_sam_table_social.shape[0] #M = 999680
    win_size = win_size
    neg_sam_table_mobility = neg_sam_table_mobility_norm
    table_num_mobility = len(neg_sam_table_mobility_norm)
    if(table_num_mobility != 4):
        print("four negative sample tables are required in neg_sam_table_mobility")
    neg_sam_table_mobility1 = neg_sam_table_mobility[0][0]
    table_size_mobility1 = neg_sam_table_mobility1.shape[0] # = 100502
    neg_sam_table_mobility2 = neg_sam_table_mobility[1][0]
    table_size_mobility2 = neg_sam_table_mobility2.shape[0] # = 100003
    neg_sam_table_mobility3 = neg_sam_table_mobility[2][0]
    table_size_mobility3 = neg_sam_table_mobility3.shape[0] # = 99730
    neg_sam_table_mobility4 = neg_sam_table_mobility[3][0]
    table_size_mobility4 = neg_sam_table_mobility4.shape[0] # = 100006
    num_epoch = num_epoch
    #num_threads = num_threads
    mobility_ratio = mobility_ratio #= 0.2
    emb_n = learn(walk, num_w, num_wl, user_checkins, user_checkins_count, emb_n, num_n, dim_emb, starting_alpha, num_neg,
					neg_sam_table_social,table_size_social, win_size, neg_sam_table_mobility1,table_size_mobility1,neg_sam_table_mobility2,
					table_size_mobility2,neg_sam_table_mobility3,table_size_mobility3,neg_sam_table_mobility4,table_size_mobility4,
					num_epoch, mobility_ratio)
    return(emb_n)
           
