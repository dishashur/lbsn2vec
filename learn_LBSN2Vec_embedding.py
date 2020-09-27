from numpy import *
from random import *

#defining constants
def ULONG_MAX():
	return(getrandbits(66))

def RAND_MULTIPLIER():
	return(25214903917)

def RAND_INCREMENT():
	return(11)

def MAX_EXP:
	return(6)

def EXP_TABLE_SIZE:
	return(1000)


################################

def getNextRand(next_random):
	next_random = next_random*RAND_MULTIPLIER() + RAND_INCREMENT(); 
	return(next_random)

def get_a_social_decision(next_random):
	v_rand_uniform = next_random/ULONG_MAX();
    if v_rand_uniform<=mobility_ratio:
		return 0;
    else:
        return 1;

def get_a_mobility_decision(next_random):
	double v_rand_uniform = next_random/ULONG_MAX();
    if (v_rand_uniform<=mobility_ratio):
		return 1;
    else:
        return 0;

def get_a_checkin_sample(next_random, table_size):
    return ((next_random >> 16) % table_size);

def get_a_neg_sample(next_random, neg_sam_table, table_size):
	ind = (next_random >> 16) % table_size;
    target_n = neg_sam_table[ind];
	return(target_n);

def get_a_neg_sample_Kless1(next_random):
	double v_rand_uniform = next_random/ULONG_MAX();
    if (v_rand_uniform<=num_neg):
		return 1;
    else:
		return 0;

def get_norm_l2_loc(long long loc_node):
    norm = 0;
	d = 0;
    for d in range(0,dim_emb):
		norm = norm + emb_n[loc_node+d] * emb_n[loc_node+d];
    return (sqrt(norm));

def get_norm_l2_pr(vec): ## will not need this, use inbuilt norm from python; here for time being
	norm = 0;
	d=0;
	for d in range(0,dim_emb):
		norm+ = vec[d] * vec[d];
    return (sqrt(norm));

def learn_a_pair_loc_pr_cosine(flag, loc1, best_fit, loss):
	f=0;
	g=0;
	a=0;
	
    norm1 = get_norm_l2_loc(loc1); ##copy this
    for d in range(0,dim_emb):   
		f += emb_n[loc1+d] * best_fit[d];

    g = f/norm1;

    a = alpha;
    c1 = 1/(norm1)*a;
    c2 = f/(norm1*norm1*norm1)*a;

    if (flag==1):
		d = 0;
		for d in range(dim_emb):   
			emb_n[loc1 + d] += c1*best_fit[d] - c2*emb_n[loc1 + d];
    else:
		d = 0;
		for d in range(dim_emb):    
			emb_n[loc1 + d] -= c1*best_fit[d] - c2*emb_n[loc1 + d];

	return(emb_n);    



def learn_an_edge_with_BFT(word, target_e, next_random, best_fit, counter):
	loc_w = (word-1)*dim_emb;
    loc_e = (target_e-1)*dim_emb;

    for d in range(dim_emb): 
		best_fit[d] = emb_n[loc_w+d] + emb_n[loc_e+d];
    norm = get_norm_l2_pr(best_fit); ##copy this
	d=0
    for d in range(0,dim_emb): 
		best_fit[d] = best_fit[d]/norm;

    emb_n = learn_a_pair_loc_pr_cosine(1, loc_w, best_fit, counter); ##copy this
    emb_n = learn_a_pair_loc_pr_cosine(1, loc_e, best_fit, counter); ##copy this

    if (num_neg<1):
        getNextRand(next_random);
        if (get_a_neg_sample_Kless1(next_random)==1):
            getNextRand(next_random);
            target_n = get_a_neg_sample(next_random, neg_sam_table_social, table_size_social);
            if ((target_n != target_e) && (target_n != word)):
                loc_neg = (target_n-1)*dim_emb;
                emb_n = learn_a_pair_loc_pr_cosine(0, loc_neg, best_fit, counter); #copy this
            
        
    else:
        for (int n=0;n<num_neg;n++):
            getNextRand(next_random);
            target_n = get_a_neg_sample(next_random, neg_sam_table_social, table_size_social); #copy this
            if ((target_n != target_e) && (target_n != word)):
                loc_neg = (target_n-1)*dim_emb;
                emb_n = learn_a_pair_loc_pr_cosine(0, loc_neg, best_fit, counter); #copy this	

	return(emb_n)



def learn_a_hyperedge(edge, edge_len, next_random, best_fit, counter):


#################### get best-fit-line
	for d in range(0,dim_emb):
		best_fit[d] = 0;
    for i in range(0,edge_len):
        loc_n = (edge[i]-1)*dim_emb;
        norm = get_norm_l2_pr(emb_n[loc_n]);
        for d in range(0,dim_emb):
			best_fit[d] += emb_n[loc_n + d]/norm;

#  normalize best fit line for fast computation
    norm = get_norm_l2_pr(best_fit);
    for d in range(0,dim_emb):
		best_fit[d] = best_fit[d]/norm;


#################### learn learn learn
    for i in range(0,edge_len):
        node = edge[i];
        loc_n = (node-1)*dim_emb;
        emb_n = learn_a_pair_loc_pr_cosine(1, loc_n, best_fit, counter); ## copy this

        if (num_neg<1): 
            next_random = getNextRand(next_random); ## copy this
            if (get_a_neg_sample_Kless1(next_random)==1):
                next_random = getNextRand(next_random);
                if (i==0): 
					target_neg = get_a_neg_sample(next_random, neg_sam_table_mobility1, table_size_mobility1); ## args to be ini 4m ip
                elif (i==1): 
					target_neg = get_a_neg_sample(next_random, neg_sam_table_mobility2, table_size_mobility2); ## also copy this
                elif (i==2): 
					target_neg = get_a_neg_sample(next_random, neg_sam_table_mobility3, table_size_mobility3);
                elif (i==3): 
					target_neg = get_a_neg_sample(next_random, neg_sam_table_mobility4, table_size_mobility4);

                if (target_neg != node):
                    loc_neg = (target_neg-1)*dim_emb;
                    emb_n = learn_a_pair_loc_pr_cosine(0, loc_neg, best_fit, counter); ## copy this
                
            
        else:
        	for n in range(num_neg): 
                next_random = getNextRand(next_random);
                if (i==0): 
					target_neg = get_a_neg_sample(next_random, neg_sam_table_mobility1, table_size_mobility1); ## args to be ini 4m ip
                elif (i==1): 
					target_neg = get_a_neg_sample(next_random, neg_sam_table_mobility2, table_size_mobility2); ## also copy this
                elif (i==2): 
					target_neg = get_a_neg_sample(next_random, neg_sam_table_mobility3, table_size_mobility3);
                elif (i==3): 
					target_neg = get_a_neg_sample(next_random, neg_sam_table_mobility4, table_size_mobility4);

                if (target_neg != node): 
                    loc_neg = (target_neg-1)*dim_emb;
                    emb_n = learn_a_pair_loc_pr_cosine(0, loc_neg, best_fit, counter); 
	
	### supposedly return(emb_n); # but idk where idi is coming from               
            
        

def learn(idi):  #id is a reserved word in python
	
	best_fit = array([[] for _ in range(dim_emb)]) 
	next_random = getrandbits(32); #equivalent to (long) rand() in C
	edge_len = 4; # here 4 is a checkin node number user-time-POI-category
	ind_start = num_w/num_threads*idi;
    ind_end = num_w/num_threads*(idi+1);

    ind_len = ind_end-ind_start;
    progress=0;
	progress_old=0;
    alpha = starting_alpha;


    for pp in range(0,num_epoch):   
		counter = 0;
		for w in range(ind_start,ind_end):  
        	progress = ((pp*ind_len)+(w-ind_start))/(ind_len*num_epoch);
            if (progress-progress_old > 0.001):
                alpha = starting_alpha * (1 - progress);
                if (alpha < starting_alpha * 0.001):
					alpha = starting_alpha * 0.001;
                progress_old = progress;


            loc_walk = w*num_wl;
            for i in range(num_wl):
	    		word = walk[loc_walk+i];

                for j in range(1,win_size+1):
                	getNextRand(next_random);
                    if (get_a_social_decision(next_random)==1):
						if (i-j)>=0:
                            target_e = walk[loc_walk+i-j];
                            if (word is not target_e):
                                learn_an_edge_with_BFT(word, target_e, next_random, best_fit, counter); ##copy this

                        
                        if (i+j)<num_wl : 
                        	target_e = walk[loc_walk+i+j];
                            if (word is not target_e):
                            	learn_an_edge_with_BFT(word, target_e, next_random, best_fit, counter); ##copy this

            

                if user_checkins_count[word-1]>0:
                    for m in range(min(win_size*2,user_checkins_count[word-1])):     
                        getNextRand(next_random);
                        if (get_a_mobility_decision(next_random)==1) {

                            user_pr = user_checkins[word-1]; #Pointer to the ith cell mxArray
                            a_user_checkins = user_pr;

                            getNextRand(next_random);
                            a_checkin_ind = get_a_checkin_sample(next_random, user_checkins_count[word-1]);

                            a_checkin_loc = a_checkin_ind*edge_len;
                            edge = a_user_checkins[a_checkin_loc];

                            emb_n = learn_a_hyperedge(edge, edge_len, next_random, best_fit, counter); 
                        

	del(best_fit)    
	### supposedly return(emb_n); # but idk where idi is coming from             

# main function ---- equivalent to mexFunction

def learn_lbsn2vec_embedding(walks,user_checkins, user_checkins_counter, embs_ini, learning_rate, K_neg,
	 neg_sam_table_social, win_size, neg_sam_table_mobility_norm, num_epoch, num_threads, mobility_ratio):
	
	walk = walks;
	num_w = walks.shape[1]; #N
	num_wl = walks.shape[0]; #M
	user_checkins = user_checkins;
	num_u = user_checkins.size;
	user_checkins_count = user_checkins_counter;
	emb_n = embs_ini;
	num_n = emb_n.shape[1]; #N
	dim_emb = emb_n.shape[0]; #M
	starting_alpha = learning_rate;
	num_neg = K_neg;
	neg_sam_table_social = neg_sam_table_social;
	table_size_social = neg_sam_table_social.shape[0]; #M
	win_size = win_size;
	neg_sam_table_mobility = neg_sam_table_mobility_norm;
	table_num_mobility = neg_sam_table_mobility.size; 
	if(table_num_mobility != 4):
		print("four negative sample tables are required in neg_sam_table_mobility");
	neg_sam_table_mobility1 = neg_sam_table_mobility[0];
	table_size_mobility1 = neg_sam_table_mobility1.shape[0]; #M
	neg_sam_table_mobility2 = neg_sam_table_mobility[1];
	table_size_mobility2 = neg_sam_table_mobility2.shape[0];
	neg_sam_table_mobility3 = neg_sam_table_mobility[2];
	table_size_mobility3 = neg_sam_table_mobility3.shape[0];
	neg_sam_table_mobility4 = neg_sam_table_mobility[3];
	table_size_mobility4 = neg_sam_table_mobility4.shape[0];
	
	num_epoch = num_epoch;
	num_threads = num_threads;
	mobility_ratio = mobility_ratio;

	##emb_n = learn ()
	##return(emb_n)
           
