
#%%
from Markov_chain import *
from Pre_process.Pre_process import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# You put here the folder with your data, in my case the toys dataset in ex1 and the real dataset in ex2
data_path="/Users/samuelgruffaz/Documents/Thèse_vrai/Madics/Git/data/ex1"

utils=pre_process_utils(data_path) # some functions to treat the data
Parcours, violation,ID_viol,Labels,EventID,Event=utils.load()
Listoftransition,nb_event,event_to_ind,unique=utils.event_to_list_of_transition(EventID,Event) #related to the markov chain estimation

num="good" # we choose ID of normal sequence
ID_selected_good=utils.ID_viol_selected(ID_viol,Labels,viol_num=num)
print("ID selected viol", len(ID_selected_good))

trace,counts=utils.maketrace(ID_selected_good,EventID,Event) # we create trace to go faster by using trace and not log

Listoftransition_good=utils.trace_to_list_of_transition(trace,counts) # we compute the related list of transition with their number in , ex: L= [(E1,E2,100)]

MC_instance=MarkovChain(Listoftransition_good,nb_event,event_to_ind,unique,counts=1)# we estimate the markov chain with the empirical transition matrix

print("total ID", len(ID_viol))
print("total ID good", len(ID_selected_good))

#%%
#num=0 # we choose at least the first violation (15-13) 
num=np.zeros(len(Labels[0]))
num[0]=1 # We choose only the fisrt violation
num= None # We choose all violations, we should mute unecessary num of course
ID_selected=utils.ID_viol_selected(ID_viol,Labels,viol_num=num)
print("ID selected viol", len(ID_selected))
trace,counts=utils.maketrace(ID_selected,EventID,Event)
Listoftransition_viol=utils.trace_to_list_of_transition(trace,counts)
MC_viol=MarkovChain(Listoftransition_viol,nb_event,event_to_ind,unique,counts=1)
ind_to_event=list(event_to_ind.keys())
print(ind_to_event)# we convert indice to event (we use indice in the markov chain)

th=0.01# difference of probability
beta=0.00# take beta >0 to remove extreme values (not necessary)
Diff=[]

Signe=[]
for i in range(nb_event): # We are looking for the main difference regarding 
    #the probability AND the frequence of apparition of transition i->j
    for j in range(nb_event):
        var_=MC_viol.std_transition[i,j]*3+MC_instance.std_transition[i,j]*3
        signe_=abs(MC_instance.prob_transition[i,j]-MC_viol.prob_transition[i,j])/max(th,var_)# KPI of significativity for the probability estimation
        freq_rel_pop_viol,freq_rel_pop_good=MC_viol.nb_transition[i,j]/len(ID_selected), MC_instance.nb_transition[i,j]/len(ID_selected_good)# frequency of apparition
        signe_=signe_*(freq_rel_pop_good/(freq_rel_pop_viol+0.001)+freq_rel_pop_viol/(freq_rel_pop_good+0.001))# KPI +the frequency of apparation
        bool_m=MC_viol.prob_transition[i,j]>=beta and MC_instance.prob_transition[i,j]>=beta
        bool_max=MC_viol.prob_transition[i,j]<=1-beta and MC_instance.prob_transition[i,j]<=1-beta
        if signe_>1 and var_>0 and bool_m and bool_max:
            Diff.append((i,j))
            Signe.append((i,j,signe_))
    
Signe_sorted=sorted(Signe, key=lambda x: x[2],reverse=True)# We sort the transition according to the KPI of difference signe
for i in range(len(Signe_sorted)):
    a,b,s=Signe_sorted[i]
    freq_rel_pop_viol,freq_rel_pop_good=MC_viol.nb_transition[a,b]/len(ID_selected), MC_instance.nb_transition[a,b]/len(ID_selected_good)
    print("{:.0f},{:.0f}, MCviol: {:.3f}, MCgood: {:.3f}, signe: {:.0f}, nb_transition (viola and good):{:.3f},{:.3f} ".format(ind_to_event[a],ind_to_event[b],MC_viol.prob_transition[a,b],MC_instance.prob_transition[a,b],s,freq_rel_pop_viol,freq_rel_pop_good))

#print(Diff)
seed=10
plt.show()
plt.figure(1)
plt.clf()
Mat_freq_good=MC_instance.freq_apparition(len(ID_selected_good))
Mat_freq_viol=MC_viol.freq_apparition(len(ID_selected))# We compute the matrices related to the frequency
utils.comparison([Mat_freq_good,Mat_freq_viol])# We compare them

mc.plot_comparison([MC_instance.markovchain,MC_viol.markovchain])# we compare the transition probability matrices
plt.show()


