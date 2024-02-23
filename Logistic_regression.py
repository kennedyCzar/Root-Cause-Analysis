
#%%
from Markov_chain import *
from Pre_process.Pre_process import *
from time_an import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#path to data folder
data_path="/Users/samuelgruffaz/Documents/Thèse_vrai/Madics/Git/data/ex2"

utils=pre_process_utils(data_path)
Parcours, violation,ID_viol,Labels,EventID,Event=utils.load()





viola=3 # select the violattion 
#class_weight={0:0.1,1:0.8}
n_event=3 # number of event to consider at the biginning
logis=Logistic_on_first_event(n_event=n_event,penalty="l1")# We apply a logistic on the first #n_event
X,Y=logis.good_format(Parcours,violation,ID_viol)
print("number of violations",np.sum(Y[Y[:,viola]==1,viola])/len(Y))
from sklearn.model_selection import train_test_split
seed=42
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=seed)
n_f=logis.n_features
#n_f=logis.n_features+n_event
#n_f=15
#print(n_f)
logis.fit(X_train[:,:n_f],y_train[:,viola])

a,M=logis.score(X_test[:,:n_f],y_test[:,viola])
print("f1 score")
print(a)
print("confusion matrix")
print(M)
# analysing the coefficients
logis.print_coef()
#visualisation à améliorer, mettre texte surout


"""
Others stuff, visualisation of the time histogram related to eaach event,
looking at the first apparition of anomaly

"""

#Listoftransition,nb_event,event_to_ind,unique=utils.event_to_list_of_transition(EventID,Event)
#ind_to_event=list(event_to_ind.keys())

#print("total ID", len(ID_viol))
#num="good"
#ID_selected_good=utils.ID_viol_selected(ID_viol,Labels,viol_num=num)
#print("ID selected good", len(ID_selected_good))

#trace,counts=utils.maketrace(ID_selected_good,EventID,Event)

num=None# we choose all violations
#num=np.zeros(len(Labels[0])) 
#num[0]=1 # we choose only the first violation
#ID_selected=utils.ID_viol_selected(ID_viol,Labels,viol_num=num)
#print("ID selected viol", len(ID_selected))
#trace,counts=utils.maketrace(ID_selected,EventID,Event)
#print("trace ok")
#an=analyse_time(Parcours,ID_selected_good,ID_selected) # to analyse time histograms

#an.compute(bins=20,range_=(0,10),y_lim=[0,1])





#%%
#ID_pb,Lind,L_deb_seq=utils.detect_first_anomaly(trace,[8,2]) 
#print(Diff)
#seed=10
#plt.show()
#plt.figure(1)
#plt.clf()
#plt.hist(Lind,weights=counts[ID_pb],bins=40,density=True,range=(0,20))
#plt.xlabel("time where the violation occured")
#plt.legend()
#plt.show()



# %%
