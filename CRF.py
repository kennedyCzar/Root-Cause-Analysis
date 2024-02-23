# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 13:28:52 2023

@author: amine.melakhsou
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn_crfsuite import CRF, scorers, metrics
from sklearn_crfsuite.metrics import flat_classification_report
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn_crfsuite.metrics import flat_classification_report,flat_f1_score
from sklearn.model_selection import train_test_split
from collections import Counter




n=100000

Parcours=pd.read_csv("Log_parcours_complets.csv",sep=";")[:n]


#Add a label column
Parcours["Label"]=0


def make_labels(data):
    for i in range(data.shape[0]-1):
        if (data["Event"][i]==15 and data["Event"][i+1]==13):
            data["Label"][i+1]=1
    

#take a sample of the data

        
make_labels(Parcours)            
    
Parcours['EventTS'] = pd.to_datetime(Parcours['EventTS'])



#Group by eventID to form a sentence
class getsentenc_events(object):
    
    def __init__(self, data):
        self.n_sent = 1.0
        self.data = data
        self.empty = False
        agg_func = lambda s: [(e, t, l) for e, t, l in zip(s["Event"].values.tolist(),
                                                           s["EventTS"].tolist(),
                                                           s["Label"].values.tolist())]
        self.grouped = self.data.groupby("EventID").apply(agg_func)
        self.sentences = [s for s in self.grouped]
        

getter = getsentenc_events(Parcours)
sentences = getter.sentences
print(sentences[0])


#Feature extraction function

def disc_time(mins):
    range_int=np.arange(10,1441,10)
    val=""
    if mins<range_int[0]:
        val="<"+str(range_int[0])
        
    for i in range(len(range_int)-1):
        if mins>range_int[i]:
            val=str(range_int[i])+"-"+str(range_int[i+1])
    if mins>range_int[-1]:
        val=">"+str(range_int[-1])
    return(val)
            
def get_features(sent):
    
    features = []
    
    for i in range(len(sent)):
        
        
        if i==0:

            features_dict  = {
                
                'Time difference Tt-(Tt-1)':str("none"),
                'value t-2':str("none")}
        
        
        if i>0:
            td=(sent[i][1]-sent[i-1][1]).total_seconds()/60/60
            td_str=round(td,0)
            
            features_dict  = {
                
                'Time difference Tt-(Tt-1)':str(disc_time(td_str)),
                'value t-2':str("none")}
        
        if i>1:
            td=(sent[i][1]-sent[i-1][1]).total_seconds()/60/60
            td_str=round(td,0)
            features_dict  = {
                
                'Time difference Tt-(Tt-1)':str(disc_time(td_str)),
                'value t-2': str(sent[i-2][0])+"-"+str(sent[i-1][0])
                }


        
        features.append(features_dict)
    
            
    return features  



#Extract features

def sent2labels(sent):
    return [label for event, time, label in sent]
    
X = [get_features(s) for s in sentences]
print(X[0])

#Extract label sequence and convert to string
y = [sent2labels(s) for s in sentences]
label_mapping = {0: "O", 1: "v"}
y = [[label_mapping[label] for label in example] for example in y]


#Prepare train test and fit CRF
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.3)

crf = CRF(algorithm='lbfgs',
          c1=0.1,
          c2=0.1,
          max_iterations=150,
          all_possible_transitions=False)

try:
    crf.fit(x_train, y_train)
except AttributeError:
    pass

predictions = crf.predict(x_test)


print("pred:",predictions[:10])
print("real:",y_test[:10])

#Evaluate---

flat_f1_score(y_test,predictions,average="macro")
#Interpret

def print_state_features(state_features):
    for (attr, label), weight in state_features:
        print("%0.6f %-8s %s" % (weight, label, attr))

print("Top positive:")
print_state_features(Counter(crf.state_features_).most_common(50))


print("\nTop negative:")
print_state_features(Counter(crf.state_features_).most_common()[-30:])

Tt_2job_ok=[]
Tt_2job_V=[]

tdelta_ok=[]
tdelta_V=[]

for i in range(len(y_train)):
    for j in range(len(y_train[i])):
        if y_train[i][j]=='O':
            Tt_2job_ok.append(x_train[i][j]['value t-2'])
            tdelta_ok.append(x_train[i][j]['Time difference Tt-(Tt-1)'])
        else:
            Tt_2job_V.append(x_train[i][j]['value t-2'])
            tdelta_V.append(x_train[i][j]['Time difference Tt-(Tt-1)'])
            
        

plt.hist(Tt_2job_ok)
plt.show() 
plt.hist(Tt_2job_V)
plt.show()


plt.hist(tdelta_ok)
plt.show() 
plt.hist(tdelta_V)
plt.show()


