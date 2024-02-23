from Markov_chain import *
from Pre_process.Pre_process import *
from time_an import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data_path="/Users/samuelgruffaz/Documents/Thèse_vrai/Madics/Git/data/ex2"

utils=pre_process_utils(data_path)
Parcours, violation,ID_viol,Labels,EventID,Event=utils.load()
n_event=3
logis=Logistic_on_first_event(n_event=n_event,penalty="l1")

ind,X1,X2,Reject=logis.pre_good_format(Parcours,ID_viol)# X1 is a list of the first #n_event event
#X2 is a list with the first #nevent times
#Reject is if the ID rejected becauses the size of the sequence is less than #n_event
ID_selected_new=np.array(list(set(list(ID_viol)) - set(Reject)))
Y=violation.values[ID_selected_new-1,1:]# vector of violation labels

df=pd.DataFrame(X1,columns=["N1","N2","N3"])
for i in range(5):
    df["Y"+str(i+1)]=Y[:,i]


import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
pd.options.plotting.backend = "plotly"

viola="Y3" # To modify to consider other type of violation Y1,Y1,Y3,Y4
fig = make_subplots(rows=3, cols=2)


for i in range(1,1+n_event):
    # We print histogram of event related to the violation Y3 or when there is no Y3 violation
    # histogram for the first #n_event
    for j in range(2):
        ass=" yes" if j==1 else "no"
        etape="N"+str(i)
        fig1=go.Histogram(
        x=df[df[viola]==j]["N"+str(i)].values,
        histnorm='probability density',
        name=etape+" "+viola+ass, 
        xbins=dict(start=0,
        end=8,
        size=1),
        autobinx=False,
        opacity=0.75
        )
        fig.append_trace(fig1,i,j+1)
    

fig.show()


df=pd.DataFrame(X2,columns=["N1","N2","N3"])
for i in range(5):
    df["Y"+str(i+1)]=Y[:,i]

figtime = make_subplots(rows=2, cols=2)

for i in range(1,3):
    # We print histogram of TIME related to the violation Y3 or when there is no Y3 violation
    # histogram for the first #n_event
    for j in range(2):
        ass=" yes" if j==1 else "no"
        etape="N"+str(i+1)
        fig2=go.Histogram(
        x=df[df[viola]==j]["N"+str(i+1)].values,
        histnorm='probability density',
        name=etape+" "+viola+ass,
        autobinx=True, 
        opacity=0.75)
        figtime.append_trace(fig2,i,j+1)
figtime.show()