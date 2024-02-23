import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class analyse_time():
    def __init__(self,Parcours,ID_selected_good,ID_selected):
        """
        We xxant to analyse time related to some index
        """
        ind=self.create_logindex_related_to_id(Parcours.values[:,0],ID_selected_good)
        Parcours_sel=Parcours.iloc[ind]
        Parcours_sel['EventTS'] = pd.to_datetime(Parcours_sel['EventTS'])
        #rint(Parcours_sel['EventTS'].min())
        Parcours_sel["Delta"]=self.time_good(Parcours_sel)
        self.parcours_good=Parcours_sel
        ind=self.create_logindex_related_to_id(Parcours.values[:,0],ID_selected)
        Parcours_vol=Parcours.iloc[ind]
        Parcours_vol['EventTS'] = pd.to_datetime(Parcours_vol['EventTS'])
        Parcours_vol["Delta"]=self.time_good(Parcours_vol)
        #print(Parcours_vol['EventTS'].min())
        self.parcours=Parcours_vol

    def time_good(self,Parcours):
        """
        we take the Dataframe of log and return the relative of each sequence 
        """
        time=Parcours['EventTS'].values
        ParcoursID=Parcours.values[:,0]
        k=0
        n=len(time)
        ID_selec=np.unique(ParcoursID)
        L=[]
        for i in range(len(ID_selec)):
            
            while k<n and ParcoursID[k]!=ID_selec[i]:
                    k=k+1
            off=time[k]
            while k<n and ParcoursID[k]==ID_selec[i]:
                L.append((time[k]-off)/np.timedelta64(1,'D'))# the time is relative to the beginning
                k=k+1
        return np.array(L)

    def compute(self,bins=20,range_=(0,20),y_lim=[0,0.5]):
        """
        Use to compute the time (relative) histogram related to each event
        """
        Unique=np.unique(self.parcours["Event"].values)
        figure, axes = plt.subplots(len(Unique), 2, dpi=100,figsize=(16,16))
        for i in range(len(Unique)):
            #axes[i,0].set_title("Event "+str(i)+" for Good", fontsize=9.0, fontweight='normal', pad=1)
            #plt.figure(i)
            #plt.title("hist time for event "+str(i))
            #plt.clf()
            
            L=self.parcours_good.loc[self.parcours_good["Event"]==Unique[i]].values
            axes[i,0].hist(L[:,3],bins=bins,color="b",density=True,range=range_)
            axes[i,0].set_ylim(y_lim)
            #axes[i,1].set_title("Event "+str(i)+" for anormal", fontsize=9.0, fontweight='normal', pad=1)
            D=self.parcours.loc[self.parcours["Event"]==Unique[i]].values
            axes[i,1].hist(D[:,3],bins=bins,color="r",density=True,range=range_)
            axes[i,1].set_ylim(y_lim)
            print("size good events",len(L),"size bad events ", len(D))
        plt.show()
        return figure, axes

    def create_logindex_related_to_id(self,ParcoursID,ID_selected):
        """
        Function which create log indices thank to the chosen ID
        """
        k=0
        n=len(ParcoursID)
        Ind=[]
        for i in range(len(ID_selected)):
            while k<n and ParcoursID[k]!=ID_selected[i]:
                    k=k+1
            while k<n and ParcoursID[k]==ID_selected[i]:
                Ind.append(k)
                k=k+1
        return Ind

#regarder si le min est bien 0 pour les histogrames
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import f1_score
import plotly.express as px
from sklearn.metrics import confusion_matrix
class Logistic_on_first_event():

    def __init__(self,n_event=3,penalty="l2",class_weight=None):
        """
        n_event the number of event to consider, penalty = "l2","l1"
        class_weight to specify how to balance ex: = {0:0.1,1:1}
        it is chosen automaticly otherwise
        """
        self.n_event=n_event
        self.clf=None
        self.penalty=penalty
        self.class_weight=class_weight
    def fit(self,X,Y,seed=0):
        """
        Once X,Y have a good format, seed 
        """
        if self.class_weight is None:# class_weight is chosen automaticly
            
            self.class_weight={ 0: np.sum(Y[Y[:]==1])/len(Y), 1:1}
        if self.penalty=="l1":
            solver="liblinear"
        else:
            solver="lbfgs"
        self.clf = LogisticRegression(random_state=seed,max_iter=200,penalty=self.penalty,solver=solver,class_weight=self.class_weight).fit(X, Y)
        
    
    def predict(self,X):
        return self.clf.predict(X)
    def score(self,X,Y,average="weighted"):
        print("score")
        return f1_score(Y,self.predict(X)),confusion_matrix(Y,self.predict(X))
    def print_coef(self,hover=None,with_time=True):
        """
        This function print the coordinates of the coeficents 
        by coloring and changing the size according to the weight
        We print it on a grid to simplify the visulation
        """
        cat=self.enc.categories_
        coef=self.clf.coef_[0]
        todf=[]

        k=0
        for i in range(len(cat)):
            event_at_i_step=cat[i]
            for j in range(len(event_at_i_step)):
                if k<len(coef):
                    todf.append(["N"+str(i+1),str(event_at_i_step[j]),coef[k]])
                k=k+1
        co_name="E"
        if k<len(coef):
        
            for j in range(self.n_event):
                todf.append(["N"+str(j+1),"time",coef[k]])
                k=k+1
            co_name="E and T"
            
        df=pd.DataFrame(todf,columns=["N",co_name,'coef'])
      
        df["coef_abs"]=abs(df["coef"])
        fig = px.scatter(df, x=co_name,y="N",color="coef",size="coef_abs",hover_data=hover)
        #fig.title("Coef of the Logisitc regression acording to the features")
        fig.show()
        


    def pre_good_format(self,Parcours,ID_selected):
        """
        some previous steps of pre_processing
        """
        Parcours_sel=Parcours
        Parcours_sel['EventTS'] = pd.to_datetime(Parcours_sel['EventTS'])
        Parcours_sel["Delta"]=(Parcours_sel['EventTS']-Parcours_sel['EventTS'].min())/np.timedelta64(1,'D')
        Parcours_val=Parcours_sel.values
        #for i in range(len(Parcours)):
         #   Parcours_val[i]=
        ind,X1,X2,Reject=self.create_logindex_related_to_id_cut(Parcours_val,ID_selected)
        return ind,X1,X2,Reject



    def good_format(self,Parcours,Violation,ID_selected):
        """
        We use the one hot encoding and a standard scaler
        
        """
        ind,X1,X2,Reject=self.pre_good_format(Parcours,ID_selected)
        self.enc = OneHotEncoder()
        print(X1[:2])
        X1_new=self.enc.fit_transform(X1).toarray()
        print(self.enc.categories_)
        h=0
        for f in self.enc.categories_:
            for e in f:
                h=h+1

        self.n_features=h
        print(np.shape(X1_new))
        self.scaler=StandardScaler()
        X2_new=self.scaler.fit_transform(X2)
        X=np.concatenate([X1_new,X2_new],axis=1)
        ID_selected_new=np.array(list(set(list(ID_selected)) - set(Reject)))
        Y=Violation.values[ID_selected_new-1,1:]

        return X,Y


    def create_logindex_related_to_id_cut(self,ParcoursID,ID_selected):
        """
        We take the beginning of the sequence , the first #n_event
        
        """
        k=0
        n=len(ParcoursID[:,0])
        Ind=[]
        XX1=[]
        XX2=[]
        Reject=[]
        for i in range(len(ID_selected)):
            while k<n and ParcoursID[k,0]!=ID_selected[i]:
                    k=k+1
            g=0
            X_1=[]
            X_2=[]
            offset=ParcoursID[k,2]
            while k<n and ParcoursID[k,0]==ID_selected[i] and g<self.n_event:
                
                X_1.append(ParcoursID[k,1])
                X_2.append((ParcoursID[k,2]-offset)/np.timedelta64(1,"m"))
                g=g+1
                k=k+1
            if len(X_1)<self.n_event:
                print("we have to skip",ID_selected[i])
                Reject.append(ID_selected[i])
            else:
                Ind.append(k-1)
                XX1.append(X_1)
                XX2.append(X_2)
        return Ind,np.array(XX1),np.array(XX2),Reject
    
