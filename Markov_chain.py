import numpy as np
import pandas as pd
import pydtmc as mc


class MarkovChain():
    def __init__(self,X,nb_state,event_to_ind,unique,counts=None):
        """ X liste de tuple, we compute the transition probability matrix
        and the variance of estimation """
        self.event_to_ind=event_to_ind
        self.nb_state=nb_state
        A=np.zeros((nb_state,nb_state))
        for i in range(len(X)):
            event1,event2=X[i][0],X[i][1]
            if counts is None:
                w=1
            else:
                w=X[i][2]
            A[event_to_ind[event1],event_to_ind[event2]]+=w

        #print(unique,np.sum(A,axis=1))
        Var=np.zeros((nb_state,nb_state))
        Prob=np.zeros((nb_state,nb_state))
        for i in range(len(A)):
            ss=np.sum(A[i])
            if ss>0:
                Prob[i]=A[i]/ss
            else:
                Prob[i,i]=1# if no transition we stay to the same state
            
            Var[i]=Prob[i]*(1-Prob[i])/(A[i]+1)

        self.prob_transition=Prob
        self.markovchain=mc.MarkovChain(p=Prob,states=[str(e) for e in unique])
        self.std_transition=np.sqrt(Var)
        self.nb_transition=A
    def freq_apparition(self,n_ind):
        """
        we normalize the matrix with the number of transition
        """
        B=self.nb_transition/n_ind
        return B


    
