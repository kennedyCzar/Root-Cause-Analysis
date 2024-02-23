import numpy as np

import matplotlib.pyplot as plt
import pandas as pd
import os
import matplotlib.colorbar as _mplcb
import matplotlib.colors as _mplcr
import matplotlib.image as _mpli
import matplotlib.ticker as _mplt

_color_black = '#000000'
_color_gray = '#E0E0E0'
_color_white = '#FFFFFF'
_colors = ('#80B1D3', '#FFED6F', '#B3DE69', '#BEBADA', '#FDB462', '#8DD3C7', '#FB8072', '#FCCDE5', '#E5C494')

_default_color_edge = _color_black
_default_color_node = _color_white
_default_color_path = _colors[0]
_default_color_symbol = _color_gray
_default_node_size = 600

data_path="/Users/samuelgruffaz/Documents/Thèse_vrai/Madics/Git/data"


class pre_process_utils() :
    def __init__(self,data_path):
        self.parcours=os.path.join(data_path,"Log_parcours_complets.csv")
        self.violation=os.path.join(data_path,"Violations.csv")
        self.trace=os.path.join(data_path,"trace.xlsx")

    def array_to_string(self,array):
        """ trace array to trace string"""
        G=list(array)
        for i in range(len(G)):
            G[i]=str(G[i])
        return "-".join(G)
    def string_to_array(self,stri):
        """ trace string to trace array"""
        L=stri.split("-")
        for e in range(len(L)):
            L[e]=int(L[e])
        return L
    
    
        

    def load_trace(self):
        """ load trace return trace as an array of string and their relative number"""
        trace=pd.read_xlsx(self.trace,sep=";")
        trace_val=trace["Traces"].values
        trace_num=trace["Effectif"].values
        return trace_val,trace_num


    
    def maketrace(self,ID_list,EventID,Event):
        """ ID_list to consider, EventID : liste des ID dans Log file,
         Event: list des Event dans Log File, 
         return Trace as a list of string and count the relative number """
        D={}
        L=[]
        k=0
        n=len(EventID)
        for i in range(len(ID_list)):
            G=[]
            while k<n and EventID[k]!=ID_list[i]:
                k=k+1
            while k<n and EventID[k]==ID_list[i]:
                G.append(Event[k])
                k=k+1

            Id_string=self.array_to_string(G)
            L.append(Id_string)
        print("for done")
        Trace,counts=np.unique(L,return_counts=True)
        return Trace,counts

    def trace_to_list_of_transition(self,Trace,counts):
        """
        Transform the trace string list as a list of transition (tuple list)
        
        """
        n=len(Trace)
        Listoftransition=[]
        for i in range(n):
            list_a=self.string_to_array(Trace[i])
            for j in range(len(list_a)-1):
                Listoftransition.append((list_a[j],list_a[j+1],counts[i]))
        return Listoftransition

            
        

    
    def load(self):
        """
        Load the violations and log file as a Dataframe, 
        return two Datafram (Parcours<-log and violation<-viola)
        and 4 array related to the ID log and violation and the labels and event
        
        """
        Parcours=pd.read_csv(self.parcours,sep=";")
        violation=pd.read_csv(self.violation,sep=";")
        if "EventID" in violation.columns:
            ID_viol=violation["EventID"].values
            Labels=violation[["15_13","15_15_16","15_15_17","17_17_17","4_2_4","18_16","13_17_2_days_and_more","1_First_appears_as_seventh","13_and_18_separated_by_one"]].values
            EventID=Parcours["EventID"].values
            Event=Parcours["Event"].values
        else:
            
            ID_viol=violation["ID"].values
            Labels=violation[["Violation_1",'Violation_2','Violation_3','Violation_4','Violation_5']].values
            EventID=Parcours["ID"].values
            Event=Parcours["Event"].values

        print("Number of different events",len(np.unique(Event)))
        print("nb id log_parcours",len(np.unique(EventID)))
        print("nb id violation",len(np.unique(ID_viol)))
        #print(Parcours.value_counts("Event"))
        return Parcours, violation,ID_viol,Labels,EventID,Event

    def event_to_list_of_transition(self,EventID,Event):
        """
        If you want to create the list of transition from the events and not the trace
        """
        n=len(Event)
        id0=EventID[0]
        Listoftransition=[]
        for i in range(1,n):
            if EventID[i]==id0 and EventID[i-1]==id0:
                Listoftransition.append((Event[i-1],Event[i]))
            else:
                id0=EventID[i]

        unique=np.unique(Event)
        nb_event=len(unique)
        event_to_ind={}
        for i in range(len(unique)):
            event_to_ind[unique[i]]=i
        return Listoftransition,nb_event,event_to_ind,unique
    def ID_viol_selected(self,ID_viol,Labels,viol_num=None):
        """
        This function select the ID thank to viol_num
        None-> all violations
        "good"-> no violations 
        i-> at least violation i
        ex :y=np.array([1,0,0,0,0])-> all ID with labels equal to y
        """
        if viol_num is None:
            Labels_max=np.max(Labels,axis=1)
            ID_selected=ID_viol[(Labels_max==1)]
        elif viol_num=="good":
            Labels_max=np.max(Labels,axis=1)
            ID_selected=ID_viol[(Labels_max==0)]
        elif type(viol_num)!=type(1):
            bool_d=(Labels[:]==viol_num).all(axis=1)
            
            ID_selected=ID_viol[bool_d]
            
        else:
            Labels_num=Labels[:,viol_num]
            ID_selected=ID_viol[(Labels_num==1)]
        
        return ID_selected
    def comparison(self,models, names = ["mat_good","mat_viola"], dpi: int = 100):
        """
        plot the comparison of the frequency matrix reporting the frequence of apparition
        related to each transition
        """
        space = len(models)
        rows = int(np.sqrt(space))
        columns = int(np.ceil(space / float(rows)))

        figure, axes = plt.subplots(rows, columns, constrained_layout=True, dpi=dpi)
        axes = list(axes.flat)
        ax_is = None

        color_map = _mplcr.LinearSegmentedColormap.from_list('ColorMap', [_color_white, _colors[0]], 20)

        for ax, model, name in zip(axes, models, names):

            matrix = model

            ax_is = ax.imshow(matrix, aspect='auto', cmap=color_map, interpolation='none', vmin=0.0, vmax=1.0)
            ax.set_title(name, fontsize=9.0, fontweight='normal', pad=1)

            #ax.set_xticks([i+1 for i in range(len(matrix[0]))])
            ax.set_xticks([i for i in range(len(matrix[0]))], minor=True)
            #ax.set_yticks([i+1 for i in range(len(matrix[0]))])
            ax.set_yticks([i for i in range(len(matrix[0]))], minor=True)

        color_map_ax, color_map_ax_kwargs = _mplcb.make_axes(axes, drawedges=True, orientation='horizontal', ticks=[0.0, 0.25, 0.5, 0.75, 1.0])
        figure.colorbar(ax_is, cax=color_map_ax, **color_map_ax_kwargs)
        color_map_ax.set_xticklabels([0.0, 0.25, 0.5, 0.75, 1.0])

        figure.suptitle('Comparison Plot for frequence n_appar/n_ind', fontsize=15.0, fontweight='bold')

        if plt.isinteractive():  # pragma: no cover
            plt.show(block=False)
            return None

        return figure, axes

    def detect_first_anomaly(self,trace,todetect):
        """
        This function look at the first apparition of anomali
        trace -> list of trace as a string list
        todect-> list of int

        return ID_pb- > list of ID with the anormal subsequence to detect
        Lind= the number of event before to detect the anormal subsequence as a list
        L_deb_seq= list of the beginnign of eeach sequence before the first anormal subsequence
        """
        Lind=[]
        L_deb_seq=[]
        ID_pb=[]
        for i in range(len(trace)):
            seq_str=trace[i]
            #print(seq_str)
            #print(self.array_to_string(todetect))
            ind=seq_str.find(self.array_to_string(todetect))
            if ind>=0: 
                
                deb_seq=self.string_to_array(seq_str[:ind-1])
                ID_pb.append(i)
                L_deb_seq.append(deb_seq)
                Lind.append(len(deb_seq))
            else:
                
                print("not find subsequence")
                pass
            
        return ID_pb,Lind,L_deb_seq

    
