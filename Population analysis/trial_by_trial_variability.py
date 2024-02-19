
#This script shows that in dishabituation the previous trial (active or passive) influences the subsequent trial at the cell level.


# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 11:50:49 2021

@author: Owner
q"""
from __future__ import print_function

################################
#For all this program:

###############################

from glob import glob
import pickle
import os
from os.path import isfile,join
os.chdir("C:/Users/Owner/Documents/Matan/Mati's Lab/FEF learning project/Data_Analyses/Code analyzes/basic classes")

import pandas as pd
import numpy as np
import re
import scipy.io
import math
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.lines as lines
from scipy.ndimage import gaussian_filter1d
import sys
from neuron_class import cell_task,filtCells,load_cell_task,smooth_data,get_cell_list
import time
import warnings
import scipy.stats as stats
from scipy.signal import savgol_filter
from scipy.stats import kruskal
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import copy
import seaborn as sb
warnings.filterwarnings("ignore")
from collections import Counter
from scipy.io import savemat
import numpy.matlib
import matplotlib as mpl
from itertools import chain
mpl.rcParams['figure.dpi'] = 600
########################################################
path="C:/Users/Owner/Documents/Matan/Mati's Lab/FEF learning project/Data_Analyses"
os.chdir(path) 
#For all this program:
cell_task_py_folder="units_task_two_monkeys_python_kinematics/"

#liste of cells recorded during dishabituation
Tasks=['Dishabituation_100_25_cue','Dishabituation']
dishabituation_cells_list=[]
for cur_task in Tasks:
    dishabituation_cells_list=dishabituation_cells_list+os.listdir(join(cell_task_py_folder,cur_task))
    dishabituation_cells_list=[int(x) for x in dishabituation_cells_list]
cutoff_cell=8229 #cutoff between yasmin and fiona
yasmin_cell_list=[x for x in dishabituation_cells_list if x>cutoff_cell]
fiona_cell_list=[x for x in dishabituation_cells_list if x<=cutoff_cell]
dishabituation_cells_list=dishabituation_cells_list

# ########################################################


#%% Neural behaviour correlations
saccade_motion_parameter=1
event='motion_onset'
window_pre=0
window_post=250
N_min_trials=20

   

FR_active_array=[]
FR_passive_array=[]
BL_active_array=[]
BL_passive_array=[]
delta_FR_active_array=[]
delta_FR_passive_array=[]
NB_corr_active_array=[]
NB_corr_passive_array=[]
cell_group_array=[]
for cell_ID in dishabituation_cells_list:
    for cur_task in Tasks:
        try:
            cur_cell_task=load_cell_task(cell_task_py_folder,cur_task,cell_ID)
        except:
            continue

    dictFilterTrials_pre={'trial_name':'v20S|v20NS'}
    trials_df=cur_cell_task.filtTrials(dictFilterTrials_pre)

    trials_list=trials_df.loc[:]['filename_name'].tolist()
    trial_number_np=np.array([int(x.split('.',1)[1]) for x in trials_list ])
    first_trial_block_indices=np.where(np.diff(trial_number_np)>=80)[0]
    
    block_begin_indexes= trials_df.index[np.append(np.array(0),first_trial_block_indices+1)]
    block_end_indexes=trials_df.index[np.append(first_trial_block_indices,len(trials_df)-1)]
    
    file_begin_indexes=[trial_number_np[0]]+list(trial_number_np[first_trial_block_indices+1])
    file_end_indexes=list(trial_number_np[first_trial_block_indices])+[trial_number_np[-1]]


    for block_index,(cur_begin,cur_end) in enumerate(zip(file_begin_indexes,file_end_indexes)):
        #passive trials of current block
        dictFilterTrials_passive={'trial_name':'v20S', 'fail':0,'saccade_motion':saccade_motion_parameter,'files_begin_end':[cur_begin,cur_end]}
        trials_df_passive=cur_cell_task.filtTrials(dictFilterTrials_passive)
        
        #active trials of current block
        dictFilterTrials_active={ 'trial_name':'v20NS', 'fail':0 ,'saccade_motion':saccade_motion_parameter,'files_begin_end':[cur_begin,cur_end]}
        trials_df_active=cur_cell_task.filtTrials(dictFilterTrials_active)  
        
        if len(trials_df_active)<N_min_trials or len(trials_df_passive)<N_min_trials:
            continue
        
        else:
            #firing rate in window
            FR_active=np.array(cur_cell_task.get_mean_FR_event(dictFilterTrials_active,event,window_pre,window_post))
            FR_passive=np.array(cur_cell_task.get_mean_FR_event(dictFilterTrials_passive,event,window_pre,window_post))
                
            #FR baseline
            FR_BL_active=cur_cell_task.get_mean_FR_event(dictFilterTrials_active,event,window_pre=-300,window_post=-100)
            FR_BL_passive=cur_cell_task.get_mean_FR_event(dictFilterTrials_passive,event,window_pre=-300,window_post=-100)

            delta_FR_active=np.nanmean(FR_active)-np.nanmean(FR_BL_active)
            delta_FR_passive=np.nanmean(FR_passive)-np.nanmean(FR_BL_passive)
            
            try:
                [stat1a,pval1a]=scipy.stats.wilcoxon(FR_active,FR_BL_active,alternative='greater')
                [stat2a,pval2a]=scipy.stats.wilcoxon(FR_active,FR_BL_active,alternative='less')
            except:
                pval1a=1
                pval2a=1

            try:
                [stat1p,pval1p]=scipy.stats.wilcoxon(FR_passive,FR_BL_passive,alternative='greater')
                [stat2p,pval2p]=scipy.stats.wilcoxon(FR_passive,FR_BL_passive,alternative='less')
            except:
                pval1p=1
                pval2p=1
            
            #change in behaviour in widow
            hPosChangeActive= trials_df_active.apply(lambda row: [(row.hPos[window_post+row.motion_onset])-(row.hPos[window_pre+row.motion_onset])],axis=1).to_list()
            hPosChangeActive = np.array([item for sublist in hPosChangeActive for item in sublist])
            vPosChangeActive = trials_df_active.apply(lambda row: [(row.vPos[window_post+row.motion_onset])-(row.vPos[window_pre+row.motion_onset])],axis=1).to_list()
            vPosChangeActive = np.array([item for sublist in vPosChangeActive for item in sublist])
            TotPosChangeActive=(hPosChangeActive**2+vPosChangeActive**2)**0.5
        
            hPosChangePassive= trials_df_passive.apply(lambda row: [(row.hPos[window_post+row.motion_onset])-(row.hPos[window_pre+row.motion_onset])],axis=1).to_list()
            hPosChangePassive = np.array([item for sublist in hPosChangePassive for item in sublist])
            vPosChangePassive = trials_df_passive.apply(lambda row: [(row.vPos[window_post+row.motion_onset])-(row.vPos[window_pre+row.motion_onset])],axis=1).to_list()
            vPosChangePassive = np.array([item for sublist in vPosChangePassive for item in sublist])
            TotPosChangePassive=(hPosChangePassive**2+vPosChangePassive**2)**0.5

          
            NB_corr_active=np.corrcoef(FR_active,TotPosChangeActive)[0,1]
            NB_corr_passive=np.corrcoef(FR_passive,TotPosChangePassive)[0,1]
          

            
            if pval1a<0.05  and np.nanmean(FR_active)>np.nanmean(FR_passive):
                cell_group_array.append(0)
        
            elif pval2a<0.05 and np.nanmean(FR_active)<np.nanmean(FR_passive):
                cell_group_array.append(1)
                
            elif pval1p<0.05  and  np.nanmean(FR_passive)>np.nanmean(FR_active):
                cell_group_array.append(2)
        
            elif pval2p<0.05  and np.nanmean(FR_passive)<np.nanmean(FR_active):
                cell_group_array.append(3)
            else:
                cell_group_array.append(4)
                
                    
            FR_active_array.append(np.nanmean(FR_active))   
            FR_passive_array.append(np.nanmean(FR_passive))
            BL_active_array.append(np.nanmean(FR_BL_active))   
            BL_passive_array.append(np.nanmean(FR_BL_passive))
            delta_FR_active_array.append(np.nanmean(delta_FR_active))   
            delta_FR_passive_array.append(np.nanmean(delta_FR_passive))   
            NB_corr_active_array.append(NB_corr_active)
            NB_corr_passive_array.append(NB_corr_passive)

#%% Group of cells          
increase_cells_active_inx=[index  for index,value  in enumerate(cell_group_array) if value==0]
decrease_cells_active_inx=[index  for index,value  in enumerate(cell_group_array) if value==1]
increase_cells_passive_inx=[index  for index,value  in enumerate(cell_group_array) if value==2]
decrease_cells_passive_inx=[index  for index,value  in enumerate(cell_group_array) if value==3]
other_cells_inx=[index  for index,value  in enumerate(cell_group_array) if value==4]

#from list to numpy array
FR_active_array=np.array(FR_active_array)
FR_passive_array=np.array(FR_passive_array)
BL_active_array=np.array(BL_active_array)
BL_passive_array=np.array(BL_passive_array)
delta_FR_active_array=np.array(delta_FR_active_array)
delta_FR_passive_array=np.array(delta_FR_passive_array)
NB_corr_active_array=np.array(NB_corr_active_array)
NB_corr_passive_array=np.array(NB_corr_passive_array)


#%%Histogram of correlations
group_list=[increase_cells_active_inx,decrease_cells_active_inx,increase_cells_passive_inx,decrease_cells_passive_inx,other_cells_inx]
cell_list_title_array=['increase cells active','decrease cells active','increase cells passive','decrease cells passive','other cells'] 

active_means=[] #The mean of each group for active trials
passive_means=[] #for passive trials
active_sems=[] #The mean of each group for active trials
passive_sems=[] #for passive trials
bin_array=np.arange(-0.8,0.9,0.1)
for group_inx in np.arange(len(group_list)):
    NB_corr_active_array_group=NB_corr_active_array[group_list[group_inx]]
    NB_corr_passive_array_group=NB_corr_passive_array[group_list[group_inx]]
    
    NB_mean_active=np.nanmean(NB_corr_active_array_group)
    NB_mean_passive=np.nanmean(NB_corr_passive_array_group)

    NB_sem_active=scipy.stats.sem(NB_corr_active_array_group,nan_policy='omit')
    NB_sem_passive=scipy.stats.sem(NB_corr_passive_array_group,nan_policy='omit')
    
    active_means.append(NB_mean_active)
    passive_means.append(NB_mean_passive)

    active_sems.append(NB_sem_active)
    passive_sems.append(NB_sem_passive)

    plt.hist(NB_corr_active_array_group,bins=bin_array,alpha=0.4,density=(True),color='C0')
    plt.hist(NB_corr_passive_array_group,bins=bin_array,alpha=0.4,density=(True),color='C1')
    plt.axvline(NB_mean_active,color='C0')
    plt.axvline(NB_mean_passive,color='C1')
    plt.ylabel('density')
    plt.xlabel('Neural-behavior correlations')
    plt.title(cell_list_title_array[group_inx])
    plt.legend(['active trials','passive trials'])
    plt.show()

fig, ax = plt.subplots()
Width=0.25
x=(np.arange(len(active_means)))    
ax.bar(x,active_means,yerr=active_sems, width = Width)
ax.bar(x+Width,passive_means,yerr=passive_sems, width = Width)
ax.legend(['active trials','passive trials'])
ax.axhline(0,color='black')
ax.set_xticklabels(['x','increase \n active','decrease \n active','increase \n passive','decrease \n passive','others'])
ax.set_title('Average neural-behaviour correlation')
plt.show()


#%% Correlation between the Neural-behaviour correlation and the delta FR (FR_MO-FR_baseline)



# group_list=[increase_cells_active_inx,decrease_cells_active_inx,increase_cells_passive_inx,decrease_cells_passive_inx,other_cells_inx]
# cell_list_title_array=['increase cells active','decrease cells active','increase cells passive','decrease cells passive','other cells']        


group_list=[increase_cells_active_inx,increase_cells_passive_inx]
cell_list_title_array=['increase cells active','increase cells passive']   

#group_list=[decrease_cells_active_inx,decrease_cells_passive_inx]
#cell_list_title_array=['decrease cells active','decrease cells passive']  
 
fig, ax = plt.subplots(2,len(group_list))
for group_inx in np.arange(len(group_list)):
    n_groups=20

    #scatter delta active vs NB CC
    delta_FR_active_array_group=delta_FR_active_array[group_list[group_inx]]
    NB_corr_active_array_group=NB_corr_active_array[group_list[group_inx]]
    ordered_inxs=np.argsort(delta_FR_active_array_group)
    delta_FR_active_array_ordered=delta_FR_active_array_group[ordered_inxs]
    NB_corr_active_array_ordered=NB_corr_active_array_group[ordered_inxs]
    inxs=np.ceil(np.linspace(0,len(NB_corr_active_array_ordered),n_groups+1))
    x=[]
    y_sem=[]
    y=[]
    for bin_inx in np.arange(np.size(inxs)-1):
        begin_inx=int(inxs[bin_inx])
        end_inx=int(inxs[bin_inx+1])

        x.append(np.nanmean(delta_FR_active_array_ordered[begin_inx:end_inx]))
        y.append(np.nanmean(NB_corr_active_array_ordered[begin_inx:end_inx]))
        y_sem.append(scipy.stats.sem(NB_corr_active_array_ordered[begin_inx:end_inx], axis=0,nan_policy='omit'))
    cur_corr=np.round(np.corrcoef(x,y)[0,1],2)
    cur_pval=np.round(stats.pearsonr(x, y)[1],3)
    ax[0,group_inx].errorbar(x,y,yerr=y_sem,fmt="o",color='red')
    ax[0,group_inx].scatter(np.nanmean(delta_FR_active_array_ordered),np.nanmean(NB_corr_active_array_ordered),marker='*',color='black')   
    ax[0,group_inx].axvline(0,color='black')
    ax[0,group_inx].axhline(0,color='black')
    ax[0,group_inx].set_xlabel('delta FR active',fontsize=6)
    ax[0,group_inx].tick_params(axis='x', labelsize=6)
    ax[0,group_inx].tick_params(axis='y', labelsize=6)
    ax[0,group_inx].set_ylabel('NB_corr_active',fontsize=6)
    ax[0,group_inx].set_title(cell_list_title_array[group_inx]+'\n active trials\n'+' r='+str(cur_corr)+ ' p='+str(cur_pval),fontsize=6)

    #scatter delta passive vs NB CC
    delta_FR_passive_array_group=delta_FR_passive_array[group_list[group_inx]]
    NB_corr_passive_array_group=NB_corr_passive_array[group_list[group_inx]]
    ordered_inxs=np.argsort(delta_FR_passive_array_group)
    delta_FR_passive_array_ordered=delta_FR_passive_array_group[ordered_inxs]
    NB_corr_passive_array_ordered=NB_corr_passive_array_group[ordered_inxs]
    inxs=np.ceil(np.linspace(0,len(NB_corr_passive_array_ordered),n_groups+1))
    x=[]
    y_sem=[]
    y=[]
    for bin_inx in np.arange(np.size(inxs)-1):
        begin_inx=int(inxs[bin_inx])
        end_inx=int(inxs[bin_inx+1])

        x.append(np.nanmean(delta_FR_passive_array_ordered[begin_inx:end_inx]))
        y.append(np.nanmean(NB_corr_passive_array_ordered[begin_inx:end_inx]))
        y_sem.append(scipy.stats.sem(NB_corr_passive_array_ordered[begin_inx:end_inx], axis=0,nan_policy='omit'))
    cur_corr=np.round(np.corrcoef(x,y)[0,1],2)
    cur_pval=np.round(stats.pearsonr(x, y)[1],3)
    ax[1,group_inx].errorbar(x,y,yerr=y_sem,fmt="o",color='red')
    ax[1,group_inx].scatter(np.nanmean(delta_FR_passive_array_ordered),np.nanmean(NB_corr_passive_array_ordered),marker='*',color='black')
    ax[1,group_inx].axvline(0,color='black')
    ax[1,group_inx].axhline(0,color='black')
    ax[1,group_inx].set_xlabel('delta FR passive',fontsize=6)
    ax[1,group_inx].set_ylabel('NB_corr_passive',fontsize=6)
    ax[1,group_inx].tick_params(axis='x', labelsize=6)
    ax[1,group_inx].tick_params(axis='y', labelsize=6)
    ax[1,group_inx].set_title(cell_list_title_array[group_inx]+'\n passive trials\n'+' r='+str(cur_corr)+ ' p='+str(cur_pval),fontsize=6)
fig.tight_layout()
plt.show()
