# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 17:32:53 2021

@author: Owner
"""
from __future__ import print_function
from glob import glob
import pickle
import os
import pandas as pd
import numpy as np
import re
import scipy.io
import scipy.stats as stats
from scipy.stats import f_oneway
from scipy.stats import pearsonr
import math
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from mat4py import loadmat
import sys
from neuron_class import cell_task,filtCells,load_cell_task,smooth_data
import time




#Find cells significant for cue onset (dir is not a criteria) -- Output: sig_cells - list with IDs of sig cells
#get list of cellID (int) with relevant task
def find_sig_cells_dishabituation_baseline_cue():
    cell_task_py_folder="C:/Users/Owner/Google Drive/Documents/Studies/Mati's Lab/FEF learning project/Data_Analyses/units_task_python/"
    task='Dishabituation_100_25_cue'
    cell_list=os.listdir(cell_task_py_folder+task) #list of strings
    cell_list=[int(item) for item in cell_list] #list of ints
    
    #Filter cells
    cell_db_file="C:/Users/Owner/Google Drive/Documents/Studies/Mati's Lab/FEF learning project/Data_Analyses/fiona_cells_db.xlsx"
    dictFilterCells ={'cell_ID':cell_list,'session':'filterOff','grade':8, 'n_trials':300}
    cell_list_filtered=filtCells(cell_db_file,dictFilterCells)
    
    #Parameters for FR calculation
    window_pre_cue=100
    window_post_cue=300
    window_length_cue=window_post_cue-window_pre_cue
    
    window_pre_baseline=-100
    window_post_baseline=100
    window_length_baseline=window_post_baseline-window_pre_baseline
    
    CR_VALUE=0.05
    sig_cells_cue=[]
    
    dictFilterTrials = {'dir':'filterOff', 'trial_name':'filterOff', 'fail':0, 'after_fail':0}
    for cell_inx,cell_ID in enumerate(cell_list_filtered):
        cur_cell_task=load_cell_task(cell_task_py_folder,task,cell_ID)
        trials_df=cur_cell_task.filtTrials(dictFilterTrials)
        
        #CUE
        trials_df.loc[:,'mean_FR_cue'] = trials_df.apply(lambda row: (row.spike_times>(row.cue_onset+window_pre_cue)) & (row.spike_times<(row.cue_onset+window_post_cue)), axis = 1)
        #return a vector with all spike times in the window
        trials_df.loc[:,'mean_FR_cue'] = trials_df.apply(lambda row: (row.spike_times[row.mean_FR_cue]), axis = 1)
        #calculate average FR within the window
        trials_df.loc[:,'mean_FR_cue'] = trials_df.apply(lambda row: 1000*(row.mean_FR_cue).size/window_length_cue, axis = 1) 
        cur_cell_cue_FR=np.mean(trials_df.loc[:,'mean_FR_cue'].values)
        
        #BASELINE
        #return a boolean vector with same length as spike times
        trials_df.loc[:,'mean_FR_pre_cue'] = trials_df.apply(lambda row: (row.spike_times>(row.cue_onset+window_pre_baseline)) & (row.spike_times<(row.cue_onset+window_post_baseline)), axis = 1)
        #return a vector with all spike times in the window
        trials_df.loc[:,'mean_FR_pre_cue'] = trials_df.apply(lambda row: (row.spike_times[row.mean_FR_pre_cue]), axis = 1)
        #calculate average FR within the window
        trials_df.loc[:,'mean_FR_pre_cue'] = trials_df.apply(lambda row: 1000*(row.mean_FR_pre_cue).size/window_length_baseline, axis = 1)
        cur_cell_baseline_FR=np.mean(trials_df.loc[:,'mean_FR_pre_cue'].values)
        
         #Check significance for cue vs baseline
        t_test_cue=scipy.stats.ttest_rel(trials_df.loc[:,'mean_FR_pre_cue'], trials_df.loc[:,'mean_FR_cue'])
        if t_test_cue[1]<CR_VALUE:
            sig_cells_cue.append(cell_ID)
            
            #cur_cell_task.PSTH({"timePoint":"cue_onset","timeBefore":-200,"timeAfter":400},dictFilterTrials,plot_option=1,smooth_option=1)
    
            #plt.plot(cur_cell_baseline_FR,cur_cell_cue_FR,'o',color='blue')
       # else:    
            #plt.plot(cur_cell_baseline_FR,cur_cell_cue_FR,'o',color='black')
            
    # plt.xlabel('baseline')
    # plt.ylabel('cue')
    # x=np.linspace(0, 100, 1000)
    # plt.plot(x,x, color='red')
    # plt.show()
    
    return sig_cells_cue

##########################
#For all sig cells make an histogram of active-passive distribution and a scatter plot passive vs active
sig_cells_cue=find_sig_cells_dishabituation_baseline_cue()  
cell_task_py_folder="C:/Users/Owner/Google Drive/Documents/Studies/Mati's Lab/FEF learning project/Data_Analyses/units_task_python/"
task='Dishabituation_100_25_cue'
#Parameters for FR calculation
window_pre_cue=100
window_post_cue=300
window_length_cue=window_post_cue-window_pre_cue
cell_mean_FR=[]
cell_active_FR=[]
cell_passive_FR=[]
for cell_inx,cell_ID in enumerate(sig_cells_cue):
    cur_cell_task=load_cell_task(cell_task_py_folder,task,cell_ID)
    
    
    #A.Mean firing of the cell across condition (active+passive)
    dictFilterTrials  = {'dir':'filterOff', 'trial_name':'filterOff', 'fail':0,'after_fail':0}
    trials_df=cur_cell_task.filtTrials(dictFilterTrials)
    trials_df.loc[:,'mean_FR_cue'] = trials_df.apply(lambda row: (row.spike_times>(row.cue_onset+window_pre_cue)) & (row.spike_times<(row.cue_onset+window_post_cue)), axis = 1)
    #return a vector with all spike times in the window
    trials_df.loc[:,'mean_FR_cue'] = trials_df.apply(lambda row: (row.spike_times[row.mean_FR_cue]), axis = 1)
    #calculate average FR within the window
    trials_df.loc[:,'mean_FR_cue'] = trials_df.apply(lambda row: 1000*(row.mean_FR_cue).size/window_length_cue, axis = 1)
    cur_cell_mean_FR=np.mean(trials_df.loc[:,'mean_FR_cue'].values)
    cell_mean_FR.append(cur_cell_mean_FR) #average FR of all trials in the trials_df
    
    #B.Mean firing of the cell for active trials
    dictFilterTrials  = {'dir':'filterOff', 'trial_name':'v20NS', 'fail':0,'after_fail':0}
    trials_df_active=cur_cell_task.filtTrials(dictFilterTrials)
    trials_df_active.loc[:,'mean_FR_cue'] = trials_df_active.apply(lambda row: (row.spike_times>(row.cue_onset+window_pre_cue)) & (row.spike_times<(row.cue_onset+window_post_cue)), axis = 1)
    #return a vector with all spike times in the window
    trials_df_active.loc[:,'mean_FR_cue'] = trials_df_active.apply(lambda row: (row.spike_times[row.mean_FR_cue]), axis = 1)
    #calculate average FR within the window
    trials_df_active.loc[:,'mean_FR_cue'] = trials_df_active.apply(lambda row: 1000*(row.mean_FR_cue).size/window_length_cue, axis = 1)  
    #cur_cell_active_FR=(np.mean(trials_df_active.loc[:,'mean_FR_cue'].values)-cur_cell_mean_FR)
    cur_cell_active_FR=np.mean(trials_df_active.loc[:,'mean_FR_cue'].values)
    cell_active_FR.append(cur_cell_active_FR)
    
    #C.Mean firing of the cell for passive trials
    dictFilterTrials  = {'dir':'filterOff', 'trial_name':'v20S', 'fail':0,'after_fail':0}
    trials_df_passive=cur_cell_task.filtTrials(dictFilterTrials)
    trials_df_passive.loc[:,'mean_FR_cue'] = trials_df_passive.apply(lambda row: (row.spike_times>(row.cue_onset+window_pre_cue)) & (row.spike_times<(row.cue_onset+window_post_cue)), axis = 1)
    #return a vector with all spike times in the window
    trials_df_passive.loc[:,'mean_FR_cue'] = trials_df_passive.apply(lambda row: (row.spike_times[row.mean_FR_cue]), axis = 1)
    #calculate average FR within the window
    trials_df_passive.loc[:,'mean_FR_cue'] = trials_df_passive.apply(lambda row: 1000*(row.mean_FR_cue).size/window_length_cue, axis = 1)  
    #cur_cell_passive_FR=(np.mean(trials_df_passive.loc[:,'mean_FR_cue'].values)-cur_cell_mean_FR)
    cur_cell_passive_FR=np.mean(trials_df_passive.loc[:,'mean_FR_cue'].values)
    cell_passive_FR.append(cur_cell_passive_FR)
    
    #Check if cell is significant (active vs passive)
    ttest_single_cell=scipy.stats.ttest_ind(trials_df_passive.loc[:,'mean_FR_cue'],trials_df_active.loc[:,'mean_FR_cue'])
    p_val_single_cell=ttest_single_cell[1]
    if p_val_single_cell>0.05:
        plt.plot(cur_cell_passive_FR,cur_cell_active_FR,'o',color='blue')
    else:
       plt.plot(cur_cell_passive_FR,cur_cell_active_FR,'o',color='black')
 
#ttest for pop active vs passive population
ttest_pop=scipy.stats.ttest_rel(cell_passive_FR,cell_active_FR)
p_val_pop=ttest_pop[1]
diff_passive_active=np.array(cell_passive_FR)-np.array(cell_active_FR)
      
#scatter plot: each point is a cell and shows mean passive FR vs mean active FR       
x=np.linspace(0, 100, 1000)
plt.plot(x,x, color='red')
plt.xlabel('FR in passive trials (Hz)')
plt.ylabel('FR in active trials (Hz)')
plt.title('cue onset:'+ str(window_pre_cue)+'-'+str(window_post_cue)+ 'ms ' + 'pvalue:'+str(round(p_val_pop,3)))
plt.show()

#Histogram of passive-active difference for cells
plt.hist(x=diff_passive_active, bins='auto', color='b',alpha=0.7, rwidth=0.85,density=1,stacked=1)
plt.xlabel('Active-passive')
plt.ylabel('Probability')
plt.title('Diffrence in FR after cue onset')
#plt.xlim(-6,6)
plt.show()       
 ##################################      


##############
#PSTH of list of cells passive vs active

cell_task_py_folder="C:/Users/Owner/Google Drive/Documents/Studies/Mati's Lab/FEF learning project/Data_Analyses/units_task_python/"
task='Dishabituation_100_25_cue'
cell_list=os.listdir(cell_task_py_folder+task) #list of strings
cell_list=[int(item) for item in cell_list] #list of ints


#Filter cells
cell_db_file="C:/Users/Owner/Google Drive/Documents/Studies/Mati's Lab/FEF learning project/Data_Analyses/fiona_cells_db.xlsx"
dictFilterCells ={'cell_ID':cell_list,'session':'filterOff','grade':8, 'n_trials':300}
cell_list_filtered=filtCells(cell_db_file,dictFilterCells)
    
sig_cells_cue=find_sig_cells_dishabituation_baseline_cue()  

not_sig_cells_cue=[x for x in cell_list_filtered if x not in sig_cells_cue]

cur_cell_list=cell_list_filtered #cell_list filtered or sig_cells_cue or not_sig_cells_cue

window={"timePoint":"motion_onset","timeBefore":-500,"timeAfter":800}
window_length=window['timeAfter']-window['timeBefore']
Pop_psth_active=np.empty([len(cur_cell_list),window_length])
Pop_psth_passive=np.empty([len(cur_cell_list),window_length])

for cell_inx,cell_ID in enumerate(cur_cell_list):
            cur_cell_task=load_cell_task(cell_task_py_folder,task,cell_ID)
            dictFilterTrials  = {'dir':'filterOff', 'trial_name':'v20NS', 'fail':0,'after_fail':0}
            PSTH_active=cur_cell_task.PSTH(window,dictFilterTrials,plot_option=0,smooth_option=1)
            Pop_psth_active[cell_inx,:]=PSTH_active
            
            dictFilterTrials  = {'dir':'filterOff', 'trial_name':'v20S', 'fail':0,'after_fail':0}
            PSTH_passive=cur_cell_task.PSTH(window,dictFilterTrials,plot_option=0,smooth_option=1)
            Pop_psth_passive[cell_inx,:]=PSTH_passive


PSTH_mean_active=np.mean(Pop_psth_active,axis=0)   
PSTH_mean_passive=np.mean(Pop_psth_passive,axis=0)   
xaxis=np.arange(window['timeBefore'],window['timeAfter'],1)
plt.plot(xaxis,PSTH_mean_active)
plt.plot(xaxis,PSTH_mean_passive)
plt.xticks(np.arange(window['timeBefore'],window['timeAfter'],100))
plt.xlabel('time (ms)')
plt.ylabel('FR (Hz)')
plt.title('population '+window['timePoint'])
plt.xlim(window['timeBefore']+100,window['timeAfter']-100)
plt.legend(['active','passive'])
plt.axvline(x=0, color='r')
plt.show()
###############################

######## Percentage of sig cells for cue vs baseline significant for passive vs active within bins around cue 
sig_cells_cue=find_sig_cells_dishabituation_baseline_cue()  
cur_cell_list=sig_cells_cue
cell_task_py_folder="C:/Users/Owner/Google Drive/Documents/Studies/Mati's Lab/FEF learning project/Data_Analyses/units_task_python/"
task='Dishabituation_100_25_cue'
p_sig_cells=[]
window_middle=[]
sig_cells_list=[]
sig_cells_dict={} #sig cells for active vs passive for each bin
for window_pre in range(-200,600,50):
     n_sig_cells=0
     sig_cells_list_bin=[]
    
     window_post=window_pre+200
     window_middle.append((window_pre+window_post)/2)
     cur_bin=(window_pre+window_post)/2
       
     for cell_ID in cur_cell_list:
         cur_cell_task=load_cell_task(cell_task_py_folder,task,cell_ID)
         trial_df=cur_cell_task.trials_df
        
         #Mean_FR parameter for significant
         window_length=window_post-window_pre
         dictFilterTrials  = {'dir':'filterOff', 'trial_name':'filterOff', 'fail':0, 'after_fail':0} 
         trial_df=cur_cell_task.filtTrials(dictFilterTrials)

         trial_df.loc[:,'mean_FR_cue'] = trial_df.apply(lambda row: (row.spike_times>(row.cue_onset+window_pre)) & (row.spike_times<(row.cue_onset+window_post)), axis = 1)
         #return a vector with all spike times in the window
         trial_df.loc[:,'mean_FR_cue'] = trial_df.apply(lambda row: (row.spike_times[row.mean_FR_cue]), axis = 1)
         #calculate average FR within the window
         trial_df.loc[:,'mean_FR_cue'] = trial_df.apply(lambda row: 1000*(row.mean_FR_cue).size/window_length, axis = 1) 

        
         FR_active=trial_df['mean_FR_cue'].loc[trial_df['trial_name'].str.contains('v20NS')]
         FR_passive=trial_df['mean_FR_cue'].loc[trial_df['trial_name'].str.contains('v20S')]
         ttest_out=scipy.stats.ttest_ind(FR_active,FR_passive)
         p_val=ttest_out[1]
         if p_val<0.05:
             n_sig_cells=n_sig_cells+1
             sig_cells_list_bin.append(cell_ID)
             sig_cells_list.append(cell_ID)
     sig_cells_dict[cur_bin]=sig_cells_list_bin
     p_sig_cells.append(n_sig_cells/len(cur_cell_list))
sig_cells_list=list((sig_cells_list))   
plt.plot(window_middle,p_sig_cells) 

plt.axvline(x=0, color='r')
plt.xlabel('bin_center')
plt.ylabel('% sig_cells')
plt.title('active vs passive-cue')   
plt.show()



######## Percentage of sig cells for cue vs baseline significant for passive vs active within bins around motion onset
sig_cells_cue=find_sig_cells_dishabituation_baseline_cue()  
cur_cell_list=sig_cells_cue
cell_task_py_folder="C:/Users/Owner/Google Drive/Documents/Studies/Mati's Lab/FEF learning project/Data_Analyses/units_task_python/"
task='Dishabituation_100_25_cue'
p_sig_cells=[]
window_middle=[]
sig_cells_list=[]
sig_cells_dict={} #sig cells for active vs passive for each bin
for window_pre in range(-300,900,50):
     n_sig_cells=0
     sig_cells_list_bin=[]
    
     window_post=window_pre+200
     window_middle.append((window_pre+window_post)/2)
     cur_bin=(window_pre+window_post)/2
       
     for cell_ID in cur_cell_list:
         cur_cell_task=load_cell_task(cell_task_py_folder,task,cell_ID)
         trial_df=cur_cell_task.trials_df
        
         #Mean_FR parameter for significant
         window_length=window_post-window_pre
         dictFilterTrials  = {'dir':'filterOff', 'trial_name':'filterOff', 'fail':0, 'after_fail':0} 
         trial_df=cur_cell_task.filtTrials(dictFilterTrials)

         trial_df.loc[:,'mean_FR_motion'] = trial_df.apply(lambda row: (row.spike_times>(row.motion_onset+window_pre)) & (row.spike_times<(row.motion_onset+window_post)), axis = 1)
         #return a vector with all spike times in the window
         trial_df.loc[:,'mean_FR_motion'] = trial_df.apply(lambda row: (row.spike_times[row.mean_FR_motion]), axis = 1)
         #calculate average FR within the window
         trial_df.loc[:,'mean_FR_motion'] = trial_df.apply(lambda row: 1000*(row.mean_FR_motion).size/window_length, axis = 1) 

        
         FR_active=trial_df['mean_FR_motion'].loc[trial_df['trial_name'].str.contains('v20S')]
         FR_passive=trial_df['mean_FR_motion'].loc[trial_df['trial_name'].str.contains('v20NS')]
         ttest_out=scipy.stats.ttest_ind(FR_active,FR_passive)
         p_val=ttest_out[1]
         if p_val<0.05:
             n_sig_cells=n_sig_cells+1
             sig_cells_list_bin.append(cell_ID)
             sig_cells_list.append(cell_ID)
     sig_cells_dict[cur_bin]=sig_cells_list_bin
     p_sig_cells.append(n_sig_cells/len(cur_cell_list))
sig_cells_list=list((sig_cells_list))   
plt.plot(window_middle,p_sig_cells) 

plt.axvline(x=0, color='r')
plt.xlabel('bin_center')
plt.ylabel('% sig_cells')
plt.title('active vs passive-motion')   
plt.show()
   

############ Modulation in FR between active and passive around motion onset (Scatter plot)
cell_task_py_folder="C:/Users/Owner/Google Drive/Documents/Studies/Mati's Lab/FEF learning project/Data_Analyses/units_task_python/"
task='Dishabituation_100_25_cue'
cell_list=os.listdir(cell_task_py_folder+task) #list of strings
cell_list=[int(item) for item in cell_list] #list of ints
#Filter cells
cell_db_file="C:/Users/Owner/Google Drive/Documents/Studies/Mati's Lab/FEF learning project/Data_Analyses/fiona_cells_db.xlsx"
dictFilterCells ={'cell_ID':cell_list,'session':'filterOff','grade':8, 'n_trials':300}
cell_list_filtered=filtCells(cell_db_file,dictFilterCells)

sig_cells_cue=find_sig_cells_dishabituation_baseline_cue()  
not_sig_cells_cue=[x for x in cell_list_filtered if x not in sig_cells_cue]
cur_cell_list=cell_list_filtered #cell_lust filtered or sig_cells_cue or not_sig_cells

window_pre_motion=200
window_post_motion=700
window_length_motion=window_post_motion-window_pre_motion

window_pre_baseline=-300
window_post_baseline=-100
window_length_baseline=window_post_baseline-window_pre_baseline
    
sig_cells_active_passive=[]
active_arr=[]
passive_arr=[]
for cell_inx,cell_ID in enumerate(cur_cell_list):
    cur_cell_task=load_cell_task(cell_task_py_folder,task,cell_ID)
    
    #BASELINE
    dictFilterTrials  = {'dir':'filterOff', 'trial_name':'filterOff', 'fail':0, 'after_fail':0}
    trials_df=cur_cell_task.filtTrials(dictFilterTrials)
    #return a boolean vector with same length as spike times
    trials_df.loc[:,'mean_FR_pre_motion'] = trials_df.apply(lambda row: (row.spike_times>(row.motion_onset+window_pre_baseline)) & (row.spike_times<(row.motion_onset+window_post_baseline)), axis = 1)
    #return a vector with all spike times in the window
    trials_df.loc[:,'mean_FR_pre_motion'] = trials_df.apply(lambda row: (row.spike_times[row.mean_FR_pre_motion]), axis = 1)
    #calculate average FR within the window
    trials_df.loc[:,'mean_FR_pre_motion'] = trials_df.apply(lambda row: 1000*(row.mean_FR_pre_motion).size/window_length_baseline, axis = 1)
    cur_cell_baseline_FR=np.mean(trials_df.loc[:,'mean_FR_pre_motion'].values)
    
    #Active
    dictFilterTrials  = {'dir':'filterOff', 'trial_name':'v20NS', 'fail':0, 'after_fail':0}
    trials_df_active=cur_cell_task.filtTrials(dictFilterTrials)
    #return a boolean vector with same length as spike times
    trials_df_active.loc[:,'mean_FR_motion'] = trials_df_active.apply(lambda row: (row.spike_times>(row.motion_onset+window_pre_motion)) & (row.spike_times<(row.motion_onset+window_post_motion)), axis = 1)
    #return a vector with all spike times in the window
    trials_df_active.loc[:,'mean_FR_motion'] = trials_df_active.apply(lambda row: (row.spike_times[row.mean_FR_motion]), axis = 1)
    #calculate average FR within the window
    trials_df_active.loc[:,'mean_FR_motion'] = trials_df_active.apply(lambda row: 1000*(row.mean_FR_motion).size/window_length_motion, axis = 1)
    cur_cell_FR_active=np.mean(trials_df_active.loc[:,'mean_FR_motion'].values)-cur_cell_baseline_FR
        
    #Passive
    dictFilterTrials  = {'dir':'filterOff', 'trial_name':'v20S', 'fail':0, 'after_fail':0} 
    trials_df_passive=cur_cell_task.filtTrials(dictFilterTrials) 
    #return a boolean vector with same length as spike times
    trials_df_passive.loc[:,'mean_FR_motion'] = trials_df_passive.apply(lambda row: (row.spike_times>(row.motion_onset+window_pre_motion)) & (row.spike_times<(row.motion_onset+window_post_motion)), axis = 1)
    #return a vector with all spike times in the window
    trials_df_passive.loc[:,'mean_FR_motion'] = trials_df_passive.apply(lambda row: (row.spike_times[row.mean_FR_motion]), axis = 1)
    #calculate average FR within the window
    trials_df_passive.loc[:,'mean_FR_motion'] = trials_df_passive.apply(lambda row: 1000*(row.mean_FR_motion).size/window_length_motion, axis = 1)
    cur_cell_FR_passive=np.mean(trials_df_passive.loc[:,'mean_FR_motion'].values)-cur_cell_baseline_FR

    try:
        stat=scipy.stats.mannwhitneyu(trials_df_active.loc[:,'mean_FR_motion'].values,trials_df_passive.loc[:,'mean_FR_motion'].values)
    except:
        print('problematic cell:'+str(cell_ID))
    p_val=stat[1]
    
    if cell_ID in sig_cells_cue:
       cur_marker='*'
    else:
       cur_marker='.'
        
    if p_val<0.05:
        cur_color='red'
        sig_cells_active_passive.append(cell_ID)
    else:
        cur_color='blue'

    plt.plot(cur_cell_FR_active,cur_cell_FR_passive,marker=cur_marker,color=cur_color)
    active_arr.append(cur_cell_FR_active)
    passive_arr.append(cur_cell_FR_passive)

lim=40
plt.xlabel('$\Delta$ active')
plt.ylabel('$\Delta$ passive')
plt.title('active vs passive around motion onset: '+str (window_pre_motion)+'-'+str (window_post_motion)+'ms ')
x=np.linspace(-lim, lim, 1000)
plt.plot(x,x, color='black')
plt.xlim(-lim,lim)
plt.ylim(-lim,lim)
plt.axvline(x=0, color='black')
plt.hlines(0,-lim,lim, color='black')
plt.show()
stat, p = stats.wilcoxon(active_arr, passive_arr)

#Analyze the difference between active and passive for cells with high modullation toward passive trials during motion
modulation_FR=[x - y for x,y in zip(passive_arr,active_arr)]
inx_sorted=np.argsort(-np.array(modulation_FR)) #minus is to get it in descending order
window={"timePoint":"motion_onset","timeBefore":-500,"timeAfter":800}
for x in range(9):
    cell_ID=cur_cell_list[inx_sorted[x]]
    cur_cell_task=load_cell_task(cell_task_py_folder,task,cell_ID)
    dictFilterTrials  = {'dir':'filterOff', 'trial_name':'v20NS', 'fail':0,'after_fail':0}
    psth_active=cur_cell_task.PSTH(window,dictFilterTrials,plot_option=0,smooth_option=1)
    
    dictFilterTrials  = {'dir':'filterOff', 'trial_name':'v20S', 'fail':0,'after_fail':0}
    psth_passive=cur_cell_task.PSTH(window,dictFilterTrials,plot_option=0,smooth_option=1)
    
    xaxis=np.arange(window['timeBefore'],window['timeAfter'],1)
    plt.plot(xaxis,psth_active)
    plt.plot(xaxis,psth_passive)
    plt.title(str(cell_ID))
    plt.xlim(window['timeBefore']+100,window['timeAfter']-100)
    plt.xlabel('time (ms)')
    plt.ylabel('FR (Hz)')
    plt.legend(['active','passive'])
    plt.axvline(x=0, color='r')  
    plt.show()  



######################### PSTH of cells sig for each bin 
cell_task_py_folder="C:/Users/Owner/Google Drive/Documents/Studies/Mati's Lab/FEF learning project/Data_Analyses/units_task_python/"
task='Dishabituation_100_25_cue'
cell_list=os.listdir(cell_task_py_folder+task) #list of strings
cell_list=[int(item) for item in cell_list] #list of ints
#Filter cells
cell_db_file="C:/Users/Owner/Google Drive/Documents/Studies/Mati's Lab/FEF learning project/Data_Analyses/fiona_cells_db.xlsx"
dictFilterCells ={'cell_ID':cell_list,'session':'filterOff','grade':8, 'n_trials':300}
cell_list_filtered=filtCells(cell_db_file,dictFilterCells)

sig_cells_cue=find_sig_cells_dishabituation_baseline_cue()  
not_sig_cells_cue=[x for x in cell_list_filtered if x not in sig_cells_cue]
sig_cells_cue=find_sig_cells_dishabituation_baseline_cue()  
cur_cell_list=sig_cells_cue


p_sig_cells=[]
window_middle=[]
sig_cells_list=[]
sig_cells_dict={} #sig cells for active vs passive for each bin
sig_cells_dict_bin={}
for window_pre in range(-300,900,50):
     sig_cells_list_bin=[]
     n_sig_cells=0
    
     window_post=window_pre+200
     window_middle.append((window_pre+window_post)/2)
     cur_bin=(window_pre+window_post)/2
       
     for cell_ID in cur_cell_list:
         cur_cell_task=load_cell_task(cell_task_py_folder,task,cell_ID)
         trial_df=cur_cell_task.trials_df
        
         #Mean_FR parameter for significant
         window_length=window_post-window_pre
         dictFilterTrials  = {'dir':'filterOff', 'trial_name':'filterOff', 'fail':0, 'after_fail':0} 
         trial_df=cur_cell_task.filtTrials(dictFilterTrials)

         trial_df.loc[:,'mean_FR_motion'] = trial_df.apply(lambda row: (row.spike_times>(row.motion_onset+window_pre)) & (row.spike_times<(row.motion_onset+window_post)), axis = 1)
         #return a vector with all spike times in the window
         trial_df.loc[:,'mean_FR_motion'] = trial_df.apply(lambda row: (row.spike_times[row.mean_FR_motion]), axis = 1)
         #calculate average FR within the window
         trial_df.loc[:,'mean_FR_motion'] = trial_df.apply(lambda row: 1000*(row.mean_FR_motion).size/window_length, axis = 1) 

        
         FR_active=trial_df['mean_FR_motion'].loc[trial_df['trial_name'].str.contains('v20S')]
         FR_passive=trial_df['mean_FR_motion'].loc[trial_df['trial_name'].str.contains('v20NS')]
         ttest_out=scipy.stats.ttest_ind(FR_active,FR_passive)
         p_val=ttest_out[1]
         if p_val<0.05:
             n_sig_cells=n_sig_cells+1
             sig_cells_list_bin.append(cell_ID)
             sig_cells_list.append(cell_ID)
     sig_cells_dict[cur_bin]=sig_cells_list_bin
     p_sig_cells.append(n_sig_cells/len(cur_cell_list))
     
     
     
     window={"timePoint":"motion_onset","timeBefore":-500,"timeAfter":800}
     window_length=window['timeAfter']-window['timeBefore']
     Pop_psth_active=np.empty([len(sig_cells_list_bin),window_length])
     Pop_psth_passive=np.empty([len(sig_cells_list_bin),window_length])
    
     for cell_inx,cell_ID in enumerate(sig_cells_list_bin):
                cur_cell_task=load_cell_task(cell_task_py_folder,task,cell_ID)
                dictFilterTrials  = {'dir':'filterOff', 'trial_name':'v20NS', 'fail':0,'after_fail':0}
                PSTH_active=cur_cell_task.PSTH(window,dictFilterTrials,plot_option=0,smooth_option=1)
                Pop_psth_active[cell_inx,:]=PSTH_active
                
                dictFilterTrials  = {'dir':'filterOff', 'trial_name':'v20S', 'fail':0,'after_fail':0}
                PSTH_passive=cur_cell_task.PSTH(window,dictFilterTrials,plot_option=0,smooth_option=1)
                Pop_psth_passive[cell_inx,:]=PSTH_passive
    
    
     PSTH_mean_active=np.mean(Pop_psth_active,axis=0)   
     PSTH_mean_passive=np.mean(Pop_psth_passive,axis=0)   
     xaxis=np.arange(window['timeBefore'],window['timeAfter'],1)
     plt.plot(xaxis,PSTH_mean_active)
     plt.plot(xaxis,PSTH_mean_passive)
     plt.axvline(x=window_pre, color='black')
     plt.axvline(x=window_post, color='black')
     plt.xticks(np.arange(window['timeBefore'],window['timeAfter'],100))
     plt.xlabel('time (ms)')
     plt.ylabel('FR (Hz)')
     plt.title('population '+window['timePoint'])
     plt.xlim(window['timeBefore']+100,window['timeAfter']-100)
     plt.legend(['active','passive'])
     plt.axvline(x=0, color='r')
     plt.show()
         
window={"timePoint":"motion_onset","timeBefore":-500,"timeAfter":800}           
for cell_ID in sig_cells_dict[-100]:
    cur_cell_task=load_cell_task(cell_task_py_folder,task,cell_ID)
    dictFilterTrials  = {'dir':'filterOff', 'trial_name':'v20NS', 'fail':0,'after_fail':0}
    psth_active=cur_cell_task.PSTH(window,dictFilterTrials,plot_option=0,smooth_option=1)
    dictFilterTrials  = {'dir':'filterOff', 'trial_name':'v20S', 'fail':0,'after_fail':0}
    psth_passive=cur_cell_task.PSTH(window,dictFilterTrials,plot_option=0,smooth_option=1)
    
    xaxis=np.arange(window['timeBefore'],window['timeAfter'],1)
    plt.plot(xaxis,psth_active)
    plt.plot(xaxis,psth_passive)
    plt.title(str(cell_ID))
    plt.xlim(window['timeBefore']+100,window['timeAfter']-100)
    plt.xlabel('time (ms)')
    plt.ylabel('FR (Hz)')
    plt.show()    
    

######################################   



######################## Visuomotor index
cell_task_py_folder="C:/Users/Owner/Google Drive/Documents/Studies/Mati's Lab/FEF learning project/Data_Analyses/units_task_python/"
task='Dishabituation_100_25_cue'

cell_list=os.listdir(cell_task_py_folder+task) #list of strings
cell_list=[int(item) for item in cell_list] #list of ints
#Filter cells
cell_db_file="C:/Users/Owner/Google Drive/Documents/Studies/Mati's Lab/FEF learning project/Data_Analyses/fiona_cells_db.xlsx"
dictFilterCells ={'cell_ID':cell_list,'session':'filterOff','grade':8, 'n_trials':300}
cell_list_filtered=filtCells(cell_db_file,dictFilterCells)

sig_cells_cue=find_sig_cells_dishabituation_baseline_cue()  
not_sig_cells_cue=[x for x in cell_list_filtered if x not in sig_cells_cue]
sig_cells_cue=find_sig_cells_dishabituation_baseline_cue()  
cur_cell_list=cell_list_filtered


window_pre_baseline=-300
window_post_baseline=-100
window_length_baseline=window_post_baseline-window_pre_baseline

window_pre_motion=100
window_post_motion=300
window_length_motion=window_post_motion-window_pre_motion

window_pre_cue=100
window_post_cue=300
window_length_cue=window_post_cue-window_pre_cue

VM_inx_arr=[]    
AP_diff_arr=[]

for cell_ID in cur_cell_list:
    cur_cell_task=load_cell_task(cell_task_py_folder,task,cell_ID)
    #BASELINE_CUE
    dictFilterTrials  = {'dir':'filterOff', 'trial_name':'v20NS', 'fail':0, 'after_fail':0}
    trials_df=cur_cell_task.filtTrials(dictFilterTrials)
    #return a boolean vector with same length as spike times
    trials_df.loc[:,'mean_FR_pre_cue'] = trials_df.apply(lambda row: (row.spike_times>(row.cue_onset+window_pre_baseline)) & (row.spike_times<(row.cue_onset+window_post_baseline)), axis = 1)
    #return a vector with all spike times in the window
    trials_df.loc[:,'mean_FR_pre_cue'] = trials_df.apply(lambda row: (row.spike_times[row.mean_FR_pre_cue]), axis = 1)
    #calculate average FR within the window
    trials_df.loc[:,'mean_FR_pre_cue'] = trials_df.apply(lambda row: 1000*(row.mean_FR_pre_cue).size/window_length_baseline, axis = 1)
    cur_cell_baseline_cue_FR=np.mean(trials_df.loc[:,'mean_FR_pre_cue'].values)


    #BASELINE_MOTOR
    dictFilterTrials  = {'dir':'filterOff', 'trial_name':'v20NS', 'fail':0, 'after_fail':0}
    trials_df=cur_cell_task.filtTrials(dictFilterTrials)
    #return a boolean vector with same length as spike times
    trials_df.loc[:,'mean_FR_pre_motion'] = trials_df.apply(lambda row: (row.spike_times>(row.motion_onset+window_pre_baseline)) & (row.spike_times<(row.motion_onset+window_post_baseline)), axis = 1)
    #return a vector with all spike times in the window
    trials_df.loc[:,'mean_FR_pre_motion'] = trials_df.apply(lambda row: (row.spike_times[row.mean_FR_pre_motion]), axis = 1)
    #calculate average FR within the window
    trials_df.loc[:,'mean_FR_pre_motion'] = trials_df.apply(lambda row: 1000*(row.mean_FR_pre_motion).size/window_length_baseline, axis = 1)
    cur_cell_baseline_motor_FR=np.mean(trials_df.loc[:,'mean_FR_pre_motion'].values)
        
    
    #Visual FR
    dictFilterTrials  = {'dir':'filterOff', 'trial_name':'v20NS', 'fail':0, 'after_fail':0}
    trials_df_active=cur_cell_task.filtTrials(dictFilterTrials)
    #return a boolean vector with same length as spike times
    trials_df_active.loc[:,'mean_FR_cue'] = trials_df_active.apply(lambda row: (row.spike_times>(row.cue_onset+window_pre_cue)) & (row.spike_times<(row.cue_onset+window_post_cue)), axis = 1)
    #return a vector with all spike times in the window
    trials_df_active.loc[:,'mean_FR_cue'] = trials_df_active.apply(lambda row: (row.spike_times[row.mean_FR_cue]), axis = 1)
    #calculate average FR within the window
    trials_df_active.loc[:,'mean_FR_cue'] = trials_df_active.apply(lambda row: 1000*(row.mean_FR_cue).size/window_length_cue, axis = 1)
    #cur_cell_FR_visual=np.mean(trials_df_active.loc[:,'mean_FR_cue'].values)-cur_cell_baseline_cue_FR
    cur_cell_FR_visual=np.mean(trials_df_active.loc[:,'mean_FR_cue'].values)
    
    #Motor FR
    dictFilterTrials  = {'dir':'filterOff', 'trial_name':'filterOff', 'fail':0, 'after_fail':0}
    trials_df_active=cur_cell_task.filtTrials(dictFilterTrials)
    #return a boolean vector with same length as spike times
    trials_df_active.loc[:,'mean_FR_motion'] = trials_df_active.apply(lambda row: (row.spike_times>(row.motion_onset+window_pre_motion)) & (row.spike_times<(row.motion_onset+window_post_motion)), axis = 1)
    #return a vector with all spike times in the window
    trials_df_active.loc[:,'mean_FR_motion'] = trials_df_active.apply(lambda row: (row.spike_times[row.mean_FR_motion]), axis = 1)
    #calculate average FR within the window
    trials_df_active.loc[:,'mean_FR_motion'] = trials_df_active.apply(lambda row: 1000*(row.mean_FR_motion).size/window_length_motion, axis = 1)
    #cur_cell_FR_motor=np.mean(trials_df_active.loc[:,'mean_FR_motion'].values)-cur_cell_baseline_motor_FR
    cur_cell_FR_motor=np.mean(trials_df_active.loc[:,'mean_FR_motion'].values)
    
    VM_inx=(cur_cell_FR_motor-cur_cell_FR_visual)/(cur_cell_FR_motor+cur_cell_FR_visual)
    VM_inx_arr.append(VM_inx)

    window_pre_motion=200
    window_post_motion=700
    window_length_motion=window_post_motion-window_pre_motion
    #BASELINE Motor
    dictFilterTrials  = {'dir':'filterOff', 'trial_name':'filterOff', 'fail':0, 'after_fail':0}
    trials_df=cur_cell_task.filtTrials(dictFilterTrials)
    #return a boolean vector with same length as spike times
    trials_df.loc[:,'mean_FR_pre_motion'] = trials_df.apply(lambda row: (row.spike_times>(row.motion_onset+window_pre_baseline)) & (row.spike_times<(row.motion_onset+window_post_baseline)), axis = 1)
    #return a vector with all spike times in the window
    trials_df.loc[:,'mean_FR_pre_motion'] = trials_df.apply(lambda row: (row.spike_times[row.mean_FR_pre_motion]), axis = 1)
    #calculate average FR within the window
    trials_df.loc[:,'mean_FR_pre_motion'] = trials_df.apply(lambda row: 1000*(row.mean_FR_pre_motion).size/window_length_baseline, axis = 1)
    cur_cell_baseline_FR=np.mean(trials_df.loc[:,'mean_FR_pre_motion'].values)


    #Active
    dictFilterTrials  = {'dir':'filterOff', 'trial_name':'v20NS', 'fail':0, 'after_fail':0}
    trials_df_active=cur_cell_task.filtTrials(dictFilterTrials)
    #return a boolean vector with same length as spike times
    trials_df_active.loc[:,'mean_FR_motion'] = trials_df_active.apply(lambda row: (row.spike_times>(row.motion_onset+window_pre_motion)) & (row.spike_times<(row.motion_onset+window_post_motion)), axis = 1)
    #return a vector with all spike times in the window
    trials_df_active.loc[:,'mean_FR_motion'] = trials_df_active.apply(lambda row: (row.spike_times[row.mean_FR_motion]), axis = 1)
    #calculate average FR within the window
    trials_df_active.loc[:,'mean_FR_motion'] = trials_df_active.apply(lambda row: 1000*(row.mean_FR_motion).size/window_length_motion, axis = 1)
    cur_cell_FR_active=np.mean(trials_df_active.loc[:,'mean_FR_motion'].values)-cur_cell_baseline_FR
        
    #Passive
    dictFilterTrials  = {'dir':'filterOff', 'trial_name':'v20S', 'fail':0, 'after_fail':0} 
    trials_df_passive=cur_cell_task.filtTrials(dictFilterTrials) 
    #return a boolean vector with same length as spike times
    trials_df_passive.loc[:,'mean_FR_motion'] = trials_df_passive.apply(lambda row: (row.spike_times>(row.motion_onset+window_pre_motion)) & (row.spike_times<(row.motion_onset+window_post_motion)), axis = 1)
    #return a vector with all spike times in the window
    trials_df_passive.loc[:,'mean_FR_motion'] = trials_df_passive.apply(lambda row: (row.spike_times[row.mean_FR_motion]), axis = 1)
    #calculate average FR within the window
    trials_df_passive.loc[:,'mean_FR_motion'] = trials_df_passive.apply(lambda row: 1000*(row.mean_FR_motion).size/window_length_motion, axis = 1)
    cur_cell_FR_passive=np.mean(trials_df_passive.loc[:,'mean_FR_motion'].values)-cur_cell_baseline_FR
    
    AP_diff_arr.append(cur_cell_FR_passive-cur_cell_FR_active)
        
#plt.hist(VM_inx_arr)
corr, pval =scipy.stats.spearmanr(VM_inx_arr, AP_diff_arr)
plt.plot(VM_inx_arr,AP_diff_arr,'o')
plt.xlabel('VM_inx')
plt.ylabel('passive - active (Hz)')
plt.title('corr:'+str(round(corr,2)))

plt.show()


#######