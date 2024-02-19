#This script shows that in washout the previous trial (active or passive) influences the subsequent trial at the cell level.

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
path="C:/Users/Owner/Google Drive/Documents/Studies/Mati's Lab/FEF learning project/Data_Analyses/Code analyzes"
os.chdir(path) 
import pandas as pd
import numpy as np
import re
import scipy.io
import scipy.stats as stats
from scipy.stats import f_oneway
import math
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.lines as lines
from scipy.ndimage import gaussian_filter1d
from mat4py import loadmat
import sys
from neuron_class import *
import time
import warnings
from scipy.signal import savgol_filter
from scipy.stats import kruskal
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import copy
import seaborn as sb
from cancel_pupil_bias import *
warnings.filterwarnings("ignore")
from collections import Counter
########################################################

#For all this program:
cell_task_py_folder="C:/Users/Owner/Google Drive/Documents/Studies/Mati's Lab/FEF learning project/Data_Analyses/units_task_python/"
behaviour_py_folder="C:/Users/Owner/Google Drive/Documents/Studies/Mati's Lab/FEF learning project/Data_Analyses/behaviour_python/"
cells_db_excel="C:/Users/Owner/Google Drive/Documents/Studies/Mati's Lab/FEF learning project/Data_Analyses/fiona_cells_db.xlsx"
cells_db=pd.read_excel(cells_db_excel)

behaviour_db_excel="C:/Users/Owner/Google Drive/Documents/Studies/Mati's Lab/FEF learning project/Data_Analyses/fiona_behaviour_db.xlsx"
behaviour_db=pd.read_excel(behaviour_db_excel)
########################################################
#This function receives a cell ID (int) and a task (string) and returns the number of trials performed by this cell during this task
#cells_db and behaviour db must be data frame!
def n_trials_per_task(cell_ID,task,behaviour_db,cells_db):


    #find session,file_begin and file end of the cell:
    session=cells_db[cells_db['cell_ID']==cell_ID]['session'].item()
    fb_cell=cells_db[cells_db['cell_ID']==cell_ID]['fb_after_stablility'].item()
    fe_cell=cells_db[cells_db['cell_ID']==cell_ID]['fe_after_stability'].item()
    
    #look at session in behaviour db
    cur_behaviour_db=behaviour_db[behaviour_db['behaviour_session']==session]
    cur_behaviour_db=cur_behaviour_db[cur_behaviour_db['Task']==task]
    
    n_trials=0
    for index, row in cur_behaviour_db.iterrows():

        if (fe_cell<row['file_begin']) or (fb_cell>row['file_end']) :
            continue
        elif (fb_cell<=row['file_begin']) and (fe_cell>=row['file_begin']) and (fe_cell<=row['file_end']):
            n_trials=n_trials+(fe_cell-row['file_begin'])
        
        elif  (fb_cell<=row['file_begin']) and (fe_cell>=row['file_end']):
            n_trials=n_trials+(row['file_end']-row['file_begin'])
        
        elif (fb_cell>=row['file_begin']) and (fb_cell<=row['file_end']) and (fe_cell<=row['file_end']):     
            n_trials=n_trials+(fe_cell-fb_cell)
        
        elif (fb_cell>=row['file_begin']) and (fb_cell<=row['file_end']) and (fe_cell>=row['file_end']): 
            n_trials=n_trials+(row['file_end']-fb_cell)
    
    return n_trials
########################################################
########################################################
#This function returns a list of cells with a grade above 8 and recorded during at least n_min_saccade trials in saccades and n_min_pursuit trials in pursuit (including mimic)
def get_mapping_cells(n_min_pursuit=80):
    cell_task_py_folder="C:/Users/Owner/Google Drive/Documents/Studies/Mati's Lab/FEF learning project/Data_Analyses/units_task_python/"
    
    cells_db_excel="C:/Users/Owner/Google Drive/Documents/Studies/Mati's Lab/FEF learning project/Data_Analyses/fiona_cells_db.xlsx"
    cells_db=pd.read_excel(cells_db_excel)
    
    behaviour_db_excel="C:/Users/Owner/Google Drive/Documents/Studies/Mati's Lab/FEF learning project/Data_Analyses/fiona_behaviour_db.xlsx"
    behaviour_db=pd.read_excel(behaviour_db_excel)
    
    #Find cells recorded during pursuit
    pursuit_tasks=['8dir_active_passive_interleaved','8dir_active_passive_interleaved_100_25']
    pursuit_cells=[]
    mapping_cells=[]
    for task in pursuit_tasks:
            #All cells in the task folder
            cell_list=os.listdir(cell_task_py_folder+task) #list of strings
            cell_list=[int(item) for item in cell_list] #list of ints
            #All cells recording during mapping
            pursuit_cells=pursuit_cells+[ elem for elem in cell_list if elem  not in [7430,7420,7423,7432,7481,7688,7730,7656,7396,7401,7855]]


    for cell_ID in pursuit_cells:
        n_trials_pursuit=n_trials_per_task(cell_ID,pursuit_tasks[0],behaviour_db,cells_db)+n_trials_per_task(cell_ID,pursuit_tasks[1],behaviour_db,cells_db)

        if  n_trials_pursuit>n_min_pursuit:
            mapping_cells.append(cell_ID)
    return mapping_cells  
########################################################
#####################################
def find_cell_entire_block(tasks,cells_db=cells_db,behaviour_db=behaviour_db):    

    #keep only rows in behaviour db from motor blocks
    task_blocks_db=behaviour_db[behaviour_db['Task'].str.contains('|'.join(tasks))]
    
    #get cell_task_block a list of cell that were recorded during a block of the current task
    cell_task_block=[]
    cell_list_all_block=[] #cells that were recorded during a WHOLE block of the current task
    
    #Find cells that were recorde during at least some part of some task in tasks
    for task in tasks:
        cur_cell_list=os.listdir(cell_task_py_folder+task) #list of strings
        cell_task_block=cell_task_block+[int(item) for item in cur_cell_list] #list of ints
        
    #Find first and last trial of the cells  
    dict_cell_entire_block={}  
    dict_cell_entire_block['cell_ID']=[]
    dict_cell_entire_block['task']=[]
    dict_cell_entire_block['session']=[]
    dict_cell_entire_block['file_begin']=[]
    dict_cell_entire_block['file_end']=[] 
     
    for cell_ID in cell_task_block:
        #find the first and last file where the cell was recorded
        cur_cell_fb=cells_db.loc[cells_db['cell_ID']==int(cell_ID),'fb_after_stablility'].item()
        cur_cell_fe=cells_db.loc[cells_db['cell_ID']==int(cell_ID),'fe_after_stability'].item()
        #find the session of the current cell
        cur_cell_session=cells_db.loc[cells_db['cell_ID']==int(cell_ID),'session'].item()
        
        #find the first and last file of the tasks in the cur cell session
        task_blocks_db_session=task_blocks_db[task_blocks_db['behaviour_session']==cur_cell_session]
        for index, row in task_blocks_db_session.iterrows():
            if (cur_cell_fb<= row['file_begin']) & (cur_cell_fe >= row['file_end']):
                cell_list_all_block.append(cell_ID)
                dict_cell_entire_block['cell_ID'].append(cell_ID)
                dict_cell_entire_block['task'].append(row['Task'])
                dict_cell_entire_block['session'].append(row['behaviour_session'])
                dict_cell_entire_block['file_begin'].append(row['file_begin'])
                dict_cell_entire_block['file_end'].append(row['file_end'])
    df_cells_entire_block=pd.DataFrame.from_dict(dict_cell_entire_block)            
            
    return df_cells_entire_block          
#####################################
# Find cells recorded in both motor, congruent and incongruent task:
motor_tasks=['Motor_learning_CW_100_25_cue','Motor_learning_CCW_100_25_cue','Motor_learning_CW','Motor_learning_CCW']
for task in motor_tasks:
        motor_cells=list(find_cell_entire_block(motor_tasks,cells_db=cells_db,behaviour_db=behaviour_db).cell_ID)

congruent_tasks=['fixation_right_probes_CW_100_25_cue','fixation_right_probes_CCW_100_25_cue','fixation_right_probes_CW','fixation_right_probes_CCW']
for task in congruent_tasks:
        congruent_cells=list(find_cell_entire_block(congruent_tasks,cells_db=cells_db,behaviour_db=behaviour_db).cell_ID)
        
incongruent_tasks=['fixation_wrong_probes_CW_100_25_cue','fixation_wrong_probes_CCW_100_25_cue','fixation_wrong_probes_CW','fixation_wrong_probes_CCW']
for task in incongruent_tasks:
        incongruent_cells=list(find_cell_entire_block(incongruent_tasks,cells_db=cells_db,behaviour_db=behaviour_db).cell_ID)
mapping_cells= get_mapping_cells(n_min_pursuit=80)  
complete_cells=[x for x in congruent_cells if x in motor_cells and x in incongruent_cells and x in mapping_cells] 
mapping_tasks=['8dir_active_passive_interleaved_100_25','8dir_active_passive_interleaved']
#####################################

#####################################
##Build array of learned_cells and base_cells 
active_base_cells=[]
active_base_cells_sig=[]
active_learned_cells=[]
active_learned_cells_sig=[]

passive_base_cells=[]
passive_base_cells_sig=[]
passive_learned_cells=[]
passive_learned_cells_sig=[]

#For each cell find the base and learned direction 
for cell_ID in complete_cells:

    #cell db :find session, File begin and file end of current cell
    fb=cells_db.loc[cells_db['cell_ID']==cell_ID,'fb_after_stablility'].item()
    fe=cells_db.loc[cells_db['cell_ID']==cell_ID,'fe_after_stability'].item()
    session=cells_db.loc[cells_db['cell_ID']==cell_ID,'session'].item()
       
    #behaviour_db-find base and learned direction
    cur_behaviour_db=behaviour_db.loc[behaviour_db['behaviour_session']==session ,: ] #keep only relevant day
    inxs=cur_behaviour_db.loc[cur_behaviour_db['file_begin']>fb ,: ].index #keep only relevant session
    cur_behaviour_db=cur_behaviour_db.loc[([inxs.to_list()[0]-1])+inxs.to_list() ,: ] #keep only relevant session
    cur_behaviour_db=cur_behaviour_db.loc[cur_behaviour_db['file_end']<=fe ,: ]
  
    #Find screen rotation during pursuit (should be 0 but there are mistakes)    
    pursuit_behaviour_db=cur_behaviour_db.loc[cur_behaviour_db['Task'].str.contains('active_passive'),'screen_rotation'] #deals with case where a cell was recorded during two pursuit session (rare)
    SR_pursuit=pursuit_behaviour_db.iloc[0]

    cur_behaviour_db=cur_behaviour_db.loc[cur_behaviour_db['Task'].str.contains('CW') ,: ] #remove dishabituations rows
    base_dir=cur_behaviour_db.iloc[0]['screen_rotation']
    if 'CCW' in cur_behaviour_db.iloc[0]['Task']:
        learned_dir=(base_dir+90)%360
    elif 'CW' in cur_behaviour_db.iloc[0]['Task']:   
        learned_dir=(base_dir-90)%360


    #Look for the FR in those directions in mapping in active and passive trials
    for cur_task in mapping_tasks:
        try:
            cur_cell_task=load_cell_task(cell_task_py_folder,cur_task,cell_ID)
        except:
            continue
    #active base
    dictFilterTrials_base_active={'dir':(base_dir-SR_pursuit)%360, 'trial_name':'v20a', 'fail':0, 'after_fail':'filterOff','saccade_motion':'filterOff'}    
    FR_active_base=cur_cell_task.get_mean_FR_event(dictFilterTrials_base_active,'motion_onset',window_pre=0,window_post=750)
    FR_active_base_mean=np.mean(np.array(FR_active_base))
    #active learned
    dictFilterTrials_learned_active={'dir':(learned_dir-SR_pursuit)%360, 'trial_name':'v20a', 'fail':0, 'after_fail':'filterOff','saccade_motion':'filterOff'}    
    FR_active_learned=cur_cell_task.get_mean_FR_event(dictFilterTrials_learned_active,'motion_onset',window_pre=0,window_post=750)
    FR_active_learned_mean=np.mean(np.array(FR_active_learned))
    #passive base
    dictFilterTrials_base_passive={'dir':(base_dir-SR_pursuit)%360, 'trial_name':'v20p', 'fail':0, 'after_fail':'filterOff','saccade_motion':'filterOff'}    
    FR_passive_base=cur_cell_task.get_mean_FR_event(dictFilterTrials_base_passive,'motion_onset',window_pre=0,window_post=750)
    FR_passive_base_mean=np.mean(np.array(FR_passive_base))
    #passive learned
    dictFilterTrials_learned_passive={'dir':(learned_dir-SR_pursuit)%360, 'trial_name':'v20p', 'fail':0, 'after_fail':'filterOff','saccade_motion':'filterOff'}    
    FR_passive_learned=cur_cell_task.get_mean_FR_event(dictFilterTrials_learned_passive,'motion_onset',window_pre=0,window_post=750)
    FR_passive_learned_mean=np.mean(np.array(FR_passive_learned))
    
    #check if the difference is significant
    try:
        stat,p_val_active=scipy.stats.mannwhitneyu(FR_active_base,FR_active_learned)
    except:
        p_val_active=1
        
    try:
        stat,p_val_passive=scipy.stats.mannwhitneyu(FR_passive_base,FR_passive_learned)
    except:
        p_val_passive=1
        
    #Build array of learned_cells and base_cells (cells that prefers base and learned directions) and learned_sig_cells and base_sig_cells
    if FR_active_base_mean>FR_active_learned_mean:
        active_base_cells.append(cell_ID)
        if p_val_active<0.05:
            active_base_cells_sig.append(cell_ID)
    elif FR_active_base_mean<FR_active_learned_mean:
        active_learned_cells.append(cell_ID)
        if p_val_active<0.05:
            active_learned_cells_sig.append(cell_ID)       
    
    if FR_passive_base_mean>FR_passive_learned_mean:
        passive_base_cells.append(cell_ID)
        if p_val_passive<0.05:
            passive_base_cells_sig.append(cell_ID)
    elif FR_passive_base_mean<FR_passive_learned_mean:
        passive_learned_cells.append(cell_ID)
        if p_val_passive<0.05:
            passive_learned_cells_sig.append(cell_ID)  
            
            
            
####################################
#Remove unstable cells by checking correlation with baseline in congruent cells

task_lists=[motor_tasks,congruent_tasks,incongruent_tasks]
cell_lists=[active_base_cells,active_learned_cells,passive_base_cells,passive_learned_cells,active_base_cells_sig,active_learned_cells_sig,passive_base_cells_sig,passive_learned_cells_sig]
unstable_cells=[]
for task_list in task_lists: 
    for cur_cell_list in cell_lists:
        for cell_ID in cur_cell_list:
            for cur_task in task_list:
                try:
                    cur_cell_task=load_cell_task(cell_task_py_folder,cur_task,cell_ID)
                except:
                    continue
                if task_list==motor_tasks:
                    dictFilterTrials={'dir':'filterOff', 'trial_name':'filterOff', 'fail':0, 'after_fail':'filterOff','saccade_motion':'filterOff','trial_begin_end':'filterOff'}
                else:
                    dictFilterTrials={'dir':'filterOff', 'trial_name':'v20S', 'fail':0, 'after_fail':'filterOff','saccade_motion':'filterOff','trial_begin_end':'filterOff'}
                    
                FR_BL=cur_cell_task.get_mean_FR_event(dictFilterTrials,'motion_onset',window_pre=-500,window_post=0)
                corr,pval=stats.pearsonr(FR_BL, np.arange(1,FR_BL.size+1))
    
                if pval<0.05:
                    unstable_cells.append(cell_ID)
                    cur_cell_list.remove(cell_ID)



####################################

#####################################         
#%%
#Loop across cell within array and build average PSTH for cells yhat prefer learn direction vs cell that prefere base direction
#rows: FR in motor, congruent and incongruent task
# columns active and passive trials
#PSTH paramters

dictFilterTrials_active={'dir':'filterOff', 'trial_name':'v20NS', 'fail':0, 'after_fail':'filterOff','saccade_motion':'filterOff'}
dictFilterTrials_passive={'dir':'filterOff', 'trial_name':'v20S', 'fail':0, 'after_fail':'filterOff','saccade_motion':'filterOff'}
filter_margin=100 #for Psth margin
window={"timePoint": 'motion_onset',"timeBefore":-500,"timeAfter":950}
time_course=np.arange(window['timeBefore']+filter_margin,window['timeAfter']-filter_margin)



task_lists=[motor_tasks,congruent_tasks,incongruent_tasks]
#cell_lists=[active_base_cells,active_learned_cells,passive_base_cells,passive_learned_cells]
cell_lists=[active_base_cells_sig,active_learned_cells_sig,passive_base_cells_sig,passive_learned_cells_sig]


fig,ax=plt.subplots(3,2,dpi=600)

for task_inx,cur_task_list in enumerate(task_lists):
    for cell_list_inx,cur_cell_list in enumerate(cell_lists):
        if cell_list_inx>=2: #for passive cells
            cur_dict=dictFilterTrials_passive
            cur_ls='--'
            col_inx=1
        else: #for active cells
            cur_dict=dictFilterTrials_active
            cur_ls='-'
            col_inx=0
        
        PSTH_array=np.zeros((len(cur_cell_list),np.size(time_course)))    
        for cell_inx,cell_ID in enumerate(cur_cell_list):
            for cur_task in cur_task_list:
                try:
                    cur_cell_task=load_cell_task(cell_task_py_folder,cur_task,cell_ID)
                except:
                    continue
                

            PSTH=cur_cell_task.PSTH(window,cur_dict,plot_option=0,smooth_option=1) 
            PSTH=PSTH[filter_margin:-filter_margin]
            PSTH_array[cell_inx,:]=PSTH
            
        #Plot PSTHs
        PSTH_average=np.nanmean(PSTH_array,axis=0)
        ax[task_inx,col_inx].plot(time_course,PSTH_average,ls=cur_ls)

        ax[0,1].legend(['base_cells','learned_cells'],loc='upper left',fontsize='small')
        
        
task_strings=['motor','cong','incong']
trial_type_string=['active','passive']
for ii in range(len(task_lists)):
    ax[ii,0].set_ylabel('FR- '+task_strings[ii])
    for jj in range(2):
        ax[ii,jj].set_ylim((10,25))
        ax[ii,jj].axvline(x=0,color='black')
        ax[ii,jj].axvline(x=250,color='black')
        ax[0,jj].set_title(trial_type_string[jj]+' trials')
        ax[2,jj].set_xlabel('time from MO (ms)')
        
fig.tight_layout()        
plt.show()

####################################


####################################
#%% 
#Differences in PSTH between congruent and incongruent
dictFilterTrials_active={'dir':'filterOff', 'trial_name':'v20NS', 'fail':0, 'after_fail':'filterOff','saccade_motion':'filterOff'}
dictFilterTrials_passive={'dir':'filterOff', 'trial_name':'v20S', 'fail':0, 'after_fail':'filterOff','saccade_motion':'filterOff'}
filter_margin=100 #for Psth margin
window={"timePoint": 'motion_onset',"timeBefore":-300,"timeAfter":950}
time_course=np.arange(window['timeBefore']+filter_margin,window['timeAfter']-filter_margin)

tasks_lists=[motor_tasks,congruent_tasks,incongruent_tasks]
cell_lists=[active_base_cells,active_learned_cells]
cur_dict=dictFilterTrials_active
colors=['black','tab:blue','tab:orange']
for cur_task_inx,cur_task_list in enumerate(tasks_lists):
    PSTH_base_array=np.zeros((len(cell_lists[0]),np.size(time_course)))
    PSTH_learned_array=np.zeros((len(cell_lists[1]),np.size(time_course)))
    cur_color=colors[cur_task_inx]
    for cell_list_inx,cur_cell_list in enumerate(cell_lists):
    
        PSTH_array=np.zeros((len(cur_cell_list),np.size(time_course)))    
        for cell_inx,cell_ID in enumerate(cur_cell_list):
            for cur_task in cur_task_list:
                try:
                    cur_cell_task=load_cell_task(cell_task_py_folder,cur_task,cell_ID)
                except:
                    continue
            PSTH=cur_cell_task.PSTH(window,cur_dict,plot_option=0,smooth_option=1) 
            PSTH=PSTH[filter_margin:-filter_margin]
            PSTH_array[cell_inx,:]=PSTH
        if cell_list_inx==0:
            PSTH_base_array=PSTH_array
        elif cell_list_inx==1:
            PSTH_learned_array=PSTH_array
            
    PSTH_base_average=np.nanmean(PSTH_base_array,axis=0) 
    PSTH_learned_average=np.nanmean(PSTH_learned_array,axis=0)
    PSTH_base_learned=PSTH_base_average-PSTH_learned_average   
    
    plt.plot(time_course,PSTH_base_learned,cur_color)
    
plt.axvline(x=0,color='black')
plt.axvline(x=250,color='black')
plt.axhline(y=0,color='black',ls='--')
plt.xlabel('time from MO')
plt.ylabel('FR')
plt.title('base - learned')
plt.legend(['motor','congruent','incongruent'])
plt.show()
#####################################         
#%%
#Compare same group of cells in active trials in cong  vs incong tasks for a given cell list
# 2 figures (PSTH in time)
# figure 1:PSTH in congruent task vs incongruent task in active trials
# figure 2: same for passive trials


#Active trials
dictFilterTrials_active={'dir':'filterOff', 'trial_name':'v20NS', 'fail':0, 'after_fail':'filterOff','saccade_motion':'filterOff','trial_begin_end':'filterOff'}
filter_margin=100 #for Psth margin
window={"timePoint": 'motion_onset',"timeBefore":-900,"timeAfter":950}
time_course=np.arange(window['timeBefore']+filter_margin,window['timeAfter']-filter_margin)
task_lists=[motor_tasks,congruent_tasks,incongruent_tasks]
colors=['black','tab:blue','tab:orange']
#Choose a list
cell_lists=[active_learned_cells]

for task_inx,cur_task_list in enumerate(task_lists):
    cur_color=colors[task_inx]
    for cell_list_inx,cur_cell_list in enumerate(cell_lists):
        PSTH_array=np.zeros((len(cur_cell_list),np.size(time_course)))    
        for cell_inx,cell_ID in enumerate(cur_cell_list):
            for cur_task in cur_task_list:
                try:
                    cur_cell_task=load_cell_task(cell_task_py_folder,cur_task,cell_ID)
                except:
                    continue
                
            PSTH=cur_cell_task.PSTH(window,dictFilterTrials_active,plot_option=0,smooth_option=1) 
            PSTH=PSTH[filter_margin:-filter_margin]
            PSTH_array[cell_inx,:]=PSTH
        PSTH_average=np.nanmean(PSTH_array,axis=0)
        plt.plot(time_course,PSTH_average,color=cur_color)
plt.legend(['base motor','base congruent','base incongruent'])
plt.axvline(x=0,color='black')
plt.axvline(x=250,color='black')
plt.xlabel('time from MO')
plt.ylabel('FR')
plt.title('active trials')
plt.ylim((10,25))
plt.show()
    
#Passive trials
dictFilterTrials_passive={'dir':'filterOff', 'trial_name':'v20NS', 'fail':0, 'after_fail':'filterOff','saccade_motion':'filterOff','trial_begin_end':'filterOff'}

#Choose a list
cell_lists=[passive_base_cells]

task_lists=[congruent_tasks,incongruent_tasks]
colors=['tab:blue','tab:orange']

for task_inx,cur_task_list in enumerate(task_lists):
    cur_color=colors[task_inx]
    for cell_list_inx,cur_cell_list in enumerate(cell_lists):
        PSTH_array=np.zeros((len(cur_cell_list),np.size(time_course)))    
        for cell_inx,cell_ID in enumerate(cur_cell_list):
            for cur_task in cur_task_list:
                try:
                    cur_cell_task=load_cell_task(cell_task_py_folder,cur_task,cell_ID)
                except:
                    continue
                
            PSTH=cur_cell_task.PSTH(window,dictFilterTrials_passive,plot_option=0,smooth_option=1) 
            PSTH=PSTH[filter_margin:-filter_margin]
            PSTH_array[cell_inx,:]=PSTH
        PSTH_average=np.nanmean(PSTH_array,axis=0)
        plt.plot(time_course,PSTH_average,color=cur_color)
plt.legend(['base congruent','base incongruent'])
plt.axvline(x=0,color='black')
plt.axvline(x=250,color='black')
plt.xlabel('time from MO')
plt.ylabel('FR')
plt.title('passive trials')
plt.ylim((10,25))
plt.show()



#%%
#Compare same group of cells same block at beggining and end of the block
window={"timePoint": 'motion_onset',"timeBefore":-300,"timeAfter":950}
time_course=np.arange(window['timeBefore']+filter_margin,window['timeAfter']-filter_margin)

#choose a cell list
cur_cell_list=passive_learned_cells_sig
#Choose a task
task_list=congruent_tasks
#choose a trial type
cur_trial_type='v20S'
#Choose a number of trials to select a beginning and end of blocks
n_trials=4
PSTH_array_begin=np.zeros((len(cur_cell_list),np.size(time_course))) 
PSTH_array_middle=np.zeros((len(cur_cell_list),np.size(time_course))) 
PSTH_array_end=np.zeros((len(cur_cell_list),np.size(time_course))) 

for cell_inx,cell_ID in enumerate(cur_cell_list):
    for cur_task in task_list:
        try:
            cur_cell_task=load_cell_task(cell_task_py_folder,cur_task,cell_ID)
        except:
            continue
        dictFilterTrials={'dir':'filterOff', 'trial_name':cur_trial_type, 'fail':0, 'after_fail':'filterOff','saccade_motion':'filterOff','trial_begin_end':'filterOff'}
        trials_df=cur_cell_task.filtTrials(dictFilterTrials)
        
        trials_df_index_begin=[trials_df.index.to_list()[0],trials_df.index.to_list()[n_trials-1]]
        dictFilterTrials_begin={'dir':'filterOff', 'trial_name':cur_trial_type, 'fail':0, 'after_fail':'filterOff','saccade_motion':'filterOff','trial_begin_end':trials_df_index_begin}
        trials_df_begin=cur_cell_task.filtTrials(dictFilterTrials_begin)
        PSTH_begin=cur_cell_task.PSTH(window,dictFilterTrials_begin,plot_option=0,smooth_option=1) 
        PSTH_begin=PSTH_begin[filter_margin:-filter_margin]
        PSTH_array_begin[cell_inx,:]=PSTH_begin

        #all the trials between begin and end
        trials_df_index_middle=[trials_df.index.to_list()[n_trials-1],trials_df.index.to_list()[-n_trials]]
        dictFilterTrials_middle={'dir':'filterOff', 'trial_name':cur_trial_type, 'fail':0, 'after_fail':'filterOff','saccade_motion':'filterOff','trial_begin_end':trials_df_index_middle}
        trials_df_middle=cur_cell_task.filtTrials(dictFilterTrials_middle)
        PSTH_middle=cur_cell_task.PSTH(window,dictFilterTrials_middle,plot_option=0,smooth_option=1)
        PSTH_middle=PSTH_middle[filter_margin:-filter_margin]
        PSTH_array_middle[cell_inx,:]=PSTH_middle


        trials_df_index_end=[trials_df.index.to_list()[-n_trials],trials_df.index.to_list()[-1]]
        dictFilterTrials_end={'dir':'filterOff', 'trial_name':cur_trial_type, 'fail':0, 'after_fail':'filterOff','saccade_motion':'filterOff','trial_begin_end':trials_df_index_end}
        trials_df_end=cur_cell_task.filtTrials(dictFilterTrials_end)
        PSTH_end=cur_cell_task.PSTH(window,dictFilterTrials_end,plot_option=0,smooth_option=1)
        PSTH_end=PSTH_end[filter_margin:-filter_margin]
        PSTH_array_end[cell_inx,:]=PSTH_end

PSTH_average_begin=np.nanmean(PSTH_array_begin,axis=0)
PSTH_average_middle=np.nanmean(PSTH_array_middle,axis=0)
PSTH_average_end=np.nanmean(PSTH_array_end,axis=0)

PSTH_sem_begin=np.std(PSTH_array_begin,axis=0)/(np.size(PSTH_array_begin,0)**0.5)
PSTH_sem_middle=np.std(PSTH_array_middle,axis=0)/(np.size(PSTH_array_middle,0)**0.5)
PSTH_sem_end=np.std(PSTH_array_end,axis=0)/(np.size(PSTH_array_end,0)**0.5)

# plt.errorbar(time_course,PSTH_average_begin,PSTH_sem_begin,color='tab:blue')
# plt.errorbar(time_course,PSTH_average_middle,PSTH_sem_middle,color='tab:green')
# plt.errorbar(time_course,PSTH_average_end,PSTH_sem_end,color='tab:orange')  
plt.plot(time_course,PSTH_average_begin,color='tab:blue')
plt.plot(time_course,PSTH_average_middle,color='tab:green')
plt.plot(time_course,PSTH_average_end,color='tab:orange')  
plt.legend(['begin','middle','end'])
plt.axvline(x=0,color='black')
plt.axvline(x=250,color='black')
plt.xlabel('time from MO')
plt.ylabel('FR')
plt.show()


#%% learning curves

#choose a cell list
cur_cell_list=active_learned_cells
#Choose a task
task_list=motor_tasks
#choose a trial type
cur_trial_type='v20NS'

#For congruent/incongruent task:
if cur_trial_type=='v20NS':
    n_trials=8
elif cur_trial_type=='v20S':
    n_trials=72    

#for motor tasks:
if task_list==motor_tasks:
    cur_trial_type='v20NS'
    n_trials=80

dictFilterTrials={'dir':'filterOff', 'trial_name':cur_trial_type, 'fail':0, 'after_fail':'filterOff','saccade_motion':'filterOff','trial_begin_end':'filterOff'}
FR_array=np.zeros((n_trials,len(cur_cell_list)))
for cell_inx,cell_ID in enumerate(cur_cell_list):
    for cur_task in task_list:
        try:
            cur_cell_task=load_cell_task(cell_task_py_folder,cur_task,cell_ID)
        except:
            continue
        
        FR_cell=cur_cell_task.get_mean_FR_event(dictFilterTrials,'motion_onset',window_pre=100,window_post=300)
        FR_array[:,cell_inx]=np.array(FR_cell)
learning_curve=np.mean(FR_array,axis=1)
plt.plot(learning_curve)
plt.title('Learning curve')
plt.xlabel('# trial')
plt.ylabel('FR')
plt.show()


