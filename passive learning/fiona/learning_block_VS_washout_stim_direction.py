#this script compares a learning block to the dishabituation before and after it
#It shows a PSTH in time and scatter plots 

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
######################################################

# This function is similar to find_cell_entire_block but returns the cells that were recorded during the whole DISHABITUATION block
#subsequent to some group ou tasks
#rel_block tells whether to chhose the previous or next dishabituation block relative to the task
def find_cell_entire_dis_block(tasks,cells_db=cells_db,behaviour_db=behaviour_db,rel_block='previous'):
    #rows in the behaviour db where the cell was recorded
    tasks_db_inx=behaviour_db[behaviour_db['Task'].str.contains('|'.join(tasks))].index
    
    if rel_block=='previous':
        rel_inx=-1
    elif rel_block=='next':
        rel_inx=1
    else:
        print ('Choose a correct rel_block argument. previous or next' )
    dis_post_tasks_inx=tasks_db_inx+rel_inx
    dis_post_tasks_db=behaviour_db.loc[dis_post_tasks_inx,:] #extract rows from behaviour db_which are before/after motor blocks
    
    if rel_block=='next':
        dis_post_tasks_db=dis_post_tasks_db[dis_post_tasks_db['Task'].str.contains('Dishabituation')]
        dis_post_tasks_db=dis_post_tasks_db[dis_post_tasks_db['paradigm'].str.contains('passive_learning')]
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
        dis_post_tasks_db_session=dis_post_tasks_db[dis_post_tasks_db['behaviour_session']==cur_cell_session]
        for index, row in dis_post_tasks_db_session.iterrows(): #for all the dishabituation block recorded with that cell 
            if (cur_cell_fb<= row['file_begin']) & (cur_cell_fe >= row['file_end']):   
                cell_list_all_block.append(cell_ID)
                dict_cell_entire_block['cell_ID'].append(cell_ID)
                dict_cell_entire_block['task'].append(row['Task'])
                dict_cell_entire_block['session'].append(row['behaviour_session'])
                dict_cell_entire_block['file_begin'].append(row['file_begin'])
                dict_cell_entire_block['file_end'].append(row['file_end'])
    df_cells_entire_block=pd.DataFrame.from_dict(dict_cell_entire_block)        
    return df_cells_entire_block
#################################################################################    
    


congruent_tasks=['fixation_right_probes_CW_100_25_cue','fixation_right_probes_CCW_100_25_cue','fixation_right_probes_CW','fixation_right_probes_CCW']
incongruent_tasks=['fixation_wrong_probes_CW_100_25_cue','fixation_wrong_probes_CCW_100_25_cue','fixation_wrong_probes_CW','fixation_wrong_probes_CCW']
motor_tasks=['Motor_learning_CW_100_25_cue','Motor_learning_CCW_100_25_cue','Motor_learning_CW','Motor_learning_CCW']


#Choose a learning block you want to analyze:
cur_task_list=motor_tasks

#Find cells recorded during whole learning block, the dishabituation around it and 80 trials during mapping 
learning_block_cells=list(find_cell_entire_block(cur_task_list,cells_db=cells_db,behaviour_db=behaviour_db).cell_ID)

dis_pre_cells_pd=(find_cell_entire_dis_block(cur_task_list,cells_db=cells_db,behaviour_db=behaviour_db,rel_block='previous'))
dis_pre_cells=list(dis_pre_cells_pd.cell_ID)

dis_next_cells_pd=(find_cell_entire_dis_block(cur_task_list,cells_db=cells_db,behaviour_db=behaviour_db,rel_block='next'))
dis_next_cells=list(dis_next_cells_pd.cell_ID)

mapping_cells= get_mapping_cells(n_min_pursuit=80)  
complete_cells=[x for x in learning_block_cells if x in dis_pre_cells and x in dis_next_cells and x in mapping_cells] 

mapping_tasks=['8dir_active_passive_interleaved_100_25','8dir_active_passive_interleaved']
#####################################

#####################################


saccade_option=1 #cancel trials with saccades that begin between 0 and 300 after motion onset
#saccade_option='filterOff' #doesnt cancel trials with saccades that begin between 0 and 300 after motion onset

active_stim_cells=[]
active_stim_cells_sig=[]
passive_stim_cells=[]
passive_stim_cells_sig=[]
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
        stim_dir=(base_dir+45)%360
    elif 'CW' in cur_behaviour_db.iloc[0]['Task']:   
        stim_dir=(base_dir-45)%360

    #Look for the FR in those directions in mapping in active and passive trials
    for cur_task in mapping_tasks:
        try:
            cur_cell_task=load_cell_task(cell_task_py_folder,cur_task,cell_ID)
        except:
            continue
    #Window we look at in the mapping:
    window_begin_MO=100
    window_end_MO=600
    
    window_begin_BL=-800
    window_end_BL=0
    #MO_stim_active
    dictFilterTrials_base_active={'dir':(stim_dir-SR_pursuit)%360, 'trial_name':'v20a', 'fail':0, 'after_fail':'filterOff','saccade_motion':saccade_option}    
    FR_active_MO_stim=cur_cell_task.get_mean_FR_event(dictFilterTrials_base_active,'motion_onset',window_pre=window_begin_MO,window_post=window_end_MO)
    FR_active_MO_stim_mean=np.mean(np.array(FR_active_MO_stim))

    #BL_stim_active
    dictFilterTrials_base_active={'dir':(stim_dir-SR_pursuit)%360, 'trial_name':'v20a', 'fail':0, 'after_fail':'filterOff','saccade_motion':saccade_option}    
    FR_active_BL_stim=cur_cell_task.get_mean_FR_event(dictFilterTrials_base_active,'motion_onset',window_pre=window_begin_BL,window_post=window_end_BL)
    FR_active_BL_stim_mean=np.mean(np.array(FR_active_BL_stim))

    #MO_stim_passive
    dictFilterTrials_base_passive={'dir':(stim_dir-SR_pursuit)%360, 'trial_name':'v20p', 'fail':0, 'after_fail':'filterOff','saccade_motion':saccade_option}    
    FR_passive_MO_stim=cur_cell_task.get_mean_FR_event(dictFilterTrials_base_active,'motion_onset',window_pre=window_begin_MO,window_post=window_end_MO)
    FR_passive_MO_stim_mean=np.mean(np.array(FR_passive_MO_stim))

    #BL_stim_active
    dictFilterTrials_base_active={'dir':(stim_dir-SR_pursuit)%360, 'trial_name':'v20p', 'fail':0, 'after_fail':'filterOff','saccade_motion':saccade_option}    
    FR_passive_BL_stim=cur_cell_task.get_mean_FR_event(dictFilterTrials_base_active,'motion_onset',window_pre=window_begin_BL,window_post=window_end_BL)
    FR_passive_BL_stim_mean=np.mean(np.array(FR_passive_BL_stim))
    
    #check if the difference is significant
    try:
        stat,p_val_active=scipy.stats.mannwhitneyu(FR_active_MO_stim,FR_active_BL_stim)
    except:
        p_val_active=1

    try:
        stat,p_val_passive=scipy.stats.mannwhitneyu(FR_passive_MO_stim,FR_passive_BL_stim)
    except:
        p_val_passive=1        

    #Build array of learned_cells and base_cells (cells that prefers base and learned directions) and learned_sig_cells and base_sig_cells
    if FR_active_MO_stim_mean>FR_active_BL_stim_mean:
        active_stim_cells.append(cell_ID)
        if p_val_active<0.05:
            active_stim_cells_sig.append(cell_ID)
    if FR_passive_MO_stim_mean>FR_passive_BL_stim_mean:
        passive_stim_cells.append(cell_ID)
        if p_val_passive<0.05:
            passive_stim_cells_sig.append(cell_ID)
 
    #PSTH of cells, base vs learned direction
  #   filter_margin=100 #for Psth margin
  #   window={"timePoint": 'motion_onset',"timeBefore":-500,"timeAfter":950}
  #   time_course=np.arange(window['timeBefore']+filter_margin,window['timeAfter']-filter_margin)
  #   PSTH_base_active=cur_cell_task.PSTH(window,dictFilterTrials_base_active,plot_option=0,smooth_option=1)
  #   PSTH_base_passive=cur_cell_task.PSTH(window,dictFilterTrials_base_passive,plot_option=0,smooth_option=1)    
  #   PSTH_learned_active=cur_cell_task.PSTH(window,dictFilterTrials_learned_active,plot_option=0,smooth_option=1)
  #   PSTH_learned_passive=cur_cell_task.PSTH(window,dictFilterTrials_learned_passive,plot_option=0,smooth_option=1)       

  #   PSTH_base_active=PSTH_base_active[filter_margin:-filter_margin]
  #   PSTH_base_passive=PSTH_base_passive[filter_margin:-filter_margin]   
  #   PSTH_learned_active=PSTH_learned_active[filter_margin:-filter_margin]
  #   PSTH_learned_passive=PSTH_learned_passive[filter_margin:-filter_margin]      
    
  #   plt.plot(time_course,PSTH_base_active)
  #   plt.plot(time_course,PSTH_learned_active)
  #   #plt.plot(time_course,PSTH_base_passive)
  # #  plt.plot(time_course,PSTH_learned_passive)
    
  #   plt.title(cell_ID)
  #   plt.legend(['base_active','learned_active','base_passive','learned_passive'])
  #   plt.axvline(x=100,color='black')
  #   plt.axvline(x=300,color='black')
  #   plt.show()
   
#######################################            


#########################################
#Remove unstable cells from the complete cells list
#It concatenates the FR in the dishabituation before, the learning block and then the dishabituation after
#If the FR is correlated with the index of the trials then the cells is removed from the analyze
unstable_cells=[]
for cell_ID in complete_cells:
    for cur_task in cur_task_list:
        try:
            cur_cell_task=load_cell_task(cell_task_py_folder,cur_task,cell_ID)
        except:
            continue
    dictFilterTrials_block={'dir':'filterOff', 'trial_name':'filterOff', 'fail':'filterOff', 'after_fail':'filterOff','saccade_motion':'filterOff'}    
    FR_block=cur_cell_task.get_mean_FR_event(dictFilterTrials_block,'motion_onset',window_pre=-500,window_post=0)
    
        
    dishabituation_tasks=['Dishabituation_100_25_cue','Dishabituation']
    for cur_task in dishabituation_tasks:
        try:
            cur_cell_task=load_cell_task(cell_task_py_folder,cur_task,cell_ID)
        except:
            continue
    
    #Dishabituation before
    fb_pre=dis_pre_cells_pd.loc[dis_pre_cells_pd['cell_ID']==cell_ID,'file_begin'].item()
    fe_pre=dis_pre_cells_pd.loc[dis_pre_cells_pd['cell_ID']==cell_ID,'file_end'].item()
    dictFilterTrials_pre={'dir':'filterOff', 'trial_name':'filterOff', 'fail':'filterOff', 'after_fail':'filterOff','saccade_motion':'filterOff','files_begin_end':[fb_pre,fe_pre]}    
    FR_dis_pre=cur_cell_task.get_mean_FR_event(dictFilterTrials_pre,'motion_onset',window_pre=-500,window_post=0)
    
    #Dishabituation after
    fb_post=dis_next_cells_pd.loc[dis_next_cells_pd['cell_ID']==cell_ID,'file_begin'].item()
    fe_post=dis_next_cells_pd.loc[dis_next_cells_pd['cell_ID']==cell_ID,'file_end'].item()
    dictFilterTrials_post={'dir':'filterOff', 'trial_name':'filterOff', 'fail':'filterOff', 'after_fail':'filterOff','saccade_motion':'filterOff','files_begin_end':[fb_post,fe_post]}    
    FR_dis_post=cur_cell_task.get_mean_FR_event(dictFilterTrials_post,'motion_onset',window_pre=-500,window_post=0)
    
    #concatenation and correlation
    FR_conc=np.concatenate((FR_dis_pre.to_numpy(),FR_block.to_numpy(),FR_dis_post.to_numpy()),axis=0)

    corr,pval=stats.pearsonr(FR_conc, np.arange(1,FR_conc.size+1))
    if pval<0.05:
        # plt.scatter( np.arange(1,FR_dis_pre.size+1),FR_dis_pre)
        # plt.scatter( np.arange(FR_dis_pre.size+1,FR_dis_pre.size+FR_block.size+1),FR_block)
        # plt.scatter( np.arange(FR_dis_pre.size+1+FR_block.size,FR_dis_pre.size+FR_block.size+FR_dis_post.size+1),FR_dis_post)
        # plt.ylabel('FR')
        # plt.xlabel('trial index')
        # plt.legend('dis pre','learning','dis next')
        # plt.title('corr: ' + str(corr)+'pval: ' + str(pval))
        # plt.show()
        
        unstable_cells.append(cell_ID)
#########################################        
#%%
#PSTHs for learning blocks

filter_margin=100 #for Psth margin
window={"timePoint": 'motion_onset',"timeBefore":-500,"timeAfter":950}
time_course=np.arange(window['timeBefore']+filter_margin,window['timeAfter']-filter_margin)

cur_cell_list=active_stim_cells
cur_cell_list=[x for x in cur_cell_list if x not in unstable_cells]

#CHOOSE A TRIAL TYPE
#v20S is unavailable for motor block
trial_type='v20NS'
saccade_option=1 #cancel trials with saccades that begin between 0 and 300 after motion onset
#saccade_option='filterOff'#doesnt cancel trials with saccades that begin between 0 and 300 after motion onset

#initialize psth arrays
PSTH_block_array=np.zeros((len(cur_cell_list),np.size(time_course)))
PSTH_disPre_array=np.zeros((len(cur_cell_list),np.size(time_course)))
PSTH_disNext_array=np.zeros((len(cur_cell_list),np.size(time_course)))

#initialize scatter array
scatter_block_array=[]
scatter_disPre_array=[]
scatter_disNext_array=[]
for cell_inx,cell_ID in enumerate(cur_cell_list):    
    #Learning block
    for cur_task in cur_task_list:
        try:
            cur_cell_task=load_cell_task(cell_task_py_folder,cur_task,cell_ID)
        except:
            continue
    dictFilterTrials_block={'dir':'filterOff', 'trial_name':trial_type, 'fail':0, 'after_fail':'filterOff','saccade_motion':saccade_option,'blink_motion':saccade_option}    
    PSTH=cur_cell_task.PSTH(window,dictFilterTrials_block,plot_option=0,smooth_option=1) 
    PSTH=PSTH[filter_margin:-filter_margin]
    PSTH_block_array[cell_inx,:]=PSTH    
    scatter_block_array.append(np.nanmean(PSTH[-(window['timeBefore']+filter_margin)+100:-(window['timeBefore']+filter_margin)+250]))    

    #Dishabituation
    dishabituation_tasks=['Dishabituation_100_25_cue','Dishabituation']
    for cur_task in dishabituation_tasks:
        try:
            cur_cell_task=load_cell_task(cell_task_py_folder,cur_task,cell_ID)
        except:
            continue
    #Dishabituation before
    fb_pre=dis_pre_cells_pd.loc[dis_pre_cells_pd['cell_ID']==cell_ID,'file_begin'].item()
    fe_pre=dis_pre_cells_pd.loc[dis_pre_cells_pd['cell_ID']==cell_ID,'file_end'].item()
    dictFilterTrials_pre={'dir':'filterOff', 'trial_name':trial_type, 'fail':0, 'after_fail':'filterOff','saccade_motion':saccade_option,'blink_motion':saccade_option,'files_begin_end':[fb_pre,fe_pre]}    
    PSTH=cur_cell_task.PSTH(window,dictFilterTrials_pre,plot_option=0,smooth_option=1) 
    PSTH=PSTH[filter_margin:-filter_margin]
    PSTH_disPre_array[cell_inx,:]=PSTH   
    scatter_disPre_array.append(np.nanmean(PSTH[-(window['timeBefore']+filter_margin)+100:-(window['timeBefore']+filter_margin)+250]))    
    
    #Dishabituation after
    fb_post=dis_next_cells_pd.loc[dis_next_cells_pd['cell_ID']==cell_ID,'file_begin'].item()
    fe_post=dis_next_cells_pd.loc[dis_next_cells_pd['cell_ID']==cell_ID,'file_end'].item()
    dictFilterTrials_post={'dir':'filterOff', 'trial_name':trial_type, 'fail':0, 'after_fail':'filterOff','saccade_motion':saccade_option,'blink_motion':saccade_option,'files_begin_end':[fb_post,fe_post]}    
    PSTH=cur_cell_task.PSTH(window,dictFilterTrials_post,plot_option=0,smooth_option=1) 
    PSTH=PSTH[filter_margin:-filter_margin]
    PSTH_disNext_array[cell_inx,:]=PSTH 
    scatter_disNext_array.append(np.nanmean(PSTH[-(window['timeBefore']-filter_margin)+100:-(window['timeBefore']-filter_margin)+250]))    

#Time dynamics  
PSTH_block=np.nanmean(PSTH_block_array,axis=0)
PSTH_disPre=np.nanmean(PSTH_disPre_array,axis=0)
PSTH_disNext=np.nanmean(PSTH_disNext_array,axis=0)
    
plt.plot(time_course,PSTH_disPre)
plt.plot(time_course,PSTH_block)
plt.plot(time_course,PSTH_disNext)
plt.axvline(x=0,color='black')
plt.axvline(x=250,color='black')
plt.xlabel('time from MO')
plt.ylabel('FR')
plt.title(str(len(cur_cell_list))+' cells')
plt.legend(['dis pre','learning block','dis next'])
#plt.ylim((3,35))
plt.show()    

#Scatter
x=np.arange(0,100,1)
corr,pval=scipy.stats.wilcoxon(scatter_block_array,scatter_disPre_array)
plt.scatter(scatter_block_array,scatter_disPre_array)
plt.plot(x,x,color='red')
plt.xlabel('learning block')
plt.ylabel('dis pre')
plt.title('learning block vs dis pre - pval: '+ str(round(pval,5)))
#plt.xlim((0,60))
#plt.ylim((0,60))
plt.show()

corr,pval=scipy.stats.wilcoxon(scatter_block_array,scatter_disNext_array)
plt.scatter(scatter_block_array,scatter_disNext_array)
plt.plot(x,x,color='red')
plt.xlabel('learning block')
plt.ylabel('dis next')
plt.title('learning block vs dis next - pval: '+ str(round(pval,5)))
#plt.xlim((0,60))
#plt.ylim((0,60))
plt.show()


#%%
#################################################################################    
# cur_cell_list=[x for x in complete_cells if x not in unstable_cells]

# #CHOOSE A TRIAL TYPE
# #v20S is unavailable for motor block
# trial_type='v20NS'
# #saccade_option=1 #cancel trials with saccades that begin between 0 and 300 after motion onset
# saccade_option=1#doesnt cancel trials with saccades that begin between 0 and 300 after motion onset

# filter_margin=100 #for Psth margin
# window={"timePoint": 'motion_onset',"timeBefore":-500,"timeAfter":950}
# time_course=np.arange(window['timeBefore']+filter_margin,window['timeAfter']-filter_margin)


# for cell_inx,cell_ID in enumerate(cur_cell_list):    
#     #Learning block
#     for cur_task in cur_task_list:
#         try:
#             cur_cell_task=load_cell_task(cell_task_py_folder,cur_task,cell_ID)
#         except:
#             continue
#     dictFilterTrials_block={'dir':'filterOff', 'trial_name':trial_type, 'fail':0, 'after_fail':'filterOff','saccade_motion':saccade_option,'blink_motion':saccade_option}    
#     PSTH=cur_cell_task.PSTH(window,dictFilterTrials_block,plot_option=0,smooth_option=1) 
#     PSTH_motor=PSTH[filter_margin:-filter_margin]

#     #Dishabituation
#     dishabituation_tasks=['Dishabituation_100_25_cue','Dishabituation']
#     for cur_task in dishabituation_tasks:
#         try:
#             cur_cell_task=load_cell_task(cell_task_py_folder,cur_task,cell_ID)
#         except:
#             continue
#     #Dishabituation before
#     fb_pre=dis_pre_cells_pd.loc[dis_pre_cells_pd['cell_ID']==cell_ID,'file_begin'].item()
#     fe_pre=dis_pre_cells_pd.loc[dis_pre_cells_pd['cell_ID']==cell_ID,'file_end'].item()
#     dictFilterTrials_pre={'dir':'filterOff', 'trial_name':trial_type, 'fail':0, 'after_fail':'filterOff','saccade_motion':saccade_option,'blink_motion':saccade_option,'files_begin_end':[fb_pre,fe_pre]}    
#     PSTH=cur_cell_task.PSTH(window,dictFilterTrials_pre,plot_option=0,smooth_option=1) 
#     PSTH_disPre=PSTH[filter_margin:-filter_margin]
    
#     #Dishabituation after
#     fb_post=dis_next_cells_pd.loc[dis_next_cells_pd['cell_ID']==cell_ID,'file_begin'].item()
#     fe_post=dis_next_cells_pd.loc[dis_next_cells_pd['cell_ID']==cell_ID,'file_end'].item()
#     dictFilterTrials_post={'dir':'filterOff', 'trial_name':trial_type, 'fail':0, 'after_fail':'filterOff','saccade_motion':saccade_option,'blink_motion':saccade_option,'files_begin_end':[fb_post,fe_post]}    
#     PSTH=cur_cell_task.PSTH(window,dictFilterTrials_post,plot_option=0,smooth_option=1) 
#     PSTH_disNext=PSTH[filter_margin:-filter_margin]

#     plt.plot(time_course,PSTH_motor)
#     plt.plot(time_course,PSTH_disNext)
#     plt.plot(time_course,PSTH_disPre)
#     plt.title(cell_ID)
#     plt.legend(['motor','disNext','disPre'])
#     plt.axvline(x=100,color='black')
#     plt.axvline(x=300,color='black')
#     plt.show()
       