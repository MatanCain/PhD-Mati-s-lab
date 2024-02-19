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
import operator
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



#Find cells recorded during whole learning block, the dishabituation around it and 80 trials during mapping 
motor_block_cells=list(find_cell_entire_block(motor_tasks,cells_db=cells_db,behaviour_db=behaviour_db).cell_ID)
cong_block_cells=list(find_cell_entire_block(congruent_tasks,cells_db=cells_db,behaviour_db=behaviour_db).cell_ID)
incong_block_cells=list(find_cell_entire_block(incongruent_tasks,cells_db=cells_db,behaviour_db=behaviour_db).cell_ID)



dis_pre_cells_pd_motor=(find_cell_entire_dis_block(motor_tasks,cells_db=cells_db,behaviour_db=behaviour_db,rel_block='previous'))
dis_pre_cells_motor=list(dis_pre_cells_pd_motor.cell_ID)

dis_pre_cells_pd_cong=(find_cell_entire_dis_block(congruent_tasks,cells_db=cells_db,behaviour_db=behaviour_db,rel_block='previous'))
dis_pre_cells_cong=list(dis_pre_cells_pd_cong.cell_ID)

dis_pre_cells_pd_incong=(find_cell_entire_dis_block(incongruent_tasks,cells_db=cells_db,behaviour_db=behaviour_db,rel_block='previous'))
dis_pre_cells_incong=list(dis_pre_cells_pd_incong.cell_ID)


mapping_cells= get_mapping_cells(n_min_pursuit=80)  
complete_cells=intersection_set = set.intersection(set(motor_block_cells), set(cong_block_cells),set(incong_block_cells),set(dis_pre_cells_motor),set(dis_pre_cells_cong), set(dis_pre_cells_incong),set(mapping_cells))


mapping_tasks=['8dir_active_passive_interleaved_100_25','8dir_active_passive_interleaved']
#####################################

#####################################
##Build array of learned_cells and base_cells 
#learned cells are cells that prefere learned direction rather than base
#sig means that the difference betenn base and learned is significant
#This is done according to the FR of the cells in the mapping
active_base_cells=[]
active_base_cells_sig=[]
active_learned_cells=[]
active_learned_cells_sig=[]

passive_base_cells=[]
passive_base_cells_sig=[]
passive_learned_cells=[]
passive_learned_cells_sig=[]

active_stim_cells_sig=[]
passive_stim_cells_sig=[]
active_stim_cells=[]
passive_stim_cells=[]

#saccade_option=1 #cancel trials with saccades that begin between 0 and 300 after motion onset
saccade_option='filterOff' #doesnt cancel trials with saccades that begin between 0 and 300 after motion onset


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
        learned_dir=(base_dir+45)%360
        stim_dir=(base_dir+45)%360

    elif 'CW' in cur_behaviour_db.iloc[0]['Task']:   
        learned_dir=(base_dir-45)%360
        stim_dir=(base_dir-45)%360


    #Look for the FR in those directions in mapping in active and passive trials
    for cur_task in mapping_tasks:
        try:
            cur_cell_task=load_cell_task(cell_task_py_folder,cur_task,cell_ID)
        except:
            continue
    #Window we look at in the mapping:
    window_begin=0
    window_end=750
    #active base
    dictFilterTrials_base_active={'dir':(base_dir-SR_pursuit)%360, 'trial_name':'v20a', 'fail':0, 'after_fail':'filterOff','saccade_motion':saccade_option}    
    FR_active_base=cur_cell_task.get_mean_FR_event(dictFilterTrials_base_active,'motion_onset',window_pre=window_begin,window_post=window_end)
    FR_active_base_mean=np.mean(np.array(FR_active_base))
    #active learned
    dictFilterTrials_learned_active={'dir':(learned_dir-SR_pursuit)%360, 'trial_name':'v20a', 'fail':0, 'after_fail':'filterOff','saccade_motion':saccade_option}    
    FR_active_learned=cur_cell_task.get_mean_FR_event(dictFilterTrials_learned_active,'motion_onset',window_pre=window_begin,window_post=window_end)
    FR_active_learned_mean=np.mean(np.array(FR_active_learned))
    #passive base
    dictFilterTrials_base_passive={'dir':(base_dir-SR_pursuit)%360, 'trial_name':'v20p', 'fail':0, 'after_fail':'filterOff','saccade_motion':saccade_option}    
    FR_passive_base=cur_cell_task.get_mean_FR_event(dictFilterTrials_base_passive,'motion_onset',window_pre=window_begin,window_post=window_end)
    FR_passive_base_mean=np.mean(np.array(FR_passive_base))
    #passive learned
    dictFilterTrials_learned_passive={'dir':(learned_dir-SR_pursuit)%360, 'trial_name':'v20p', 'fail':0, 'after_fail':'filterOff','saccade_motion':saccade_option}    
    FR_passive_learned=cur_cell_task.get_mean_FR_event(dictFilterTrials_learned_passive,'motion_onset',window_pre=window_begin,window_post=window_end)
    FR_passive_learned_mean=np.mean(np.array(FR_passive_learned))
    
    #active stim (direction of the stimulus after directionnal change)
    dictFilterTrials_stim_active={'dir':(stim_dir-SR_pursuit)%360, 'trial_name':'v20a', 'fail':0, 'after_fail':'filterOff','saccade_motion':saccade_option}    
    FR_active_stim_MO=cur_cell_task.get_mean_FR_event(dictFilterTrials_stim_active,'motion_onset',window_pre=window_begin,window_post=window_end)
    FR_active_stim_MO_mean=np.mean(np.array(FR_active_stim_MO))    

    dictFilterTrials_stim_active={'dir':(stim_dir-SR_pursuit)%360, 'trial_name':'v20a', 'fail':0, 'after_fail':'filterOff','saccade_motion':saccade_option}    
    FR_active_stim_BL=cur_cell_task.get_mean_FR_event(dictFilterTrials_stim_active,'motion_onset',window_pre=-1000,window_post=-500)
    FR_active_stim_BL_mean=np.mean(np.array(FR_active_stim_BL))      

    #passive stim
    dictFilterTrials_stim_passive={'dir':(stim_dir-SR_pursuit)%360, 'trial_name':'v20p', 'fail':0, 'after_fail':'filterOff','saccade_motion':saccade_option}    
    FR_passive_stim_MO=cur_cell_task.get_mean_FR_event(dictFilterTrials_stim_passive,'motion_onset',window_pre=window_begin,window_post=window_end)
    FR_passive_stim_MO_mean=np.mean(np.array(FR_passive_stim_MO))    

    dictFilterTrials_stim_passive={'dir':(stim_dir-SR_pursuit)%360, 'trial_name':'v20p', 'fail':0, 'after_fail':'filterOff','saccade_motion':saccade_option}    
    FR_passive_stim_BL=cur_cell_task.get_mean_FR_event(dictFilterTrials_stim_passive,'motion_onset',window_pre=-1000,window_post=-500)
    FR_passive_stim_BL_mean=np.mean(np.array(FR_passive_stim_BL))  
    
    #check if the difference is significant
    try:
        stat,p_val_active=scipy.stats.mannwhitneyu(FR_active_base,FR_active_learned)
    except:
        p_val_active=1
        
    try:
        stat,p_val_passive=scipy.stats.mannwhitneyu(FR_passive_base,FR_passive_learned)
    except:
        p_val_passive=1

    try:
        stat,p_val_active_stim=scipy.stats.mannwhitneyu(FR_active_stim_MO,FR_active_stim_BL)
    except:
        p_val_active_stim=1
    
    try:
        stat,p_val_passive_stim,=scipy.stats.mannwhitneyu(FR_passive_stim_MO,FR_passive_stim_BL)
    except:
        p_val_passive_stim=1


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
 
    if FR_active_stim_MO_mean>FR_active_stim_BL_mean:
        active_stim_cells.append(cell_ID)       
        if p_val_active_stim<0.05:
            active_stim_cells_sig.append(cell_ID)       
    if FR_passive_stim_MO_mean>FR_passive_stim_BL_mean:
        passive_stim_cells.append(cell_ID)
        if p_val_passive_stim<0.05:
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
tasks_lists=[motor_tasks,congruent_tasks,incongruent_tasks]
washout_df_lists=[dis_pre_cells_pd_motor,dis_pre_cells_pd_cong,dis_pre_cells_pd_incong]

dishabituation_tasks=['Dishabituation_100_25_cue','Dishabituation']

for cell_ID in complete_cells:
    fb_dis_motor=dis_pre_cells_pd_motor.loc[dis_pre_cells_pd_motor['cell_ID']==cell_ID,'file_begin'].item()
    fb_dis_cong=dis_pre_cells_pd_cong.loc[dis_pre_cells_pd_cong['cell_ID']==cell_ID,'file_begin'].item()
    fb_dis_incong=dis_pre_cells_pd_incong.loc[dis_pre_cells_pd_incong['cell_ID']==cell_ID,'file_begin'].item()
    file_begin_list=[fb_dis_motor,fb_dis_cong,fb_dis_incong]
    
    enumerate_object = enumerate(file_begin_list)
    sorted_pairs = sorted(enumerate_object, key=operator.itemgetter(1))
    sorted_indices = [index for index, element in sorted_pairs]


    for cur_task in tasks_lists[sorted_indices[0]]:
        try:
            cur_cell_task=load_cell_task(cell_task_py_folder,cur_task,cell_ID)
        except:
            continue
    dictFilterTrials_block={'dir':'filterOff', 'trial_name':'filterOff', 'fail':'filterOff', 'after_fail':'filterOff','saccade_motion':'filterOff'}    
    FR_block_0=cur_cell_task.get_mean_FR_event(dictFilterTrials_block,'motion_onset',window_pre=-1000,window_post=-500)

    #dishabituation
    for cur_task in dishabituation_tasks:
        try:
            cur_cell_task=load_cell_task(cell_task_py_folder,cur_task,cell_ID)
        except:
            continue
    fb_pre=washout_df_lists[sorted_indices[0]].loc[washout_df_lists[sorted_indices[0]]['cell_ID']==cell_ID,'file_begin'].item()
    fe_pre=washout_df_lists[sorted_indices[0]].loc[washout_df_lists[sorted_indices[0]]['cell_ID']==cell_ID,'file_end'].item()
    dictFilterTrials_pre={'dir':'filterOff', 'trial_name':'filterOff', 'fail':'filterOff', 'after_fail':'filterOff','saccade_motion':'filterOff','files_begin_end':[fb_pre,fe_pre]}    
    FR_dis_pre_0=cur_cell_task.get_mean_FR_event(dictFilterTrials_pre,'motion_onset',window_pre=-1000,window_post=-500)

    fb_pre=washout_df_lists[sorted_indices[1]].loc[washout_df_lists[sorted_indices[1]]['cell_ID']==cell_ID,'file_begin'].item()
    fe_pre=washout_df_lists[sorted_indices[1]].loc[washout_df_lists[sorted_indices[1]]['cell_ID']==cell_ID,'file_end'].item()
    dictFilterTrials_pre={'dir':'filterOff', 'trial_name':'filterOff', 'fail':'filterOff', 'after_fail':'filterOff','saccade_motion':'filterOff','files_begin_end':[fb_pre,fe_pre]}    
    FR_dis_pre_1=cur_cell_task.get_mean_FR_event(dictFilterTrials_pre,'motion_onset',window_pre=-1000,window_post=-500)
    
    fb_pre=washout_df_lists[sorted_indices[2]].loc[washout_df_lists[sorted_indices[2]]['cell_ID']==cell_ID,'file_begin'].item()
    fe_pre=washout_df_lists[sorted_indices[2]].loc[washout_df_lists[sorted_indices[2]]['cell_ID']==cell_ID,'file_end'].item()
    dictFilterTrials_pre={'dir':'filterOff', 'trial_name':'filterOff', 'fail':'filterOff', 'after_fail':'filterOff','saccade_motion':'filterOff','files_begin_end':[fb_pre,fe_pre]}    
    FR_dis_pre_2=cur_cell_task.get_mean_FR_event(dictFilterTrials_pre,'motion_onset',window_pre=-1000,window_post=-500)
    

    

    for cur_task in tasks_lists[sorted_indices[1]]:
        try:
            cur_cell_task=load_cell_task(cell_task_py_folder,cur_task,cell_ID)
        except:
            continue
    dictFilterTrials_block={'dir':'filterOff', 'trial_name':'filterOff', 'fail':'filterOff', 'after_fail':'filterOff','saccade_motion':'filterOff'}    
    FR_block_1=cur_cell_task.get_mean_FR_event(dictFilterTrials_block,'motion_onset',window_pre=-1000,window_post=-500)
    

    for cur_task in tasks_lists[sorted_indices[2]]:
        try:
            cur_cell_task=load_cell_task(cell_task_py_folder,cur_task,cell_ID)
        except:
            continue
    dictFilterTrials_block={'dir':'filterOff', 'trial_name':'filterOff', 'fail':'filterOff', 'after_fail':'filterOff','saccade_motion':'filterOff'}    
    FR_block_2=cur_cell_task.get_mean_FR_event(dictFilterTrials_block,'motion_onset',window_pre=-1000,window_post=-500)
        
        

    #concatenation and correlation
    FR_conc=np.concatenate((FR_dis_pre_0.to_numpy(),FR_block_0.to_numpy(),FR_dis_pre_1.to_numpy(),FR_block_1.to_numpy(),FR_dis_pre_2.to_numpy(),FR_block_2.to_numpy()),axis=0)
    corr,pval=stats.pearsonr(FR_conc, np.arange(1,FR_conc.size+1))

    plt.scatter( np.arange(1,FR_block_0.size+1),FR_block_0)
    plt.scatter( np.arange(FR_block_0.size+1,FR_block_0.size+FR_block_1.size+1),FR_block_1)
    plt.scatter( np.arange(FR_block_0.size+1+FR_block_1.size,FR_block_0.size+FR_block_1.size+FR_block_2.size+1),FR_block_2)
    plt.ylabel('FR')
    plt.xlabel('trial index')
    plt.legend(['block 1','block 2','block 3'])
    plt.title('cell_ID: '+ str(cell_ID) + ' corr: ' + str(round(corr,3))+' pval: ' + str(round(pval,3)))
    plt.show()

    if pval<0.05:

        unstable_cells.append(cell_ID)
#########################################   





#%%
#################################################################################    
#Look at different task together
       
cur_cell_list=active_base_cells
cur_cell_list=[x for x in cur_cell_list if x not in unstable_cells]


filter_margin=100 #for Psth margin
window={"timePoint": 'motion_onset',"timeBefore":-900,"timeAfter":950}
time_course=np.arange(window['timeBefore']+filter_margin,window['timeAfter']-filter_margin)

#CHOOSE A TRIAL TYPE
#v20S is unavailable for motor block
trial_type='v20NS'
#saccade_option=1 #cancel trials with saccades that begin between 0 and 300 after motion onset
saccade_option=1 #doesnt cancel trials with saccades that begin between 0 and 300 after motion onset

#initialize psth arrays
PSTH_cong_array=np.zeros((len(cur_cell_list),np.size(time_course)))
PSTH_incong_array=np.zeros((len(cur_cell_list),np.size(time_course)))
PSTH_motor_array=np.zeros((len(cur_cell_list),np.size(time_course)))
PSTH_disPre_array_motor=np.zeros((len(cur_cell_list),np.size(time_course)))
PSTH_disPre_array_cong=np.zeros((len(cur_cell_list),np.size(time_course)))
PSTH_disPre_array_incong=np.zeros((len(cur_cell_list),np.size(time_course)))


for cell_inx,cell_ID in enumerate(cur_cell_list):    
    #Learning block
    for cur_task in congruent_tasks:
        try:
            cur_cell_task=load_cell_task(cell_task_py_folder,cur_task,cell_ID)
        except:
            continue
    dictFilterTrials_block={'dir':'filterOff', 'trial_name':trial_type, 'fail':0, 'after_fail':'filterOff','saccade_motion':saccade_option,'blink_motion':saccade_option}    
    PSTH=cur_cell_task.PSTH(window,dictFilterTrials_block,plot_option=0,smooth_option=1) 
    PSTH=PSTH[filter_margin:-filter_margin]
    PSTH_cong_array[cell_inx,:]=PSTH

    for cur_task in incongruent_tasks:
        try:
            cur_cell_task=load_cell_task(cell_task_py_folder,cur_task,cell_ID)
        except:
            continue
    dictFilterTrials_block={'dir':'filterOff', 'trial_name':trial_type, 'fail':0, 'after_fail':'filterOff','saccade_motion':saccade_option,'blink_motion':saccade_option}    
    PSTH=cur_cell_task.PSTH(window,dictFilterTrials_block,plot_option=0,smooth_option=1) 
    PSTH=PSTH[filter_margin:-filter_margin]
    PSTH_incong_array[cell_inx,:]=PSTH
    
    if trial_type=='v20NS':
        for cur_task in motor_tasks:
            try:
                cur_cell_task=load_cell_task(cell_task_py_folder,cur_task,cell_ID)
            except:
                continue
        dictFilterTrials_block={'dir':'filterOff', 'trial_name':trial_type, 'fail':0, 'after_fail':'filterOff','saccade_motion':saccade_option,'blink_motion':saccade_option}    
        trials_df_motor=cur_cell_task.filtTrials(dictFilterTrials_block) 
        PSTH=cur_cell_task.PSTH(window,dictFilterTrials_block,plot_option=0,smooth_option=1) 
        PSTH=PSTH[filter_margin:-filter_margin]
        PSTH_motor_array[cell_inx,:]=PSTH

    #Dishabituation
    dishabituation_tasks=['Dishabituation_100_25_cue','Dishabituation']
    for cur_task in dishabituation_tasks:
        try:
            cur_cell_task=load_cell_task(cell_task_py_folder,cur_task,cell_ID)
        except:
            continue
    #Dishabituation before motor
    fb_pre=dis_pre_cells_pd_motor.loc[dis_pre_cells_pd_motor['cell_ID']==cell_ID,'file_begin'].item()
    fe_pre=dis_pre_cells_pd_motor.loc[dis_pre_cells_pd_motor['cell_ID']==cell_ID,'file_end'].item()
    dictFilterTrials_pre={'dir':'filterOff', 'trial_name':trial_type, 'fail':0, 'after_fail':'filterOff','saccade_motion':saccade_option,'blink_motion':saccade_option,'files_begin_end':[fb_pre,fe_pre]}    
    PSTH=cur_cell_task.PSTH(window,dictFilterTrials_pre,plot_option=0,smooth_option=1) 
    PSTH=PSTH[filter_margin:-filter_margin]
    PSTH_disPre_array_motor[cell_inx,:]=PSTH   

    #Dishabituation before congruent
    fb_pre=dis_pre_cells_pd_cong.loc[dis_pre_cells_pd_cong['cell_ID']==cell_ID,'file_begin'].item()
    fe_pre=dis_pre_cells_pd_cong.loc[dis_pre_cells_pd_cong['cell_ID']==cell_ID,'file_end'].item()
    dictFilterTrials_pre={'dir':'filterOff', 'trial_name':trial_type, 'fail':0, 'after_fail':'filterOff','saccade_motion':saccade_option,'blink_motion':saccade_option,'files_begin_end':[fb_pre,fe_pre]}    
    PSTH=cur_cell_task.PSTH(window,dictFilterTrials_pre,plot_option=0,smooth_option=1) 
    PSTH=PSTH[filter_margin:-filter_margin]
    PSTH_disPre_array_cong[cell_inx,:]=PSTH   


    #Dishabituation before incongruent
    fb_pre=dis_pre_cells_pd_incong.loc[dis_pre_cells_pd_incong['cell_ID']==cell_ID,'file_begin'].item()
    fe_pre=dis_pre_cells_pd_incong.loc[dis_pre_cells_pd_incong['cell_ID']==cell_ID,'file_end'].item()
    dictFilterTrials_pre={'dir':'filterOff', 'trial_name':trial_type, 'fail':0, 'after_fail':'filterOff','saccade_motion':saccade_option,'blink_motion':saccade_option,'files_begin_end':[fb_pre,fe_pre]}    
    PSTH=cur_cell_task.PSTH(window,dictFilterTrials_pre,plot_option=0,smooth_option=1) 
    PSTH=PSTH[filter_margin:-filter_margin]
    PSTH_disPre_array_incong[cell_inx,:]=PSTH   

PSTH_motor=np.nanmean(PSTH_motor_array,axis=0)
PSTH_congruent=np.nanmean(PSTH_cong_array,axis=0)
PSTH_incongruent=np.nanmean(PSTH_incong_array,axis=0)  
  
PSTH_dis_motor=np.nanmean(PSTH_disPre_array_motor,axis=0)
PSTH_dis_cong=np.nanmean(PSTH_disPre_array_cong,axis=0)
PSTH_dis_incong=np.nanmean(PSTH_disPre_array_incong,axis=0)

plt.plot(time_course,PSTH_dis_motor,color='lightgray')
plt.plot(time_course,PSTH_dis_cong,color='lightgray')
plt.plot(time_course,PSTH_dis_incong,color='lightgray')
if trial_type=='v20NS':
    plt.plot(time_course,PSTH_motor,color='tab:blue')
plt.plot(time_course,PSTH_congruent,color='tab:orange')
plt.plot(time_course,PSTH_incongruent,color='tab:green')




plt.axvline(x=0,color='black')
plt.axvline(x=300,color='black')
plt.xlabel('time from MO')
plt.ylabel('FR')
plt.title(str(len(cur_cell_list))+' cells')
if trial_type=='v20NS':
    plt.legend(['dis motor','dis cong','dis incong','motor','cong','incong'])
else:
    plt.legend(['dis motor','dis cong','dis incong','cong','incong'])
#plt.ylim((3,35))
plt.show()    



#################################################################################    
#%%
#Look at different task together
       
cur_cell_list=active_base_cells
cur_cell_list=[x for x in cur_cell_list if x not in unstable_cells]


filter_margin=100 #for Psth margin
window={"timePoint": 'motion_onset',"timeBefore":-900,"timeAfter":950}
time_course=np.arange(window['timeBefore']+filter_margin,window['timeAfter']-filter_margin)

#CHOOSE A TRIAL TYPE
#v20S is unavailable for motor block
trial_type='v20NS'
#saccade_option=1 #cancel trials with saccades that begin between 0 and 300 after motion onset
saccade_option=1 #doesnt cancel trials with saccades that begin between 0 and 300 after motion onset


#initialize psth arrays
PSTH_cong_array=np.empty((len(cur_cell_list),np.size(time_course)))
PSTH_incong_array=np.empty((len(cur_cell_list),np.size(time_course)))
PSTH_motor_array=np.empty((len(cur_cell_list),np.size(time_course)))

PSTH_disPre_array_motor=np.empty((len(cur_cell_list),np.size(time_course)))
PSTH_disPre_array_cong=np.empty((len(cur_cell_list),np.size(time_course)))
PSTH_disPre_array_incong=np.empty((len(cur_cell_list),np.size(time_course)))

PSTH_cong_array[:]=np.NaN
PSTH_incong_array[:]=np.NaN
PSTH_motor_array[:]=np.NaN
PSTH_disPre_array_motor[:]=np.NaN
PSTH_disPre_array_cong[:]=np.NaN
PSTH_disPre_array_incong[:]=np.NaN


tasks_lists=[motor_tasks,congruent_tasks,incongruent_tasks,mapping_tasks]
washout_df_lists=[dis_pre_cells_pd_motor,dis_pre_cells_pd_cong,dis_pre_cells_pd_incong]

n_cells_cong=0
n_cells_incong=0
n_cells_motor=0
for cell_inx,cell_ID in enumerate(cur_cell_list): 
    
    fb_dis_motor=dis_pre_cells_pd_motor.loc[dis_pre_cells_pd_motor['cell_ID']==cell_ID,'file_begin'].item()
    fb_dis_cong=dis_pre_cells_pd_cong.loc[dis_pre_cells_pd_cong['cell_ID']==cell_ID,'file_begin'].item()
    fb_dis_incong=dis_pre_cells_pd_incong.loc[dis_pre_cells_pd_incong['cell_ID']==cell_ID,'file_begin'].item()
    file_begin_list=[fb_dis_motor,fb_dis_cong,fb_dis_incong]
    
    enumerate_object = enumerate(file_begin_list)
    sorted_pairs = sorted(enumerate_object, key=operator.itemgetter(1))
    sorted_indices = [index for index, element in sorted_pairs]

    #index of motor task:
    inx_motor=sorted_indices.index(0)
    inx_cong=sorted_indices.index(1)
    inx_incong=sorted_indices.index(2)

    
    if inx_motor>inx_incong:
        continue
    else:
        n_cells_motor=n_cells_motor+1            
        
        n_cells_cong=n_cells_cong+1
        for cur_task in congruent_tasks:
            try:
                cur_cell_task=load_cell_task(cell_task_py_folder,cur_task,cell_ID)
            except:
                continue
        dictFilterTrials_block={'dir':'filterOff', 'trial_name':trial_type, 'fail':0, 'after_fail':'filterOff','saccade_motion':saccade_option,'blink_motion':saccade_option}    
        PSTH=cur_cell_task.PSTH(window,dictFilterTrials_block,plot_option=0,smooth_option=1) 
        PSTH=PSTH[filter_margin:-filter_margin]
        PSTH_cong_array[cell_inx,:]=PSTH
        
        n_cells_incong=n_cells_incong+1
        for cur_task in incongruent_tasks:
            try:
                cur_cell_task=load_cell_task(cell_task_py_folder,cur_task,cell_ID)
            except:
                continue
        dictFilterTrials_block={'dir':'filterOff', 'trial_name':trial_type, 'fail':0, 'after_fail':'filterOff','saccade_motion':saccade_option,'blink_motion':saccade_option}    
        PSTH=cur_cell_task.PSTH(window,dictFilterTrials_block,plot_option=0,smooth_option=1) 
        PSTH=PSTH[filter_margin:-filter_margin]
        PSTH_incong_array[cell_inx,:]=PSTH
    
    
        for cur_task in motor_tasks:
            try:
                cur_cell_task=load_cell_task(cell_task_py_folder,cur_task,cell_ID)
            except:
                continue
        dictFilterTrials_block={'dir':'filterOff', 'trial_name':trial_type, 'fail':0, 'after_fail':'filterOff','saccade_motion':saccade_option,'blink_motion':saccade_option}    
        trials_df_motor=cur_cell_task.filtTrials(dictFilterTrials_block) 
        PSTH=cur_cell_task.PSTH(window,dictFilterTrials_block,plot_option=0,smooth_option=1) 
        PSTH=PSTH[filter_margin:-filter_margin]
        PSTH_motor_array[cell_inx,:]=PSTH
    
        #### find base and learned direcitons
        #cell db :find session, File begin and file end of current cell
        fb=cells_db.loc[cells_db['cell_ID']==cell_ID,'fb_after_stablility'].item()
        fe=cells_db.loc[cells_db['cell_ID']==cell_ID,'fe_after_stability'].item()
        session=cells_db.loc[cells_db['cell_ID']==cell_ID,'session'].item()
           
        #behaviour_db-find base and learned direction
        cur_behaviour_db=behaviour_db.loc[behaviour_db['behaviour_session']==session ,: ] #keep only relevant day
        inxs=cur_behaviour_db.loc[cur_behaviour_db['file_begin']>fb ,: ].index #keep only relevant session
        cur_behaviour_db=cur_behaviour_db.loc[([inxs.to_list()[0]-1])+inxs.to_list() ,: ] #keep only relevant session
        cur_behaviour_db=cur_behaviour_db.loc[cur_behaviour_db['file_end']<=fe ,: ]

        #Dishabituation
        dishabituation_tasks=['Dishabituation_100_25_cue','Dishabituation']
        for cur_task in dishabituation_tasks:
            try:
                cur_cell_task=load_cell_task(cell_task_py_folder,cur_task,cell_ID)
            except:
                continue
        #Dishabituation before motor
        fb_pre=dis_pre_cells_pd_motor.loc[dis_pre_cells_pd_motor['cell_ID']==cell_ID,'file_begin'].item()
        fe_pre=dis_pre_cells_pd_motor.loc[dis_pre_cells_pd_motor['cell_ID']==cell_ID,'file_end'].item()
        dictFilterTrials_pre={'dir':'filterOff', 'trial_name':trial_type, 'fail':0, 'after_fail':'filterOff','saccade_motion':saccade_option,'blink_motion':saccade_option,'files_begin_end':[fb_pre,fe_pre]}    
        PSTH=cur_cell_task.PSTH(window,dictFilterTrials_pre,plot_option=0,smooth_option=1) 
        PSTH=PSTH[filter_margin:-filter_margin]
        PSTH_disPre_array_motor[cell_inx,:]=PSTH
        
        #Dishabituation before congruent
        fb_pre=dis_pre_cells_pd_cong.loc[dis_pre_cells_pd_cong['cell_ID']==cell_ID,'file_begin'].item()
        fe_pre=dis_pre_cells_pd_cong.loc[dis_pre_cells_pd_cong['cell_ID']==cell_ID,'file_end'].item()
        dictFilterTrials_pre={'dir':'filterOff', 'trial_name':trial_type, 'fail':0, 'after_fail':'filterOff','saccade_motion':saccade_option,'blink_motion':saccade_option,'files_begin_end':[fb_pre,fe_pre]}    
        PSTH=cur_cell_task.PSTH(window,dictFilterTrials_pre,plot_option=0,smooth_option=1) 
        PSTH=PSTH[filter_margin:-filter_margin]
        PSTH_disPre_array_cong[cell_inx,:]=PSTH   
    
        #Dishabituation before incongruent
        fb_pre=dis_pre_cells_pd_incong.loc[dis_pre_cells_pd_incong['cell_ID']==cell_ID,'file_begin'].item()
        fe_pre=dis_pre_cells_pd_incong.loc[dis_pre_cells_pd_incong['cell_ID']==cell_ID,'file_end'].item()
        dictFilterTrials_pre={'dir':'filterOff', 'trial_name':trial_type, 'fail':0, 'after_fail':'filterOff','saccade_motion':saccade_option,'blink_motion':saccade_option,'files_begin_end':[fb_pre,fe_pre]}    
        PSTH=cur_cell_task.PSTH(window,dictFilterTrials_pre,plot_option=0,smooth_option=1) 
        PSTH=PSTH[filter_margin:-filter_margin]
        PSTH_disPre_array_incong[cell_inx,:]=PSTH   

PSTH_motor=np.nanmean(PSTH_motor_array,axis=0)
PSTH_congruent=np.nanmean(PSTH_cong_array,axis=0)
PSTH_incongruent=np.nanmean(PSTH_incong_array,axis=0)  
  
PSTH_dis_motor=np.nanmean(PSTH_disPre_array_motor,axis=0)
PSTH_dis_cong=np.nanmean(PSTH_disPre_array_cong,axis=0)
PSTH_dis_incong=np.nanmean(PSTH_disPre_array_incong,axis=0)

plt.plot(time_course,PSTH_dis_motor,color='lightgray')
plt.plot(time_course,PSTH_dis_cong,color='lightgray')
plt.plot(time_course,PSTH_dis_incong,color='lightgray')
if trial_type=='v20NS':
    plt.plot(time_course,PSTH_motor,color='tab:blue')
plt.plot(time_course,PSTH_congruent,color='tab:orange')
plt.plot(time_course,PSTH_incongruent,color='tab:green')




plt.axvline(x=0,color='black')
plt.axvline(x=300,color='black')
plt.xlabel('time from MO')
plt.ylabel('FR')
plt.title(str(len(cur_cell_list))+' cells')
if trial_type=='v20NS':
    plt.legend(['dis motor','dis cong','dis incong','motor','cong','incong'])
else:
    plt.legend(['dis motor','dis cong','dis incong','cong','incong'])
#plt.ylim((5,28))
plt.show()    


#%%
#################################################################################    
#Look at different task together
       
cur_cell_list=complete_cells
cur_cell_list=[x for x in cur_cell_list if x not in unstable_cells]


filter_margin=100 #for Psth margin
window={"timePoint": 'motion_onset',"timeBefore":-900,"timeAfter":950}
time_course=np.arange(window['timeBefore']+filter_margin,window['timeAfter']-filter_margin)

#CHOOSE A TRIAL TYPE
#v20S is unavailable for motor block
trial_type='v20NS'
#saccade_option=1 #cancel trials with saccades that begin between 0 and 300 after motion onset
saccade_option=1 #doesnt cancel trials with saccades that begin between 0 and 300 after motion onset

#initialize psth arrays
PSTH_cong_array=np.zeros((len(cur_cell_list),np.size(time_course)))
PSTH_incong_array=np.zeros((len(cur_cell_list),np.size(time_course)))
PSTH_motor_array=np.zeros((len(cur_cell_list),np.size(time_course)))
PSTH_disPre_array_motor=np.zeros((len(cur_cell_list),np.size(time_course)))
PSTH_disPre_array_cong=np.zeros((len(cur_cell_list),np.size(time_course)))
PSTH_disPre_array_incong=np.zeros((len(cur_cell_list),np.size(time_course)))

scatter_motor=[]
scatter_cong=[]
scatter_incong=[]

for cell_inx,cell_ID in enumerate(cur_cell_list):    
    #Learning block
    for cur_task in congruent_tasks:
        try:
            cur_cell_task=load_cell_task(cell_task_py_folder,cur_task,cell_ID)
        except:
            continue
    dictFilterTrials_block={'dir':'filterOff', 'trial_name':trial_type, 'fail':0, 'after_fail':'filterOff','saccade_motion':saccade_option,'blink_motion':saccade_option}    
    PSTH=cur_cell_task.PSTH(window,dictFilterTrials_block,plot_option=0,smooth_option=1) 
    PSTH=PSTH[filter_margin:-filter_margin]
    PSTH_cong_array[cell_inx,:]=PSTH
    #scatter cong
    scatter_cong.append(np.nanmean(PSTH[-window['timeBefore']-filter_margin+320:-window['timeBefore']-filter_margin+750]))

    for cur_task in incongruent_tasks:
        try:
            cur_cell_task=load_cell_task(cell_task_py_folder,cur_task,cell_ID)
        except:
            continue
    dictFilterTrials_block={'dir':'filterOff', 'trial_name':trial_type, 'fail':0, 'after_fail':'filterOff','saccade_motion':saccade_option,'blink_motion':saccade_option}    
    PSTH=cur_cell_task.PSTH(window,dictFilterTrials_block,plot_option=0,smooth_option=1) 
    PSTH=PSTH[filter_margin:-filter_margin]
    PSTH_incong_array[cell_inx,:]=PSTH
 
    #scatter incong
    scatter_incong.append(np.nanmean(PSTH[-window['timeBefore']-filter_margin+320:-window['timeBefore']-filter_margin+750]))
    
    if trial_type=='v20NS':
        for cur_task in motor_tasks:
            try:
                cur_cell_task=load_cell_task(cell_task_py_folder,cur_task,cell_ID)
            except:
                continue
        dictFilterTrials_block={'dir':'filterOff', 'trial_name':trial_type, 'fail':0, 'after_fail':'filterOff','saccade_motion':saccade_option,'blink_motion':saccade_option}    
        trials_df_motor=cur_cell_task.filtTrials(dictFilterTrials_block) 
        PSTH=cur_cell_task.PSTH(window,dictFilterTrials_block,plot_option=0,smooth_option=1) 
        PSTH=PSTH[filter_margin:-filter_margin]
        PSTH_motor_array[cell_inx,:]=PSTH
        
        #scatter motor
        scatter_motor.append(np.nanmean(PSTH[-window['timeBefore']-filter_margin+320:-window['timeBefore']-filter_margin+750]))


    #Dishabituation
    dishabituation_tasks=['Dishabituation_100_25_cue','Dishabituation']
    for cur_task in dishabituation_tasks:
        try:
            cur_cell_task=load_cell_task(cell_task_py_folder,cur_task,cell_ID)
        except:
            continue
    #Dishabituation before motor
    fb_pre=dis_pre_cells_pd_motor.loc[dis_pre_cells_pd_motor['cell_ID']==cell_ID,'file_begin'].item()
    fe_pre=dis_pre_cells_pd_motor.loc[dis_pre_cells_pd_motor['cell_ID']==cell_ID,'file_end'].item()
    dictFilterTrials_pre={'dir':'filterOff', 'trial_name':trial_type, 'fail':0, 'after_fail':'filterOff','saccade_motion':saccade_option,'blink_motion':saccade_option,'files_begin_end':[fb_pre,fe_pre]}    
    PSTH=cur_cell_task.PSTH(window,dictFilterTrials_pre,plot_option=0,smooth_option=1) 
    PSTH=PSTH[filter_margin:-filter_margin]
    PSTH_disPre_array_motor[cell_inx,:]=PSTH   

    #Dishabituation before congruent
    fb_pre=dis_pre_cells_pd_cong.loc[dis_pre_cells_pd_cong['cell_ID']==cell_ID,'file_begin'].item()
    fe_pre=dis_pre_cells_pd_cong.loc[dis_pre_cells_pd_cong['cell_ID']==cell_ID,'file_end'].item()
    dictFilterTrials_pre={'dir':'filterOff', 'trial_name':trial_type, 'fail':0, 'after_fail':'filterOff','saccade_motion':saccade_option,'blink_motion':saccade_option,'files_begin_end':[fb_pre,fe_pre]}    
    PSTH=cur_cell_task.PSTH(window,dictFilterTrials_pre,plot_option=0,smooth_option=1) 
    PSTH=PSTH[filter_margin:-filter_margin]
    PSTH_disPre_array_cong[cell_inx,:]=PSTH   


    #Dishabituation before incongruent
    fb_pre=dis_pre_cells_pd_incong.loc[dis_pre_cells_pd_incong['cell_ID']==cell_ID,'file_begin'].item()
    fe_pre=dis_pre_cells_pd_incong.loc[dis_pre_cells_pd_incong['cell_ID']==cell_ID,'file_end'].item()
    dictFilterTrials_pre={'dir':'filterOff', 'trial_name':trial_type, 'fail':0, 'after_fail':'filterOff','saccade_motion':saccade_option,'blink_motion':saccade_option,'files_begin_end':[fb_pre,fe_pre]}    
    PSTH=cur_cell_task.PSTH(window,dictFilterTrials_pre,plot_option=0,smooth_option=1) 
    PSTH=PSTH[filter_margin:-filter_margin]
    PSTH_disPre_array_incong[cell_inx,:]=PSTH   

PSTH_motor=np.nanmean(PSTH_motor_array,axis=0)
PSTH_congruent=np.nanmean(PSTH_cong_array,axis=0)
PSTH_incongruent=np.nanmean(PSTH_incong_array,axis=0)  
  
PSTH_dis_motor=np.nanmean(PSTH_disPre_array_motor,axis=0)
PSTH_dis_cong=np.nanmean(PSTH_disPre_array_cong,axis=0)
PSTH_dis_incong=np.nanmean(PSTH_disPre_array_incong,axis=0)

plt.plot(time_course,PSTH_dis_motor,color='lightgray')
plt.plot(time_course,PSTH_dis_cong,color='lightgray')
plt.plot(time_course,PSTH_dis_incong,color='lightgray')
if trial_type=='v20NS':
    plt.plot(time_course,PSTH_motor,color='tab:blue')
plt.plot(time_course,PSTH_congruent,color='tab:orange')
plt.plot(time_course,PSTH_incongruent,color='tab:green')




plt.axvline(x=0,color='black')
plt.axvline(x=300,color='black')
plt.xlabel('time from MO')
plt.ylabel('FR')
plt.title(str(len(cur_cell_list))+' cells')
if trial_type=='v20NS':
    plt.legend(['dis motor','dis cong','dis incong','motor','cong','incong'])
else:
    plt.legend(['dis motor','dis cong','dis incong','cong','incong'])
#plt.ylim((3,35))
plt.show()    


#extract base cells from complete cells:
    #index of base cells
base_inx=[ind for ind, x in enumerate(cur_cell_list)  if x in active_base_cells]    
learned_inx=[ind for ind, x in enumerate(cur_cell_list)  if x in active_learned_cells]

scatter_motor_base=[x for ind,x in enumerate(scatter_motor) if ind in base_inx]
scatter_motor_learned=[x for ind,x in enumerate(scatter_motor) if ind in learned_inx]

scatter_cong_base=[x for ind,x in enumerate(scatter_cong) if ind in base_inx]
scatter_cong_learned=[x for ind,x in enumerate(scatter_cong) if ind in learned_inx]

scatter_incong_base=[x for ind,x in enumerate(scatter_incong) if ind in base_inx]
scatter_incong_learned=[x for ind,x in enumerate(scatter_incong) if ind in learned_inx]

stat, pval=scipy.stats.wilcoxon(scatter_motor,scatter_cong)
x=np.arange(0,100,1)
plt.scatter(scatter_motor_base,scatter_cong_base,color='tab:blue')
plt.scatter(scatter_motor_learned,scatter_cong_learned,color='tab:orange')
plt.title('motor vs cong'+' pval:'+str(round(pval,3)))
plt.xlabel('motor')
plt.ylabel('cong')
plt.legend(['base','learned'])
plt.plot(x,x,color='red')
plt.xlim((0,40))
plt.ylim((0,40))
plt.show()

stat, pval=scipy.stats.wilcoxon(scatter_motor,scatter_incong)
x=np.arange(0,100,1)
plt.scatter(scatter_motor_base,scatter_incong_base,color='tab:blue')
plt.scatter(scatter_motor_learned,scatter_incong_learned,color='tab:orange')
plt.title('motor vs incong'+' pval:'+str(round(pval,3)))
plt.xlabel('motor')
plt.ylabel('incong')
plt.legend(['base','learned'])
plt.plot(x,x,color='red')
plt.xlim((0,40))
plt.ylim((0,40))
plt.show()
