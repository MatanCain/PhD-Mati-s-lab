# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 11:50:49 2021

@author: Owner
"""
from __future__ import print_function

################################
#For all this program:
cell_task_py_folder="C:/Users/Owner/Google Drive/Documents/Studies/Mati's Lab/FEF learning project/Data_Analyses/units_task_python/"
###############################

from glob import glob
import pickle
import os
path="C:/Users/Owner/Google Drive/Documents/Studies/Mati's Lab/FEF learning project/Data_Analyses/code analyzes"
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
from scipy.ndimage import gaussian_filter1d
from mat4py import loadmat
import sys
from neuron_class import cell_task,filtCells,load_cell_task,smooth_data
import time
import warnings
from scipy.signal import savgol_filter
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
########################################################
cells_db_excel="C:/Users/Owner/Google Drive/Documents/Studies/Mati's Lab/FEF learning project/Data_Analyses/fiona_cells_db.xlsx"
cells_db=pd.read_excel(cells_db_excel)

behaviour_db_excel="C:/Users/Owner/Google Drive/Documents/Studies/Mati's Lab/FEF learning project/Data_Analyses/fiona_behaviour_db.xlsx"
behaviour_db=pd.read_excel(behaviour_db_excel)
#####################################

#This function receives a list of task and returns thea data frame for cells that were recorded during the whole block of a task
#The df columns: cell_ID, task, session(behaviour), file_begin(of the block),file_end
#Each row is one block. If a cell was recorded two entire block it will appear twice in the df
#ex: tasks=['Motor_learning_CCW_100_25_cue','Motor_learning_CW_100_25_cue']  

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
    
#######################################################
#This function receives:
    # trial_df - the data frame (cur_cell_task.trials_df)
    # event (str) - the event we are interested in. needs to be a column of trial_df ('cue_onset','dir_change'...)
    # window_pre and window_post (int) - How much time around the event where '-' is necessary if window pre is before the event 

# The functions caculates the FR around the event a return it as a serie
def mean_FR_event(trial_df,event,window_pre=-50,window_post=50):    
    
    #convert spike_time serie to df and adds nan to create a rectangular df
    cur_list=trial_df.loc[:,"spike_times"].tolist()

    cur_list=[np.append(spike_array,float('nan')) for spike_array in cur_list  ]
    spike_times_df=pd.DataFrame(cur_list) 
        
    #extract event serie from trial_df:
    event_serie=trial_df.loc[:,event].to_numpy()
    
    #define windows around event
    window_len=window_post-window_pre
    
    #Create a boolean array where true means that spike occured within the window
    spike_event_before=spike_times_df.gt(event_serie+window_pre,axis=0)  
    spike_event_before=spike_event_before.loc[:len(event_serie)-1,:] #it removes weird rows that are added in the end
    spike_event_before=spike_event_before.to_numpy()
    
    spike_event_after=spike_times_df.lt(event_serie+window_post,axis=0) 
    spike_event_after=spike_event_after.loc[:len(event_serie)-1,:] #it removes weird rows that are added in the end
    spike_event_after=spike_event_after.to_numpy()
    
    spike_event=np.logical_and(spike_event_before,spike_event_after)
        
    #SCalculates number of true events (spikes in the window) trial wise
    n_spikes=np.sum(spike_event,axis=1)
    #convert to FR (Hz)
    FR_event=n_spikes*(1000/window_len) 
    
    FR_event=pd.Series(FR_event)
    return FR_event
##########################################################



############################################################
#This function returns a trial_df and returns the base and learned direction
def find_base_and_learned_dir(trial_df):    
    task=trial_df.loc[0,'task']
    base_dir=trial_df.loc[0,'screen_rotation']
    if 'CCW' in task:
        learned_dir=(base_dir+90)%360
    elif 'CW' in task:
        learned_dir=(base_dir-90)%360
    else:
        print('Learned direction NOT FOUND!')
    return (base_dir,learned_dir)    
  
#############################################################


#########################################################################
#This funtion retunrs a list of cell which fire more in the motor learning block than in the previous dishabituation(100-320 ms after motion onset)
def motor_learning_cell():    
    learning_tasks=['Motor_learning_CCW_100_25_cue','Motor_learning_CW_100_25_cue','Motor_learning_CW','Motor_learning_CCW']
    
    # find cell_list: a list of cells recorded during a whole motor block and the previous/next dishabituation
    df_cells_motor=find_cell_entire_block(learning_tasks,cells_db=cells_db,behaviour_db=behaviour_db)
    df_cells_dis=find_cell_entire_dis_block(learning_tasks,cells_db=cells_db,behaviour_db=behaviour_db,rel_block='next')
    cell_motor_list=df_cells_motor.loc[:,'cell_ID'].to_list()
    cell_dis_list=df_cells_dis.loc[:,'cell_ID'].to_list()
    cell_list=[cell_ID for cell_ID in cell_motor_list if cell_ID in cell_dis_list]
    
    #Step 2- Check FR of cells in cell_list around time of directionnal change and check if significant  
    dictFilterTrials_learning = {'dir':'filterOff', 'trial_name':'filterOff', 'fail':0, 'after_fail':'filterOff','saccade_motion':'filterOff','trial_begin_end':'filterOff'}
    learning_cells=[]
    for cell_ID in cell_list:
        #Find learning and dis task of current cell (from the list)
        cur_learning_task=df_cells_motor.loc[df_cells_motor['cell_ID']==cell_ID,'task'].item()
        cur_dis_task=df_cells_dis.loc[df_cells_dis['cell_ID']==cell_ID,'task'].item()
        #load cell task for learning and dis
        cur_cell_learning=load_cell_task(cell_task_py_folder,cur_learning_task,cell_ID) #load the cell_task for learning  
        cur_cell_dis=load_cell_task(cell_task_py_folder,cur_dis_task,cell_ID) #load the cell_task for dishabituation
        
        
        file_begin=df_cells_dis.loc[df_cells_dis['cell_ID']==cell_ID,'file_begin'].item()    
        file_end=df_cells_dis.loc[df_cells_dis['cell_ID']==cell_ID,'file_end'].item() 
        dictFilterTrials_dis = {'dir':'filterOff', 'trial_name':'d0v20NS', 'fail':0, 'after_fail':'filterOff','saccade_motion':'filterOff','trial_begin_end':'filterOff','files_begin_end':[file_begin,file_end]}

        #extract trial_df according to dictFilterTrials
        trial_learning_df=cur_cell_learning.filtTrials(dictFilterTrials_learning)
        trial_dis_df=cur_cell_dis.filtTrials(dictFilterTrials_dis)
        #calculate FR around DC
        FR_DC_learning=mean_FR_event(trial_learning_df,'motion_onset',window_pre=100,window_post=320)
        FR_DC_dis=mean_FR_event(trial_dis_df,'motion_onset',window_pre=100,window_post=320)
        #Check if significant
        try:
            [stat,p_val]=scipy.stats.mannwhitneyu(FR_DC_learning,FR_DC_dis,alternative='greater')
        except:
            print('cell without spikes in learning_cell_M1')
            print(cell_ID)
            continue
            
        if p_val<0.05:
            learning_cells.append(cell_ID)
    return learning_cells    
###############################################################   


###############################################################
def motor_learning_curve(cell_list,steps=[3,5,10]) :
    #general parameters
    n_trials=80 #n of trials in a learning block
    learning_tasks=['Motor_learning_CCW_100_25_cue','Motor_learning_CW_100_25_cue','Motor_learning_CW','Motor_learning_CCW']
    df_cells_motor=find_cell_entire_block(learning_tasks,cells_db=cells_db,behaviour_db=behaviour_db)
     #PSTH parameters
    filter_margin=200
    window={"timePoint": 'motion_onset',"timeBefore": -400,"timeAfter":850}
    plot_option=0
    PSTH_length=window['timeAfter']-window['timeBefore']-2*filter_margin
    x=np.arange(window['timeBefore']+filter_margin,window['timeAfter']-filter_margin)    
    
    #creation of trial_limits
    for step in steps:        
        cur_begin=0
        trial_limits=[]
        learned_response=[]
        while cur_begin<=n_trials-step:
            cur_cluster=[cur_begin,cur_begin+step]
            trial_limits.append(cur_cluster)
            cur_begin=cur_begin+step
        trial_begin=[x[0] for x in trial_limits]


        for trial_limit in trial_limits:
            counter=0
            for cell_inx,cell_ID in enumerate(cell_list): 
                cur_learning_task=df_cells_motor.loc[df_cells_motor['cell_ID']==cell_ID,'task'].item()
                cur_cell_learning=load_cell_task(cell_task_py_folder,cur_learning_task,cell_ID) #load the cell_task
                dictFilterTrials = {'dir':'filterOff', 'trial_name':'filterOff', 'fail':0, 'after_fail':'filterOff','saccade_motion':'filterOff','trial_begin_end':trial_limit} #keep all non failed trials (80)
                cur_PSTH=cur_cell_learning.PSTH(window,dictFilterTrials,plot_option) 
                cur_PSTH=np.array(cur_PSTH[filter_margin:-filter_margin])
            
                if counter==0:
                    PSTH_increase_array=cur_PSTH
                    counter=counter+1
                else:    
                    PSTH_increase_array=np.concatenate((PSTH_increase_array,cur_PSTH),axis=0)            
            mean_PSTH_increase=np.reshape(PSTH_increase_array,(int(np.size(PSTH_increase_array)/PSTH_length),PSTH_length))
            mean_PSTH_increase=np.average(mean_PSTH_increase,axis=0)
            
            PSTH_baseline=np.mean(np.mean(mean_PSTH_increase[0:200]))
            learned_response_cluster=np.mean(np.mean(mean_PSTH_increase[-window['timeBefore']-filter_margin+100:-window['timeBefore']-filter_margin+320]))
            learned_response.append(learned_response_cluster)
            normalized_mean_PSTH_increase=mean_PSTH_increase-PSTH_baseline
            plt.plot(x,normalized_mean_PSTH_increase)
        
        #Show PSTH of each traces
        plt.xlabel('time from dir_change (ms)')
        plt.ylabel('FR')
        plt.title('average PSTH by trial cluster of cells in list')
        plt.legend(['0','1','2','3','4']) #Improge the legend
        plt.axvline(x=0, color='r') #motion onset
        plt.axvline(x=250, color='r') #dir_change
        plt.show()
        
        #Show learning curve
        plt.plot(trial_begin,learned_response)
        plt.title('Learning curve-'+'step:'+str(step))
        plt.xlabel('trial_cluster')
        plt.ylabel('Learned response (Hz)')
        plt.show()
#########################################################################

#########################################################################
#This funtion retunrs a list of cell which fire more in the passive trials of fixation (in)congruent block than in the previous (next) dishabituation(0-320 ms after motion onset)
#The argument rel_dis_block is a string ('previous' or 'next') and tells whether to cpmapre the learning block to the previous or next dishabituation block
#fixation can be either 'congruent' or 'incongruent'
# event: the event we calculate the FR around ('motion_onset','cue_onset'...)
#time_pre: time in ms for begining of window (put '-' for before the event) 
#time_post: time in ms for end of window (put '-' for before the event) 
def passive_learning_cell_v1(fixation_type='congruent',rel_dis_block='previous',event='motion_onset',time_pre=100,time_post=320):    
    if fixation_type =='congruent':
        learning_tasks=['fixation_right_probes_CCW_100_25_cue','fixation_right_probes_CW_100_25_cue','fixation_right_probes_CW','fixation_right_probes_CCW']
    elif fixation_type =='incongruent':
        learning_tasks=['fixation_wrong_probes_CCW_100_25_cue','fixation_wrong_probes_CW_100_25_cue','fixation_wrong_probes_CW','fixation_wrong_probes_CCW']


    # find cell_list: a list of cells recorded during a whole motor block and the previous/next dishabituation
    df_cells_passive_fixation=find_cell_entire_block(learning_tasks,cells_db=cells_db,behaviour_db=behaviour_db)
    df_cells_dis=find_cell_entire_dis_block(learning_tasks,cells_db=cells_db,behaviour_db=behaviour_db,rel_block=rel_dis_block)
    
    cell_passive_fix_list=df_cells_passive_fixation.loc[:,'cell_ID'].to_list()
    cell_dis_list=df_cells_dis.loc[:,'cell_ID'].to_list()
    
    cell_list=[cell_ID for cell_ID in cell_passive_fix_list if cell_ID in cell_dis_list]
    
    # Uncomment if you want in the list only cells that were recorded during a whole passive INCONGRUENT block
    # learning_incong_tasks=['fixation_wrong_probes_CCW_100_25_cue','fixation_wrong_probes_CW_100_25_cue','fixation_wrong_probes_CW','fixation_wrong_probes_CCW']
    # df_cells_passive_incong=find_cell_entire_block(learning_incong_tasks,cells_db=cells_db,behaviour_db=behaviour_db)
    # cell_passive_incong_list=df_cells_passive_incong.loc[:,'cell_ID'].to_list()
    # cell_list=[cell_ID for cell_ID in cell_passive_fix_list if cell_ID in cell_dis_list and cell_ID in cell_passive_incong_list]
    
    #Step 2- Check FR of cells in cell_list around time of directionnal change and check if significant  
    dictFilterTrials_learning = {'dir':'filterOff', 'trial_name':'v20S', 'fail':0, 'after_fail':'filterOff','saccade_motion':'filterOff','trial_begin_end':'filterOff'}
    
    learning_cells=[]
 
    for cell_ID in cell_list:
        #Find learning and dis task of current cell (from the list)
        cur_learning_task=df_cells_passive_fixation.loc[df_cells_passive_fixation['cell_ID']==cell_ID,'task'].item()
        cur_dis_task=df_cells_dis.loc[df_cells_dis['cell_ID']==cell_ID,'task'].item()
        
        file_begin=df_cells_dis.loc[df_cells_dis['cell_ID']==cell_ID,'file_begin'].item()    
        file_end=df_cells_dis.loc[df_cells_dis['cell_ID']==cell_ID,'file_end'].item() 
        dictFilterTrials_dis = {'dir':'filterOff', 'trial_name':'d0v20S', 'fail':0, 'after_fail':'filterOff','saccade_motion':'filterOff','trial_begin_end':'filterOff','files_begin_end':[file_begin,file_end]}

        #load cell task for learning and dis
        cur_cell_learning=load_cell_task(cell_task_py_folder,cur_learning_task,cell_ID) #load the cell_task for learning  
        cur_cell_dis=load_cell_task(cell_task_py_folder,cur_dis_task,cell_ID) #load the cell_task for dishabituation
        #extract trial_df according to dictFilterTrials
        trial_learning_df=cur_cell_learning.filtTrials(dictFilterTrials_learning)
        trial_dis_df=cur_cell_dis.filtTrials(dictFilterTrials_dis)
        
        #calculate FR around DC
        FR_DC_learning=mean_FR_event(trial_learning_df,'motion_onset',window_pre=time_pre,window_post=time_post)
        FR_DC_dis=mean_FR_event(trial_dis_df,'motion_onset',window_pre=time_pre,window_post=time_post)

        
        #Check if significant
        try:
            [stat,p_val]=scipy.stats.mannwhitneyu(FR_DC_learning,FR_DC_dis,alternative='greater')
        except:
            print('cell without spikes in passive learning cell')
            print(cell_ID)
            continue
            
        if p_val<0.05:
            learning_cells.append(cell_ID)
    return learning_cells    
###############################################################  


#########################################################################
#This funtion retunrs a list of cell which fire more in the passive trials of fixation congruent block than in the fixation incongruent(0-320 ms after motion onset)
# window_begin,window_end are the time window around we calculate the FR to check if the cell is significant
def passive_learning_cell_v2(window_begin,window_end):    
    learning_congruent_tasks=['fixation_right_probes_CCW_100_25_cue','fixation_right_probes_CW_100_25_cue','fixation_right_probes_CW','fixation_right_probes_CCW']
    learning_incongruent_tasks=['fixation_wrong_probes_CCW_100_25_cue','fixation_wrong_probes_CW_100_25_cue','fixation_wrong_probes_CW','fixation_wrong_probes_CCW']
    
    # find cell_list: a list of cells recorded during a whole motor block and the previous/next dishabituation
    df_cells_cong_passive=find_cell_entire_block(learning_congruent_tasks,cells_db=cells_db,behaviour_db=behaviour_db)
    df_cells_incong_passive=find_cell_entire_block(learning_incongruent_tasks,cells_db=cells_db,behaviour_db=behaviour_db)
    cell_passive_cong_list=df_cells_cong_passive.loc[:,'cell_ID'].to_list()
    cell_passive_incong_list=df_cells_incong_passive.loc[:,'cell_ID'].to_list()
    cell_list=[cell_ID for cell_ID in cell_passive_cong_list if cell_ID in cell_passive_incong_list]
    
    #Step 2- Check FR of cells in cell_list around time of directionnal change and check if significant  
    dictFilterTrials_cong = {'dir':'filterOff', 'trial_name':'d45v20S|d315v20S', 'fail':0, 'after_fail':'filterOff','saccade_motion':'filterOff','trial_begin_end':'filterOff'}
    dictFilterTrials_incong = {'dir':'filterOff', 'trial_name':'d0v20S|d0v20S', 'fail':0, 'after_fail':'filterOff','saccade_motion':'filterOff','trial_begin_end':'filterOff'}
    learning_cells=[]
    for cell_ID in cell_list:
        #Find learning and dis task of current cell (from the list)
        cur_cong_task=df_cells_cong_passive.loc[df_cells_cong_passive['cell_ID']==cell_ID,'task'].item()
        cur_incong_task=df_cells_incong_passive.loc[df_cells_incong_passive['cell_ID']==cell_ID,'task'].item()
        #load cell task for learning and dis
        cur_cell_cong=load_cell_task(cell_task_py_folder,cur_cong_task,cell_ID) #load the cell_task for learning  
        cur_cell_incong=load_cell_task(cell_task_py_folder,cur_incong_task,cell_ID) #load the cell_task for dishabituation
        #extract trial_df according to dictFilterTrials
        trial_cong_df=cur_cell_cong.filtTrials(dictFilterTrials_cong)
        trial_incong_df=cur_cell_incong.filtTrials(dictFilterTrials_incong)
        #calculate FR around DC
        FR_DC_cong=mean_FR_event(trial_cong_df,'motion_onset',window_pre=window_begin,window_post=window_end)
        FR_DC_incong=mean_FR_event(trial_incong_df,'motion_onset',window_pre=window_begin,window_post=window_end)
        
        #Normalize FR_DC to baseline for stationnarity problem
        FR_baseline_cong=mean_FR_event(trial_cong_df,'motion_onset',window_pre=-200,window_post=0)
        FR_baseline_incong=mean_FR_event(trial_incong_df,'motion_onset',window_pre=-200,window_post=0)        
        
        #FR normalized
        FR_DC_cong=FR_DC_cong-FR_baseline_cong
        FR_DC_incong=FR_DC_incong-FR_baseline_incong
        
        #Check if significant
        try:
            [stat,p_val]=scipy.stats.mannwhitneyu(FR_DC_cong,FR_DC_incong,alternative='greater')
        except:
            print('cell without spikes in passive learning cell')
            print(cell_ID)
            continue
            
        if p_val<0.01:
            learning_cells.append(cell_ID)
    return learning_cells    
###############################################################  



#########################################################################
#This funtion retunrs a list of cell from fixation congruent blocks that in passive trials fire more around the direction change than in the motion onset
def passive_learning_cell_v3():    
    learning_congruent_tasks=['fixation_right_probes_CCW_100_25_cue','fixation_right_probes_CW_100_25_cue','fixation_right_probes_CW','fixation_right_probes_CCW']
    
    # find cell_list: a list of cells recorded during a whole motor block and the previous/next dishabituation
    df_cells_cong_passive=find_cell_entire_block(learning_congruent_tasks,cells_db=cells_db,behaviour_db=behaviour_db)
    cell_passive_cong_list=df_cells_cong_passive.loc[:,'cell_ID'].to_list()
    
    #Step 2- Check FR of cells in cell_list around time of directionnal change and check if significant  
    dictFilterTrials_cong = {'dir':'filterOff', 'trial_name':'d45v20S|d315v20S', 'fail':0, 'after_fail':'filterOff','saccade_motion':'filterOff','trial_begin_end':'filterOff'}
    learning_cells=[]
    for cell_ID in cell_passive_cong_list:
        #Find learning and dis task of current cell (from the list)
        cur_cong_task=df_cells_cong_passive.loc[df_cells_cong_passive['cell_ID']==cell_ID,'task'].item()
        #load cell task for learning and dis
        cur_cell_cong=load_cell_task(cell_task_py_folder,cur_cong_task,cell_ID) #load the cell_task for learning  
        #extract trial_df according to dictFilterTrials
        trial_cong_df=cur_cell_cong.filtTrials(dictFilterTrials_cong)
        #calculate FR around baseline
        FR_DC_cong_BL=mean_FR_event(trial_cong_df,'motion_onset',window_pre=-400,window_post=-200)
        #calculate FR around motion onset
        FR_DC_cong_MO=mean_FR_event(trial_cong_df,'motion_onset',window_pre=0,window_post=150)
        #calculate FR around direction change
        FR_DC_cong_DC=mean_FR_event(trial_cong_df,'motion_onset',window_pre=150,window_post=350)        
        #Check if significant
        try:
            [stat,p_val]=scipy.stats.mannwhitneyu(FR_DC_cong_DC,FR_DC_cong_MO,alternative='greater')
           # [stat2,p_val2]=scipy.stats.mannwhitneyu(FR_DC_cong_MO,FR_DC_cong_BL,alternative='greater')
        except:
            print('cell without spikes in passive learning cell')
            print(cell_ID)
            continue
            
        if (p_val<0.01) :
            learning_cells.append(cell_ID)
    return learning_cells    
###############################################################  





###############################################################
#This function receives:
# cell_list: list of cells
    # steps: vector where each element is the number of trials we want to average together in the analyse
    # trial_type: 'fixation_trials'\'motor trials' - the kind of trials we want to consider in the learning curve
    # block_type: 'congruent/'incongruent' - the kind of fixation block we are intersted in
    # sub_bl : subtract baseline (200 first ms of the PSTH)
#The function return a list learned_response_array where each element in a list is a 2d np array (2* n_trials in cluster). 
    #the first row in the array is the first trial of the clusters and the second is the learned response 

# the cell list should include cells that were recorded during entire block of block_type
def passive_learning_curve(cell_list,steps=[3,5,10],trial_type='fixation_trials',block_type='congruent',learned_response_begin=100,learned_response_end=320,sub_bl=True) :
    
    #general parameters
    if trial_type=='fixation_trials':    
        n_trials=72 #n of passive trials in a fixation learning block
        square='S' #for dictfiltertrials
    elif trial_type=='motor_trials': 
        n_trials=8 #n of motor trials in a fixation learning block
        square='NS' #for dictfiltertrials
    
    n_trials=40    
    if block_type=='congruent':
        trial_names='v20'+square
        learning_tasks=['fixation_right_probes_CCW_100_25_cue','fixation_right_probes_CW_100_25_cue','fixation_right_probes_CW','fixation_right_probes_CCW']

    elif block_type=='incongruent':
        trial_names='v20'+square 
        learning_tasks=['fixation_wrong_probes_CCW_100_25_cue','fixation_wrong_probes_CW_100_25_cue','fixation_wrong_probes_CW','fixation_wrong_probes_CCW']

      
    df_cells_task=find_cell_entire_block(learning_tasks,cells_db=cells_db,behaviour_db=behaviour_db)
    
    #PSTH parameters
    filter_margin=200
    window={"timePoint": 'motion_onset',"timeBefore": -800,"timeAfter":800}
    plot_option=0
    PSTH_length=window['timeAfter']-window['timeBefore']-2*filter_margin
    x=np.arange(window['timeBefore']+filter_margin,window['timeAfter']-filter_margin)    
    
    learned_response_array=[]
    #creation of trial_limits
    for step in steps:        
        cur_begin=0
        trial_limits=[]
        learned_response=[]
        while cur_begin<=n_trials-step:
            cur_cluster=[cur_begin,cur_begin+step]
            trial_limits.append(cur_cluster)
            cur_begin=cur_begin+step
        trial_begin=[x[0] for x in trial_limits]
    
    
        for trial_inx,trial_limit in enumerate(trial_limits):
            counter=0
            for cell_inx,cell_ID in enumerate(cell_list):
                
                cur_learning_task=df_cells_task.loc[df_cells_task['cell_ID']==cell_ID,'task'].item()
                cur_cell_learning=load_cell_task(cell_task_py_folder,cur_learning_task,cell_ID) #load the cell_task
                dictFilterTrials = {'dir':'filterOff', 'trial_name':trial_names, 'fail':0, 'after_fail':'filterOff','saccade_motion':'filterOff','trial_begin_end':trial_limit} #keep all non failed trials (80)
                cur_PSTH=cur_cell_learning.PSTH(window,dictFilterTrials,plot_option) 
                cur_PSTH=np.array(cur_PSTH[filter_margin:-filter_margin])
                cur_PSTH_baseline=np.mean(cur_PSTH[0:200]) #calculates baseline of the cell
                if sub_bl:
                    cur_PSTH=cur_PSTH-cur_PSTH_baseline
            
                if counter==0:
                    PSTH_increase_array=cur_PSTH
                    counter=counter+1
                else:    
                    PSTH_increase_array=np.concatenate((PSTH_increase_array,cur_PSTH),axis=0)            
            mean_PSTH_increase=np.reshape(PSTH_increase_array,(int(np.size(PSTH_increase_array)/PSTH_length),PSTH_length))
            mean_PSTH_increase=np.average(mean_PSTH_increase,axis=0)
            
            learned_response_cluster=np.mean(np.mean(mean_PSTH_increase[-window['timeBefore']-filter_margin+learned_response_begin:-window['timeBefore']-filter_margin+learned_response_end]))

            learned_response.append(learned_response_cluster)
            plt.plot(x,mean_PSTH_increase)
        
        #Show PSTH of each traces
        # plt.xlabel('time from dir_change (ms)')
        # plt.ylabel('FR')
        # plt.title('average PSTH by trial cluster of cells in list')
        # plt.legend(['0','1','2','3','4','5','6','7']) #Improge the legend
        # plt.axvline(x=0, color='r') #motion onset
        # plt.axvline(x=250, color='r') #dir_change
        # plt.show()
        
        
        #Stat test (correlation)
        corr_test=scipy.stats.spearmanr(trial_begin, learned_response)
        r_coeff=round(corr_test[0],2)
        pval_coeff=round(corr_test[1],2)
        
        #Show learning curve
        # plt.plot(trial_begin,learned_response)
        # m, b = np.polyfit(trial_begin, learned_response, 1)
        # plt.plot(trial_begin, m*np.array(trial_begin) + b)
        # plt.title('Learning curve-'+'step:'+str(step)+' r:'+str(r_coeff)+' pval:'+str(pval_coeff))
        # plt.xlabel('trial_cluster')
        # plt.ylabel('Learned response (Hz)')
        # plt.show()
        learned_response_array.append(np.array([trial_begin,learned_response])) #create a 2d np array where first row is the beginning of each cluster of trials and the second row is the corresponding learned response
        

    return learned_response_array,pval_coeff,r_coeff     
#########################################################################

#This script 
    #1- select 2 list of cells: cells from fixation congruent block with passive trials higher than dishabituation trials and same for incongruent block
    #2- calculates and plot PSTHs for cells recorde during cong and incong blocks 
    #3- Test for stastistical significance between the congruent and incongruent blocks around directional change (100-300 ms after motion onset)

#List of cells
window_response_begin=100
window_response_end=320
learning_incong_cells=passive_learning_cell_v1(fixation_type='incongruent',rel_dis_block='previous',event='motion_onset',time_pre=window_response_begin,time_post=window_response_end)
learning_cong_cells=passive_learning_cell_v1(fixation_type='congruent',rel_dis_block='previous',event='motion_onset',time_pre=window_response_begin,time_post=window_response_end) 

cell_lists=[learning_cong_cells,learning_incong_cells]

#PSTH parameters
filter_margin=200
window={"timePoint": 'motion_onset',"timeBefore": -600,"timeAfter":800}
dictFilterTrials_learning = {'dir':'filterOff', 'trial_name':'v20S', 'fail':0, 'after_fail':'filterOff','saccade_motion':'filterOff'} #keep all non failed trials (80)
plot_option=0
PSTH_length=window['timeAfter']-window['timeBefore']-2*filter_margin
time_course=np.arange(window['timeBefore']+filter_margin,window['timeAfter']-filter_margin)    

learning_cong_tasks=['fixation_right_probes_CCW_100_25_cue','fixation_right_probes_CW_100_25_cue','fixation_right_probes_CW','fixation_right_probes_CCW']
learning_incong_tasks=['fixation_wrong_probes_CCW_100_25_cue','fixation_wrong_probes_CW_100_25_cue','fixation_wrong_probes_CW','fixation_wrong_probes_CCW']
task_lists=[learning_cong_tasks,learning_incong_tasks]

learned_response_array=[]
for list_inx,cell_list in enumerate(cell_lists):
    cur_task_list=task_lists[list_inx]
    df_cells_task=find_cell_entire_block(cur_task_list,cells_db=cells_db,behaviour_db=behaviour_db)

    for cell_inx,cell_ID in enumerate(cell_list):
        #learning
        cur_learning_task=df_cells_task.loc[df_cells_task['cell_ID']==cell_ID,'task'].item() # find task relevant to the cell within cur_task_list   
        cur_cell_task=load_cell_task(cell_task_py_folder,cur_learning_task,cell_ID) #load the cell_task
        cur_PSTH=cur_cell_task.PSTH(window,dictFilterTrials_learning,plot_option)  #calculate PSTH
        cur_PSTH=np.array(cur_PSTH[filter_margin:-filter_margin]) #removes margins for filter
        cur_PSTH_baseline=np.mean(cur_PSTH[0:200]) #calculates baseline of the cell
        cur_PSTH=cur_PSTH-cur_PSTH_baseline #remove baseline from PSTH

        if cell_inx==0:
            PSTH_array=cur_PSTH
        else:    
            PSTH_array=np.concatenate((PSTH_array,cur_PSTH),axis=0) #add the PSTH of the current cell to PSTH array
    
    pop_PSTH=np.reshape(PSTH_array,(int(np.size(PSTH_array)/PSTH_length),PSTH_length)) # np array where each row is a cell and each column a timepoint in the PSTH
    learned_response_array.append(pop_PSTH) # first element is the pop PSTH of congruent blocks. Second for incongruent
    pop_PSTH2=np.average(pop_PSTH,axis=0) #averge PSTH across cells
    sem_PSTH=(scipy.stats.sem(pop_PSTH, axis=0)) #calculate SEM across cells
    
    plt.plot(time_course,pop_PSTH2)
    plt.fill_between(time_course, pop_PSTH2 - sem_PSTH, pop_PSTH2 + sem_PSTH, alpha=0.2)
    
plt.xlabel('time from ' +window['timePoint']+ '(ms)')
plt.ylabel('FR')
plt.title('Pop PSTH')
plt.legend(['cong','incong'])
plt.axvline(x=0, color='r')
plt.axvline(x=250, color='r')
plt.show()

#statistical (non parametric independent)- Test whether congruent PSTH an incongruent are significantly different around change in direciton
cong_lr_array=np.mean(learned_response_array[0][:,-window['timeBefore']-filter_margin+100:-window['timeBefore']-filter_margin+320],axis=1)
incong_lr_array=np.mean(learned_response_array[1][:,-window['timeBefore']-filter_margin+100:-window['timeBefore']-filter_margin+320],axis=1)
[stat,p_val]=scipy.stats.mannwhitneyu(cong_lr_array,incong_lr_array)
print(p_val)

###########################################################################

###########################################################################
#This function compares learning curve of passive trials in congruent and incongruent blocks
def learning_curve_passive_blocks():

    #Step 1 - select cells
    
    # Method 1: Choose cell significant relative to baseline
    window_response_begin=100
    window_response_end=500
    learning_incong_cells=passive_learning_cell_v1(fixation_type='incongruent',rel_dis_block='previous',event='motion_onset',time_pre=window_response_begin,time_post=window_response_end)
    learning_cong_cells=passive_learning_cell_v1(fixation_type='congruent',rel_dis_block='previous',event='motion_onset',time_pre=window_response_begin,time_post=window_response_end)    
    
    #Step 2 : calculate average PSTH learning curves around motion onset
    Learned_response_begin=100
    Learned_response_end=500
    steps=[1] #number of trials per cluster   
    SUB_BL=True
    learning_array_congruent,pval=passive_learning_curve(learning_cong_cells,steps,trial_type='fixation_trials',block_type='congruent',learned_response_begin=Learned_response_begin,learned_response_end=Learned_response_end,sub_bl=SUB_BL)
    learning_array_incongruent,pval=passive_learning_curve(learning_incong_cells,steps,trial_type='fixation_trials',block_type='incongruent',learned_response_begin=Learned_response_begin,learned_response_end=Learned_response_end,sub_bl=SUB_BL)
    
    #Plot learning curve
    for i,step in enumerate(steps):
        step=steps[i]
        plt.plot(learning_array_congruent[i][0,:],learning_array_congruent[i][1,:])
        plt.plot(learning_array_incongruent[i][0,:],learning_array_incongruent[i][1,:]) 
        plt.title('LC-'+'win:'+str(window_response_begin)+' bin:'+str(Learned_response_begin))
        plt.xlabel('trial_cluster')
        plt.ylabel('LR (Hz)')
        plt.legend(['cong '+'n='+str(len(learning_cong_cells)),'incong '+'n='+str(len(learning_incong_cells))],fontsize=6)
       # plt.ylim((10,25)) 
        plt.show()
###########################################################################

######################################################
#This function receives a list of cell recorded at learst during an fixation congruent block
# For each cell it calculates and plots a PSTH and the learning curve    
def learning_curve_single_cell(cell_list):

    learning_cong_tasks=['fixation_right_probes_CCW_100_25_cue','fixation_right_probes_CW_100_25_cue','fixation_right_probes_CW','fixation_right_probes_CCW']
    df_cells_cong=find_cell_entire_block(learning_cong_tasks,cells_db=cells_db,behaviour_db=behaviour_db)
    
    #PSTH parameters
    filter_margin=200
    window={"timePoint": 'motion_onset',"timeBefore": -400,"timeAfter":850}
    dictFilterTrials_learning = {'dir':'filterOff', 'trial_name':'v20S', 'fail':0, 'after_fail':'filterOff','saccade_motion':'filterOff'} #keep all non failed trials (80)
    plot_option=0
    x=np.arange(window['timeBefore']+filter_margin,window['timeAfter']-filter_margin)    
    
    for cell_ID in cell_list:
        #Cong
        cur_cong_task=df_cells_cong.loc[df_cells_cong['cell_ID']==cell_ID,'task'].item()
        cur_cell_cong=load_cell_task(cell_task_py_folder,cur_cong_task,cell_ID) #load the cell_task  
        cur_PSTH_cong=cur_cell_cong.PSTH(window,dictFilterTrials_learning,plot_option)
        cur_PSTH_cong=np.array(cur_PSTH_cong[filter_margin:-filter_margin])
        plt.plot(x,cur_PSTH_cong)    
    
        plt.title(str(cell_ID))
        plt.axvline(x=0, color='r') #motion_onset
        plt.axvline(x=250, color='r') #dir_change
        plt.show()
        
        Learned_response_begin=100
        Learned_response_end=500
        steps=[1] #number of trials per cluster   
        SUB_BL=True
        passive_learning_curve([cell_ID],steps,trial_type='fixation_trials',block_type='congruent',learned_response_begin=Learned_response_begin,learned_response_end=Learned_response_end,sub_bl=SUB_BL)
######################################################



######################################################
# learning_congruent_tasks=['fixation_right_probes_CCW_100_25_cue','fixation_right_probes_CW_100_25_cue','fixation_right_probes_CW','fixation_right_probes_CCW']
# df_cells_cong=find_cell_entire_block(learning_congruent_tasks,cells_db=cells_db,behaviour_db=behaviour_db)
# cell_list_cong=df_cells_cong.loc[:,'cell_ID'].to_list()

# for cell_ID in cell_list_cong:
#             #Cong
#         cur_cong_task=df_cells_cong.loc[df_cells_cong['cell_ID']==cell_ID,'task'].item()
#         cur_cell_cong=load_cell_task(cell_task_py_folder,cur_cong_task,cell_ID) #load the cell_task  
#         Learned_response_begin=100
#         Learned_response_end=500
#         steps=[1] #number of trials per cluster   
#         SUB_BL=True
#         passive_learning_curve([cell_ID],steps,trial_type='fixation_trials',block_type='congruent',learned_response_begin=Learned_response_begin,learned_response_end=Learned_response_end,sub_bl=SUB_BL)
# ######################################################
    

######################################################
#Find cells in learning paradigm according to activity in mapping paradigm

# mapping_task='8dir_active_passive_interleaved_100_25'

# mapping_cell_list=os.listdir(cell_task_py_folder+mapping_task) #list of strings
# mapping_cell_list=[int(item) for item in mapping_cell_list] #list of ints
# mapping_cell_list = [ elem for elem in mapping_cell_list if elem  not in [7430,7420,7423,7481,7688,7730,7656,7396,7401]] #remove problematic cells


# learning_tasks=['fixation_right_probes_CCW_100_25_cue','fixation_right_probes_CW_100_25_cue','fixation_right_probes_CW','fixation_right_probes_CCW']

# dishabituation_tasks=['Dishabituation_100_25_cue','Dishabituation']

# learning_dir_cells=[] #cells that react more in Base direction than learned direction during mapping
# modulated_cells=[] #learnind_dir_cell that increase their FR after change in direction in motor learning task
# for learning_task in learning_tasks: #run across different motor learning tasks
#         df_output=find_cell_entire_block([learning_task]) #find cells recorded during an entire block of the task
#         cell_task_list=df_output.loc[:,'cell_ID']
#         if 'cue' in learning_task:
#             dis_task=dishabituation_tasks[0]
#         else:
#             dis_task=dishabituation_tasks[1]
#         for cell_ID in cell_task_list:
#             if cell_ID in mapping_cell_list:
#                 cur_cell_learning=load_cell_task(cell_task_py_folder,learning_task,cell_ID) #load the cell_task for learning  
                
#                 trial_learning_df=cur_cell_learning.trials_df
#                 dirs=find_base_and_learned_dir(trial_learning_df) #find base and learned direction of the block while cur cell was recorded
#                 #find base direction
#                 base_dir=dirs[0]
#                 #find learned direction
#                 learned_dir=dirs[1]
#                 #Find target direction after change in direction
#                 if 'CCW' in learning_task:
#                     target_dir_DC=(base_dir+45)%360
#                 else:
#                     target_dir_DC=(base_dir-45)%360
                    
                
#                 dictFilterTrials_mapping_base = {'dir':base_dir, 'trial_name':'v20p', 'fail':0, 'after_fail':'filterOff','saccade_motion':'filterOff','trial_begin_end':'filterOff'}
#                 dictFilterTrials_mapping_learned = {'dir':learned_dir, 'trial_name':'v20p', 'fail':0, 'after_fail':'filterOff','saccade_motion':'filterOff','trial_begin_end':'filterOff'}
#                 dictFilterTrials_mapping_target = {'dir':target_dir_DC, 'trial_name':'v20p', 'fail':0, 'after_fail':'filterOff','saccade_motion':'filterOff','trial_begin_end':'filterOff'}


#                 cur_cell_mapping=load_cell_task(cell_task_py_folder,mapping_task,cell_ID) #load the cell_task 
                
#                 trial_mapping_base_df=cur_cell_mapping.filtTrials(dictFilterTrials_mapping_base)
#                 trial_mapping_learning_df=cur_cell_mapping.filtTrials(dictFilterTrials_mapping_learned)
#                 trial_mapping_target_df=cur_cell_mapping.filtTrials(dictFilterTrials_mapping_target)


#                 n_base=len(trial_mapping_base_df)
#                 n_learned=len(trial_mapping_learning_df)
#                 n_target=len(trial_mapping_target_df)
                
#                 if n_base>=5 and n_learned>=5:
#                     FR_mapping_base_BL=mean_FR_event(trial_mapping_base_df,'motion_onset',window_pre=-600,window_post=-100)
#                     FR_mapping_learned_BL=mean_FR_event(trial_mapping_learning_df,'motion_onset',window_pre=-600,window_post=-100)
#                     FR_mapping_target_BL=mean_FR_event(trial_mapping_target_df,'motion_onset',window_pre=-600,window_post=-100)
                    
                    
#                     #MO=motion onset
#                     FR_mapping_base_MO=mean_FR_event(trial_mapping_base_df,'motion_onset',window_pre=0,window_post=500)
#                     FR_mapping_learned_MO=mean_FR_event(trial_mapping_learning_df,'motion_onset',window_pre=0,window_post=500)            
#                     FR_mapping_target_MO=mean_FR_event(trial_mapping_target_df,'motion_onset',window_pre=0,window_post=500)
                    
#                     cond1=(np.mean(FR_mapping_learned_MO)-np.mean(FR_mapping_learned_BL))>5
#                     cond2=np.mean(FR_mapping_learned_MO)-np.mean(FR_mapping_base_MO)>0
#                     try:
#                         [stat1,p_val1]=scipy.stats.mannwhitneyu(FR_mapping_learned_MO,FR_mapping_learned_BL,alternative='greater',) #Find cells with significnat difference in FR between base and learned direction in active pursuit during motion
#                         [stat2,p_val2]=scipy.stats.mannwhitneyu(FR_mapping_learned_MO,FR_mapping_base_MO,alternative='greater',) #Find cells with significnat difference in FR between base and learned direction in active pursuit during motion

#                     except:
#                         continue
#                     if  cond1 and cond2:

#                         modulated_cells.append(cell_ID)
#                         # #PSTH learning parameters
#                         filter_margin=200
#                         window={"timePoint": 'motion_onset',"timeBefore": -600,"timeAfter":800}
#                         plot_option=0
                        
#                         #PSTH1- learning block
#                         # dictFilterTrials = {'dir':'filterOff', 'trial_name':'filterOff', 'fail':0, 'after_fail':'filterOff','saccade_motion':'filterOff','trial_begin_end':'filterOff',} #keep all non failed trials (80)
#                         # cur_PSTH_learning=cur_cell_learning.PSTH(window,dictFilterTrials,0) 
#                         # cur_PSTH_learning=np.array(cur_PSTH_learning[filter_margin:-filter_margin])
                       
#                         #PSTH2 mapping - base direction
#                         cur_PSTH_base=cur_cell_mapping.PSTH(window,dictFilterTrials_mapping_base,0) 
#                         cur_PSTH_base=np.array(cur_PSTH_base[filter_margin:-filter_margin])
                        
#                         #PSTH2b mapping - target direction
#                         cur_PSTH_target=cur_cell_mapping.PSTH(window,dictFilterTrials_mapping_target,0) 
#                         cur_PSTH_target=np.array(cur_PSTH_target[filter_margin:-filter_margin])
                            
#                         #PSTH3 mapping - learned direction
#                         cur_PSTH_learned=cur_cell_mapping.PSTH(window,dictFilterTrials_mapping_learned,0) 
#                         cur_PSTH_learned=np.array(cur_PSTH_learned[filter_margin:-filter_margin])
    
#                         #PSTH4 dishabituation - learned direction
#                         # dictFilterTrials = {'dir':'filterOff','screen_rot':base_dir ,'trial_name':'d0v20NS', 'fail':0, 'after_fail':'filterOff','saccade_motion':'filterOff','trial_begin_end':[10,40]} #keep all non failed trials (80)
#                         # cur_cell_dis=load_cell_task(cell_task_py_folder,dis_task,cell_ID) #load the cell_task for dishabituation
#                         # cur_PSTH_dis=cur_cell_dis.PSTH(window,dictFilterTrials,0) 
#                         # cur_PSTH_dis=np.array(cur_PSTH_dis[filter_margin:-filter_margin])
                        
#                         plt.title(str(cell_ID))
#                         Time=np.arange(window['timeBefore']+filter_margin,window['timeAfter']-filter_margin)
#                       #  plt.plot(Time,cur_PSTH_learning)
#                         plt.plot(Time,cur_PSTH_base)
#                         plt.plot(Time,cur_PSTH_target)
#                         plt.plot(Time,cur_PSTH_learned)
#                         #plt.plot(Time,cur_PSTH_dis)
#                         plt.legend(['base_dir','target_dir','learned_dir'])
#                         plt.axvline(x=0, color='r') #motion_onset
#                         plt.axvline(x=250, color='r') #dir_change
                 
#                         plt.show()    
    

# #########################################################################




# #########################################################################

# # Method 1: we choose cells significant for all the 80 trials
# start_time = time.time()
    
# tasks=['Motor_learning_CCW_100_25_cue','Motor_learning_CW_100_25_cue','Motor_learning_CW','Motor_learning_CCW']
# sig_dir_change_increase=[]
# dictFilterTrials = {'dir':'filterOff', 'trial_name':'filterOff', 'fail':0, 'after_fail':'filterOff','saccade_motion':'filterOff','trial_begin_end':'filterOff'} # all non failed trials (80)

# for task in tasks:
#         df_output=find_cell_entire_block([task]) #find cells recorded during an entire block of the task
#         cell_task_list=df_output.loc[:,'cell_ID']
#         ### Check if a cell is significant around change in direction
#         for cell_ID in cell_task_list:
#             cur_cell_task=load_cell_task(cell_task_py_folder,task,cell_ID) #load the cell_task
#             trial_df=cur_cell_task.filtTrials(dictFilterTrials)
#             pre_dir_change_FR=mean_FR_event(trial_df,'motion_onset',0,250).to_numpy()#mean across trias 
#             post_dir_change_FR=mean_FR_event(trial_df,'motion_onset',250,650).to_numpy()#mean across trias             
#             [stat,pval]=stats.ttest_rel(post_dir_change_FR,pre_dir_change_FR)
#             if pval<0.05 and (np.mean(post_dir_change_FR)>np.mean(pre_dir_change_FR)):
#                 sig_dir_change_increase.append(cell_ID)
# #PSTH parameters
# filter_margin=200
# window={"timePoint": 'motion_onset',"timeBefore": -400,"timeAfter":850}
# plot_option=0
# PSTH_length=window['timeAfter']-window['timeBefore']-2*filter_margin
# dir_change_timepoint=-window['timeBefore']+filter_margin+250
# x=np.arange(window['timeBefore']+filter_margin,window['timeAfter']-filter_margin)

# trial_limits=[[0,5],[5,10],[10,15],[15,20]]

# #trial_limits=[[45,48],[55,58],[60,65],[75,78],'filterOff']

# for trial_limit in trial_limits:
#     counter=0
#     for task in tasks:
#         df_output=find_cell_entire_block([task]) #find cells recorded during an entire block of the task
#         cell_task_list=df_output.loc[:,'cell_ID'].tolist()
#         cur_sig_dir_change_increase=[x for x in sig_dir_change_increase if x in cell_task_list]
#         ### Check if a cell is significant around change in direction
#         for cell_inx,cell_ID in enumerate(cur_sig_dir_change_increase):  
#             cur_cell_task=load_cell_task(cell_task_py_folder,task,cell_ID) #load the cell_task
#             dictFilterTrials = {'dir':'filterOff', 'trial_name':'filterOff', 'fail':0, 'after_fail':'filterOff','saccade_motion':'filterOff','trial_begin_end':trial_limit} #keep all non failed trials (80)
#             cur_PSTH=cur_cell_task.PSTH(window,dictFilterTrials,plot_option) 
#             cur_PSTH=np.array(cur_PSTH[filter_margin:-filter_margin])
        
#             if counter==0:
#                 PSTH_increase_array=cur_PSTH
#                 counter=counter+1
#             else:    
#                 PSTH_increase_array=np.concatenate((PSTH_increase_array,cur_PSTH),axis=0)            
#             #Uncomment to show single cells
#             # dictFilterTrials = {'dir':'filterOff', 'trial_name':'filterOff', 'fail':0, 'after_fail':'filterOff','saccade_motion':'filterOff','trial_begin_end':'filterOff'} #keep all non failed trials (80)
#             # trial_df=cur_cell_task.filtTrials(dictFilterTrials)
#             # post_dir_change_FR=mean_FR_dir_change(trial_df,0,400).to_numpy()
#             # post_dir_change_FR_mean=round(np.mean(post_dir_change_FR),2)
#             # pre_dir_change_FR=mean_FR_dir_change(trial_df,-300,0).to_numpy()#mean across trias
#             # pre_dir_change_FR_mean=round(np.mean(pre_dir_change_FR),2)
#             # plt.plot(cur_PSTH)
#             # plt.title(str(cell_ID)+'pre: '+str(pre_dir_change_FR_mean)+' post:'+str(post_dir_change_FR_mean))
#             # plt.show()
#     mean_PSTH_increase=np.reshape(PSTH_increase_array,(int(np.size(PSTH_increase_array)/PSTH_length),PSTH_length))
#     mean_PSTH_increase=np.average(mean_PSTH_increase,axis=0)
    
#     PSTH_baseline=np.mean(np.mean(mean_PSTH_increase[0:200]))
#     normalized_mean_PSTH_increase=mean_PSTH_increase-PSTH_baseline
#     plt.plot(x,normalized_mean_PSTH_increase)

# plt.xlabel('time from dir_change (ms)')
# plt.ylabel('FR')
# plt.title('Increasing cells')
# plt.legend(['0','1','2','all trials'])
# plt.axvline(x=0, color='r')
# plt.axvline(x=250, color='r')
# plt.show()
   
# print("--- %s seconds ---" % round(time.time() - start_time))
 
# # ##############################################

# # trial_bin_edges=np.arange(0,50,1)
# # cluster_size=10
# # cur_learned_response_array=[]
# # for trial_inx,trial_limit in enumerate(trial_bin_edges[0:-cluster_size]):
    
# #     learned_response_list=[]
# #     first_trial=trial_bin_edges[trial_inx]
# #     last_trial=trial_bin_edges[trial_inx+cluster_size]
# #     trial_limit=[first_trial,last_trial]
# #     for task in tasks:
# #         df_output=find_cell_entire_block([task]) #find cells recorded during an entire block of the task
# #         cell_task_list=df_output.loc[:,'cell_ID'].tolist()
        
# #         cur_sig_dir_change_increase=[x for x in sig_dir_change_increase if x in cell_task_list]
# #         ### Check if a cell is significant around change in direction
# #         for cell_inx,cell_ID in enumerate(cur_sig_dir_change_increase):
# #             cur_cell_task=load_cell_task(cell_task_py_folder,task,cell_ID) #load the cell_task
# #             dictFilterTrials = {'dir':'filterOff', 'trial_name':'filterOff', 'fail':0, 'after_fail':'filterOff','saccade_motion':'filterOff','trial_begin_end':trial_limit} #keep all non failed trials (80)
# #             trial_df=cur_cell_task.filtTrials(dictFilterTrials)
# #             cell_learned_response=mean_FR_event(trial_df,'dir_change',-100,100).to_numpy()            
# #             learned_response_list.append( np.mean(cell_learned_response))
       
# #     cur_learned_response=np.mean(np.array(learned_response_list))
# #     cur_learned_response_array.append(cur_learned_response)     
# # plt.plot(np.arange(0,50-cluster_size,1),cur_learned_response_array,'-b')
# # plt.xlabel('trial_number (cluster of 3)')
# # plt.ylabel('FR')    
# # plt.show()    

# ##############################################


###########################################################################

#PSTH of single cells in the list for  pre dishabituation, fixation passive learning, and post dishabituation
# passive_learning_cells=passive_learning_cell_v1()

# learning_tasks=['fixation_right_probes_CCW_100_25_cue','fixation_right_probes_CW_100_25_cue','fixation_right_probes_CW','fixation_right_probes_CCW']
# df_cells_learning=find_cell_entire_block(learning_tasks,cells_db=cells_db,behaviour_db=behaviour_db)
# df_cells_disPre=find_cell_entire_dis_block(learning_tasks,cells_db=cells_db,behaviour_db=behaviour_db,rel_block='previous')
# df_cells_disPost=find_cell_entire_dis_block(learning_tasks,cells_db=cells_db,behaviour_db=behaviour_db,rel_block='next')

# #PSTH parameters
# filter_margin=200
# window={"timePoint": 'motion_onset',"timeBefore": -400,"timeAfter":850}
# dictFilterTrials_learning = {'dir':'filterOff', 'trial_name':'v20S', 'fail':0, 'after_fail':'filterOff','saccade_motion':'filterOff'} #keep all non failed trials (80)
# plot_option=0
# PSTH_length=window['timeAfter']-window['timeBefore']-2*filter_margin
# x=np.arange(window['timeBefore']+filter_margin,window['timeAfter']-filter_margin)    
    
# for cell_ID in passive_learning_cells:
#     cur_learning_task=df_cells_learning.loc[df_cells_learning['cell_ID']==cell_ID,'task'].item()
#     cur_cell_learning=load_cell_task(cell_task_py_folder,cur_learning_task,cell_ID) #load the cell_task  
#     cur_PSTH_learning=cur_cell_learning.PSTH(window,dictFilterTrials_learning,plot_option)
#     cur_PSTH_learning=np.array(cur_PSTH_learning[filter_margin:-filter_margin]) 
    
#     file_begin_pre=df_cells_disPre.loc[df_cells_disPre['cell_ID']==cell_ID,'file_begin'].item()    
#     file_end_pre=df_cells_disPre.loc[df_cells_disPre['cell_ID']==cell_ID,'file_end'].item()    
#     dictFilterTrials_preDis = {'dir':'filterOff', 'trial_name':'v20S', 'fail':0, 'after_fail':'filterOff','saccade_motion':'filterOff','files_begin_end':[file_begin_pre,file_end_pre]} 
#     cur_disPre_task=df_cells_disPre.loc[df_cells_disPre['cell_ID']==cell_ID,'task'].item() 
#     cur_cell_disPre=load_cell_task(cell_task_py_folder,cur_disPre_task,cell_ID) #load the cell_task  
#     cur_PSTH_disPre=cur_cell_disPre.PSTH(window,dictFilterTrials_preDis,plot_option)
#     cur_PSTH_disPre=np.array(cur_PSTH_disPre[filter_margin:-filter_margin]) 


        
#     plt.plot(x,cur_PSTH_disPre)   
#     plt.plot(x,cur_PSTH_learning)
    
#     if cell_ID in df_cells_disPost.loc[:,'cell_ID'].to_list():
#         file_begin_post=df_cells_disPost.loc[df_cells_disPost['cell_ID']==cell_ID,'file_begin'].item()    
#         file_end_post=df_cells_disPost.loc[df_cells_disPost['cell_ID']==cell_ID,'file_end'].item()    
#         dictFilterTrials_postDis = {'dir':'filterOff', 'trial_name':'v20S', 'fail':0, 'after_fail':'filterOff','saccade_motion':'filterOff','files_begin_end':[file_begin_post,file_end_post]} 
#         cur_disPost_task=df_cells_disPost.loc[df_cells_disPost['cell_ID']==cell_ID,'task'].item()
#         cur_cell_disPost=load_cell_task(cell_task_py_folder,cur_disPost_task,cell_ID) #load the cell_task  
#         cur_PSTH_disPost=cur_cell_disPost.PSTH(window,dictFilterTrials_postDis,plot_option)
#         cur_PSTH_disPost=np.array(cur_PSTH_disPost[filter_margin:-filter_margin])
#         plt.plot(x,cur_PSTH_disPost)
#     plt.title(str(cell_ID))
#     plt.legend(['dis pre','fix','dis post'])
#     plt.axvline(x=0, color='r') #motion_onset
#     plt.axvline(x=250, color='r') #dir_change
#     plt.show()
###########################################################################