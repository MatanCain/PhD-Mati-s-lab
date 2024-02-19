#this script compares a learning block to the dishabituation before and after it
#It shows a PSTH in time and scatter plots 

# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 11:50:49 2021

@author: Owner
q"""
from __future__ import print_function

import pickle
import os
from os.path import isfile,join
import pandas as pd
import numpy as np
import scipy.stats as stats
import math
import matplotlib.pyplot as plt
from mat4py import loadmat
import sys
path="C:/Users/Owner/Documents/Matan/Mati's Lab/FEF learning project/Data_Analyses/Code analyzes/basic classes"
os.chdir(path) 
from neuron_class import *
import warnings
from cancel_pupil_bias import *
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn import metrics
import random
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
warnings.filterwarnings("ignore")
########################################################
path="C:/Users/Owner/Documents/Matan/Mati's Lab/FEF learning project/Data_Analyses/Code analyzes"
os.chdir(path) 
#For all this program:
cell_task_py_folder="C:/Users/Owner/Documents/Matan/Mati's Lab/FEF learning project/Data_Analyses/units_task_python/"
behaviour_py_folder="C:/Users/Owner/Documents/Matan/Mati's Lab/FEF learning project/Data_Analyses/behaviour_python/"
cells_db_excel="C:/Users/Owner/Documents/Matan/Mati's Lab/FEF learning project/Data_Analyses/fiona_cells_db.xlsx"
cells_db=pd.read_excel(cells_db_excel)

behaviour_db_excel="C:/Users/Owner/Documents/Matan/Mati's Lab/FEF learning project/Data_Analyses/fiona_behaviour_db.xlsx"
behaviour_db=pd.read_excel(behaviour_db_excel)
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

######################################### 
#this function split the data into training and test samples where the classes are equally represented   
def train_test_split_manual(data_array,label_array, training_test_size):    
    #indexes of the different classes in label_array
    ones_inxs=np.where(label_array==1)[0]
    zeros_inxs=np.where(label_array==0)[0]
    #find smallest frequency (classest with less trials)
    unique_elements, counts_elements = np.unique(label_array, return_counts=True) #frequency of each label
    min_freq=np.min(counts_elements)
    #find indexes for training
    training_size=math.floor((1-training_test_size)*min_freq) #for each block
    test_size=math.floor(training_test_size*min_freq) #for each block

    class_1_train_inxs=np.sort(np.random.choice(ones_inxs,training_size,replace=False))
    class_0_train_inxs=np.sort(np.random.choice(zeros_inxs,training_size,replace=False))
    train_inxs=np.concatenate((class_1_train_inxs,class_0_train_inxs),axis=0)
    #find indexes for test
    class_1_non_train_inxs=np.delete(ones_inxs,np.isin(ones_inxs,class_1_train_inxs))
    class_0_non_train_inxs=np.delete(zeros_inxs,np.isin(zeros_inxs,class_0_train_inxs))
    class_1_test_inxs=np.sort(np.random.choice(class_1_non_train_inxs,test_size,replace=False))
    class_0_test_inxs=np.sort(np.random.choice(class_0_non_train_inxs,test_size,replace=False))
    
    test_inxs=np.concatenate((class_1_test_inxs,class_0_test_inxs),axis=0)
    #prepare outputs
    x_train=data_array[train_inxs,:]
    y_train=label_array[train_inxs,:]
    x_test=data_array[test_inxs,:]
    y_test=label_array[test_inxs,:]
    
    return x_train,x_test,y_train,y_test
#########################################
    
congruent_tasks=['fixation_right_probes_CW_100_25_cue','fixation_right_probes_CCW_100_25_cue','fixation_right_probes_CW','fixation_right_probes_CCW']
incongruent_tasks=['fixation_wrong_probes_CW_100_25_cue','fixation_wrong_probes_CCW_100_25_cue','fixation_wrong_probes_CW','fixation_wrong_probes_CCW']
motor_tasks=['Motor_learning_CW_100_25_cue','Motor_learning_CCW_100_25_cue','Motor_learning_CW','Motor_learning_CCW']

#Choose a learning block you want to analyze:
cur_task_list=incongruent_tasks

#Find cells recorded during whole learning block, the dishabituation around it and 80 trials during mapping 
learning_block_cells=list(find_cell_entire_block(cur_task_list,cells_db=cells_db,behaviour_db=behaviour_db).cell_ID)

dis_pre_cells_pd=(find_cell_entire_dis_block(cur_task_list,cells_db=cells_db,behaviour_db=behaviour_db,rel_block='previous'))
dis_pre_cells=list(dis_pre_cells_pd.cell_ID)

dis_next_cells_pd=(find_cell_entire_dis_block(cur_task_list,cells_db=cells_db,behaviour_db=behaviour_db,rel_block='next'))
dis_next_cells=list(dis_next_cells_pd.cell_ID)

#Select cells recorded in motor block and washout around it (to check stability between washout)
cell_list=[x for x in learning_block_cells if x in dis_pre_cells and x in dis_next_cells] 
#########################################
#Remove unstable cells from the complete cells list
#It concatenates the FR in the dishabituation before, the learning block and then the dishabituation after
#If the FR is correlated with the index of the trials then the cells is removed from the analyze
unstable_cells=[]
for cell_ID in cell_list:
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

    try:
        corr,pval=stats.kruskal(FR_dis_pre.to_numpy(),FR_block.to_numpy(),FR_dis_post.to_numpy())
    except:
        pval=0
    if pval<0.05:
        unstable_cells.append(cell_ID)
stable_cells=[x for x in cell_list if x not in unstable_cells ]    
#########################################      

#%%#########################################
#Decoding using logisitic regression and single cells


# bin_length=200
# bin_begins=np.arange(-500,500,bin_length)
# #parameteres for logistic regression
# model = LogisticRegression(solver='liblinear', random_state=0)
# training_test_size=0.2 #in percentage

# accuracy_array=np.empty([len(stable_cells),len(bin_begins)])
# accuracy_array[:]=np.nan

# trial_type='v20S'
# #saccade_option=1 #cancel trials with saccades that begin between 0 and 300 after motion onset
# saccade_option='filterOff'#doesnt cancel trials with saccades that begin between 0 and 300 after motion onset


# if trial_type=='v20S': 
#     n_trials_cong=72
#     n_trials_active_dis=40
#     training_test_size=0.2 #in percentage
#     n_iterations=200

# elif trial_type=='v20NS': 
#     n_trials_cong=8
#     n_trials_active_dis=40
#     training_test_size=0.125 #in percentage
#     n_iterations=200

# for cell_inx,cell_ID in enumerate(stable_cells):
#     for bin_inx,bin_begin in enumerate(bin_begins):
#         bin_end=bin_begin+bin_length
#         for cur_task in cur_task_list:
#             try:
#                 cur_cell_task=load_cell_task(cell_task_py_folder,cur_task,cell_ID)
#             except:
#                 continue
#         dictFilterTrials_block={'dir':'filterOff', 'trial_name':trial_type, 'fail':0,'saccade_motion':saccade_option}    
#         FR_block=cur_cell_task.get_mean_FR_event(dictFilterTrials_block,'motion_onset',window_pre=bin_begin,window_post=bin_end)
               
#         dishabituation_tasks=['Dishabituation_100_25_cue','Dishabituation']
#         for cur_task in dishabituation_tasks:
#             try:
#                 cur_cell_task=load_cell_task(cell_task_py_folder,cur_task,cell_ID)
#             except:
#                 continue
        
#         #Dishabituation before
#         fb_pre=dis_pre_cells_pd.loc[dis_pre_cells_pd['cell_ID']==cell_ID,'file_begin'].item()
#         fe_pre=dis_pre_cells_pd.loc[dis_pre_cells_pd['cell_ID']==cell_ID,'file_end'].item()
#         dictFilterTrials_pre={'dir':'filterOff', 'trial_name':trial_type, 'fail':0,'saccade_motion':saccade_option,'files_begin_end':[fb_pre,fe_pre]}    
#         FR_dis_pre=cur_cell_task.get_mean_FR_event(dictFilterTrials_pre,'motion_onset',window_pre=bin_begin,window_post=bin_end)
        
#         #Dishabituation after
#         fb_post=dis_next_cells_pd.loc[dis_next_cells_pd['cell_ID']==cell_ID,'file_begin'].item()
#         fe_post=dis_next_cells_pd.loc[dis_next_cells_pd['cell_ID']==cell_ID,'file_end'].item()
#         dictFilterTrials_post={'dir':'filterOff', 'trial_name':trial_type, 'fail':0,'saccade_motion':saccade_option,'files_begin_end':[fb_post,fe_post]}    
#         FR_dis_post=cur_cell_task.get_mean_FR_event(dictFilterTrials_post,'motion_onset',window_pre=bin_begin,window_post=bin_end)
        
#         #concatenation and correlation
#         FR_conc=np.concatenate((FR_block.to_numpy(),FR_dis_pre.to_numpy(),FR_dis_post.to_numpy()),axis=0).reshape(-1, 1)
#         n_learning_trials=FR_block.size
#         n_dis_trials=FR_dis_post.size+FR_dis_pre.size
#         label_array=np.concatenate((np.ones(n_learning_trials,),np.zeros(n_dis_trials,)),axis=0).reshape(-1, 1)
        
#         n_test_trials=round(np.size(FR_conc)*training_test_size/100) #in trials
#         accuracy_iterations_array=np.empty(n_iterations)
#         accuracy_iterations_array[:]=np.nan
#         for ii in np.arange(n_iterations):
#             X_train, X_test, y_train, y_test = train_test_split_manual(FR_conc, label_array, training_test_size)
#             model.fit(X_train, y_train)
#             y_pred=model.predict(X_test)
#             cur_accuracy=metrics.accuracy_score(y_test, y_pred)
#             accuracy_iterations_array[ii]=cur_accuracy
            
#         accuracy_array[cell_inx,bin_inx]=np.mean(accuracy_iterations_array)

# plt.figure
# plt.scatter(accuracy_array[:,2],accuracy_array[:,3])
# plt.axvline(x=0.5)
# plt.axhline(y=0.5)
# plt.xlabel('pre')
# plt.ylabel('post')


#%%#########################################
#Decoding using logistic regression and population of cells

bin_length=200

bin_begins=np.arange(-500,500,bin_length)

accuracy_array=np.empty([len(bin_begins)])
accuracy_array[:]=np.nan

trial_type='v20NS'
saccade_option='filterOff'#doesnt cancel trials with saccades that begin between 0 and 300 after motion onset
n_block_dis=2 # 2 if pre and post, 1 if only pre

if trial_type=='v20S': 
    n_trials_cong=72
    n_trials_active_dis=40
    training_test_size=0.2 #in percentage
    n_iterations=200

elif trial_type=='v20NS': 
    n_trials_cong=8
    n_trials_active_dis=40
    training_test_size=0.125 #in percentage
    n_iterations=200

remove_baseline_blockwise=0

win_begin_bl=-500
win_end_bl=-300

model = LogisticRegression(random_state=0) 
weight_array=np.empty([len(stable_cells),n_iterations,len(bin_begins)])#weight of each cell in the current iteration and current bin
weight_array[:]=np.nan

label_array=np.concatenate((np.ones(n_trials_cong,),np.zeros(n_block_dis*n_trials_active_dis,)),axis=0).reshape(-1, 1)
for bin_inx,bin_begin in enumerate(bin_begins):
    bin_end=bin_begin+bin_length
    data_array=np.empty([n_trials_cong+n_block_dis*n_trials_active_dis,len(stable_cells)])
    data_array[:]=np.nan
    for cell_inx,cell_ID in enumerate(stable_cells):
        for cur_task in cur_task_list:
            try:
                cur_cell_task=load_cell_task(cell_task_py_folder,cur_task,cell_ID)
            except:
                continue
        dictFilterTrials_block={'dir':'filterOff', 'trial_name':trial_type, 'fail':0,'saccade_motion':saccade_option}    
        FR_block=cur_cell_task.get_mean_FR_event(dictFilterTrials_block,'motion_onset',window_pre=bin_begin,window_post=bin_end)
               
        dishabituation_tasks=['Dishabituation_100_25_cue','Dishabituation']
        for cur_task in dishabituation_tasks:
            try:
                cur_cell_task=load_cell_task(cell_task_py_folder,cur_task,cell_ID)
            except:
                continue
        
        #Dishabituation before
        fb_pre=dis_pre_cells_pd.loc[dis_pre_cells_pd['cell_ID']==cell_ID,'file_begin'].item()
        fe_pre=dis_pre_cells_pd.loc[dis_pre_cells_pd['cell_ID']==cell_ID,'file_end'].item()
        dictFilterTrials_pre={'dir':'filterOff', 'trial_name':trial_type, 'fail':0,'saccade_motion':saccade_option,'files_begin_end':[fb_pre,fe_pre]}    
        FR_dis_pre=cur_cell_task.get_mean_FR_event(dictFilterTrials_pre,'motion_onset',window_pre=bin_begin,window_post=bin_end)
        
        #Dishabituation after
        fb_post=dis_next_cells_pd.loc[dis_next_cells_pd['cell_ID']==cell_ID,'file_begin'].item()
        fe_post=dis_next_cells_pd.loc[dis_next_cells_pd['cell_ID']==cell_ID,'file_end'].item()
        dictFilterTrials_post={'dir':'filterOff', 'trial_name':trial_type, 'fail':0,'saccade_motion':saccade_option,'files_begin_end':[fb_post,fe_post]}    
        FR_dis_post=cur_cell_task.get_mean_FR_event(dictFilterTrials_post,'motion_onset',window_pre=bin_begin,window_post=bin_end)
        
        if remove_baseline_blockwise: #remove the baseline for each block (500 to 300 before MO)
            BL_block=cur_cell_task.get_mean_FR_event(dictFilterTrials_block,'motion_onset',window_pre=win_begin_bl,window_post=win_end_bl)
            BL_dis_pre=cur_cell_task.get_mean_FR_event(dictFilterTrials_pre,'motion_onset',window_pre=win_begin_bl,window_post=win_end_bl)
            BL_dis_post=cur_cell_task.get_mean_FR_event(dictFilterTrials_post,'motion_onset',window_pre=win_begin_bl,window_post=win_end_bl)        
            BL_block_mean=np.nanmean(np.array(BL_block))
            BL_disPre_mean=np.nanmean(np.array(BL_dis_pre))
            BL_disPost_mean=np.nanmean(np.array(BL_dis_post))
            FR_block=FR_block.to_numpy()-BL_block_mean
            FR_dis_pre=FR_dis_pre.to_numpy()-BL_disPre_mean
            FR_dis_post=FR_dis_post.to_numpy()-BL_disPost_mean
            FR_conc=np.concatenate((FR_block,FR_dis_pre,FR_dis_post),axis=0).reshape(-1, 1)
        else:
            #concatenation and correlation
            FR_conc=np.concatenate((FR_block.to_numpy(),FR_dis_pre.to_numpy(),FR_dis_post.to_numpy()),axis=0).reshape(-1, 1)

        if FR_dis_pre.size!=n_trials_active_dis or FR_dis_post.size!=n_trials_active_dis or FR_block.size!=n_trials_cong:
            continue
        data_array[:,cell_inx]=np.transpose(FR_conc)
    data_array2=data_array[:, ~np.isnan(data_array).any(axis=0)]
    accuracy_iterations_array=np.empty(n_iterations)
    accuracy_iterations_array[:]=np.nan
    for ii in np.arange(n_iterations):
        X_train, X_test, y_train, y_test = train_test_split_manual(data_array2, label_array, training_test_size)
        model.fit(X_train, y_train)
        y_pred=model.predict(X_test)
        cur_accuracy=metrics.accuracy_score(y_test, y_pred)
        accuracy_iterations_array[ii]=cur_accuracy
        weight_array[~np.isnan(data_array).any(axis=0),ii,bin_inx]=np.std(X_train, 0)*model.coef_
    accuracy_array[bin_inx]=np.mean(accuracy_iterations_array)
    
plt.figure()
timecourse=np.array(bin_begins)+(bin_length/2) #center of bins
plt.plot(timecourse,accuracy_array)
plt.xlabel('time from MO (ms)')
plt.xticks(timecourse)
plt.ylabel('% accuracy')
plt.title('logistic regression population')
plt.ylim((0.3,1))
plt.grid()
plt.show()

#histogram of normalized weights of the neurons 
weight_array_mean=np.nanmean(np.abs(weight_array),axis=1) #average across iterations
weight_array_mean[np.isnan(weight_array_mean)] = 0 #to prevent from nans to be sorted as high!

fig, ax = plt.subplots(1,len(bin_begins))
hist_bins=np.arange(0,4,0.5)
for bin_inx,bin_begin in enumerate(bin_begins):
    ax[bin_inx].hist(weight_array_mean[:,bin_inx],hist_bins)
    ax[bin_inx].set_title('bin '+str(bin_inx))
plt.suptitle('histogram of normalized weights of the neurons')    
plt.tight_layout()

#%%#########################################
#PSTH of cells with high weights in logistic regression
# n_max=5
# trial_type='v20NS'
# dictFilterTrials={'dir':'filterOff', 'trial_name':trial_type, 'fail':0}  
# filter_margin=100
# window={"timePoint": 'motion_onset',"timeBefore":-900,"timeAfter":950}
# time_course=np.arange(window['timeBefore']+filter_margin,window['timeAfter']-filter_margin)

# for bin_inx,bin_begin in enumerate(bin_begins):
#     weight_array_mean=np.nanmean(np.abs(weight_array),axis=1) #average across iterations
#     sorted_inxs=np.argsort(weight_array_mean[:,bin_inx])[-n_max:] #inxs of n_max cells with heighest weights
    
#     for inx in sorted_inxs:
#         cell_ID=stable_cells[inx]

#         for cur_block in [cur_task_list,dishabituation_tasks]:
#             for cur_task in cur_block:
#                 try:
#                     cur_cell_task=load_cell_task(cell_task_py_folder,cur_task,cell_ID)
#                 except:
#                     continue
#             PSTH=cur_cell_task.PSTH(window,dictFilterTrials,smooth_option=1)
#             PSTH=PSTH[filter_margin:-filter_margin]
#             plt.plot(time_course,PSTH)
#         plt.axvline(x=bin_begin,color='red')
#         plt.axvline(x=bin_begin+bin_length,color='red')
#         plt.xlabel('time from MO')
#         plt.ylabel('FR')
#         plt.legend(('fixation block','washout'))
#         plt.title(str(cell_ID)+' bin:'+str(bin_inx))
#         plt.show()
