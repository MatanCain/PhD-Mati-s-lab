# neural behaviour correlation
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
behaviour_py_folder="C:/Users/Owner/Google Drive/Documents/Studies/Mati's Lab/FEF learning project/Data_Analyses/behaviour_python/"
behaviour_db_excel="C:/Users/Owner/Google Drive/Documents/Studies/Mati's Lab/FEF learning project/Data_Analyses/fiona_behaviour_db.xlsx"
behaviour_db=pd.read_excel(behaviour_db_excel)

cell_task_py_folder="C:/Users/Owner/Google Drive/Documents/Studies/Mati's Lab/FEF learning project/Data_Analyses/units_task_python/"
cells_db_excel="C:/Users/Owner/Google Drive/Documents/Studies/Mati's Lab/FEF learning project/Data_Analyses/fiona_cells_db.xlsx"
cells_db=pd.read_excel(cells_db_excel)    
########################################################
def load_behaviour_session(behaviour_py_folder,task,session):
    filename=behaviour_py_folder+task+'/'+str(session)
    infile = open(filename,'rb')
    cur_session = pickle.load(infile)
    infile.close()
    return cur_session
########################################################

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

########################################################
#This function receives a list of tasks (e.g: ['8dir_active_passive_interleaved','8dir_active_passive_interleaved_100_25']
# For all the behavioral session recorded during those tasks the function calculates the pupil biased model for vPos
    # it returns a dictionnary where keys are sessions (strings) and values a list [slope, intercept] for each session
def get_pupil_dict(Tasks): 
    #Tasks=['8dir_active_passive_interleaved','8dir_active_passive_interleaved_100_25']
    pupil_dict={}
    for task in Tasks:
        session_list=os.listdir(os.path.join(behaviour_py_folder,task))
    
        for session in session_list:
            [slope,intercept]=get_pupil_correction_model(session,task,behaviour_py_folder,0)
            pupil_dict[session]=[slope,intercept]
    return pupil_dict     
########################################################    
########################################################
#This function receives a 2d np array (data) an a 1d np array (shift_array). 
#The function shifts each row according to the corresponding element in shift_array. (for example if first element of shift array is -2 the first row of data will be shifted by 2 to the left)
def circshift_row_independently(data,shift_array):
    rows, column_indices = np.ogrid[:data.shape[0], :data.shape[1]]
    
    # Use always a negative shift, so that column_indices are valid.
    # (could also use module operation)
    shift_array[shift_array < 0] += data.shape[1]
    column_indices = column_indices - shift_array[:, np.newaxis]
    
    shifted_array = data[rows, column_indices]
    return shifted_array
########################################################
dishabituation_tasks=['Dishabituation_100_25_cue','Dishabituation']
motor_tasks=['Motor_learning_CW_100_25_cue','Motor_learning_CCW_100_25_cue','Motor_learning_CW','Motor_learning_CCW']
#Choose a learning block you want to analyze:
cur_task_list=motor_tasks

#Find cells recorded during whole learning block, the dishabituation around it and 80 trials during mapping 
motor_block_cells=list(find_cell_entire_block(cur_task_list,cells_db=cells_db,behaviour_db=behaviour_db).cell_ID)
dis_pre_cells_pd=find_cell_entire_dis_block(cur_task_list,cells_db=cells_db,behaviour_db=behaviour_db,rel_block='previous')
dis_pre_cells=list(find_cell_entire_dis_block(cur_task_list,cells_db=cells_db,behaviour_db=behaviour_db,rel_block='previous').cell_ID)

complete_cells=[x for x in motor_block_cells if x in dis_pre_cells]

learning_cells=[]
learning_cells_sig=[]

for cell_inx,cell_ID in enumerate(complete_cells):
    for cur_task in motor_tasks:
        try:
            cur_cell_task=load_cell_task(cell_task_py_folder,cur_task,cell_ID)
            task=cur_task
        except:
            continue
    #Motor block
    dictFilterTrials={'dir':'filterOff', 'trial_name':'v20NS', 'fail':0, 'after_fail':'filterOff'}    
    FR_motor=cur_cell_task.get_mean_FR_event(dictFilterTrials,'motion_onset',window_pre=300,window_post=600)

    for cur_task in dishabituation_tasks:
        try:
            cur_cell_task=load_cell_task(cell_task_py_folder,cur_task,cell_ID)
        except:
            continue
    #Dishabituation before
    fb_pre=dis_pre_cells_pd.loc[dis_pre_cells_pd['cell_ID']==cell_ID,'file_begin'].item()
    fe_pre=dis_pre_cells_pd.loc[dis_pre_cells_pd['cell_ID']==cell_ID,'file_end'].item()
    dictFilterTrials_pre={'dir':'filterOff', 'trial_name':'filterOff', 'fail':0, 'after_fail':'filterOff','saccade_motion':'filterOff','files_begin_end':[fb_pre,fe_pre]}    
    FR_dis_pre=cur_cell_task.get_mean_FR_event(dictFilterTrials_pre,'motion_onset',window_pre=300,window_post=600)
    
    
    try:
        stat,p_val1=scipy.stats.wilcoxon(FR_motor,FR_dis_pre)
    except:
        p_val=1

    if np.mean(FR_motor)<np.mean(FR_dis_pre) :
        learning_cells.append(cell_ID)
        if p_val1<0.05 :   
            learning_cells_sig.append(cell_ID)
#########################################     
#calculate pupuil dict
pupil_dict=get_pupil_dict(motor_tasks)            
#####################################


## Things to add
#smoothing the learned velocity
# add the possibility to remove saccades



#%%

cur_cell_list=learning_cells_sig
saccade_option=1 #1 remove trials with saccade, 0 don't   
n_group=5 #number of cluster when sorting trials according to learned response 
    
FR_array=np.zeros((len(cur_cell_list),n_group))
learned_vel_array=np.zeros((len(cur_cell_list),n_group))

#PSTHs for motor blocks
filter_margin=100 #for Psth margin
window={"timePoint": 'motion_onset',"timeBefore":-400,"timeAfter":950}
time_course=np.arange(window['timeBefore']+filter_margin,window['timeAfter']-filter_margin)

gp_inx=list(np.arange(0,n_group))
PSTH_dict = { cur_inx :np.zeros((len(cur_cell_list),np.size(time_course)))  for cur_inx in gp_inx }

time_dynamics_length=1000
baseVel_dict = { cur_inx :np.zeros((len(cur_cell_list),time_dynamics_length))  for cur_inx in gp_inx }
learnedVel_dict = { cur_inx :np.zeros((len(cur_cell_list),time_dynamics_length))  for cur_inx in gp_inx }


for cell_inx,cell_ID in enumerate(cur_cell_list):
    for cur_task in motor_tasks:
        try:
            cur_cell_task=load_cell_task(cell_task_py_folder,cur_task,cell_ID)
            task=cur_task
        except:
            continue

    dictFilterTrials={'dir':'filterOff', 'trial_name':'filterOff', 'fail':'filterOff', 'after_fail':'filterOff','saccade_motion':'filterOff','blink_motion':'filterOff'} 
    trials_df=cur_cell_task.filtTrials(dictFilterTrials)
    
    #find session,file_begin and file end of the cell:
    session=cells_db[cells_db['cell_ID']==cell_ID]['session'].item()
    fb_cell=int(trials_df.iloc[0]['filename_name'][-4:])
    fe_cell=int(trials_df.iloc[-1]['filename_name'][-4:])
    
    #look at session in behaviour db
    cur_behaviour_db=behaviour_db[behaviour_db['behaviour_session']==session]
    cur_behaviour_db=cur_behaviour_db[cur_behaviour_db['Task']==task]
    cur_behaviour_db=cur_behaviour_db[(cur_behaviour_db['file_begin']<=fb_cell) & (cur_behaviour_db['file_end']>=fe_cell)]
    
    #load the relevant behaviour session
    behaviour_session=load_behaviour_session(behaviour_py_folder,task,session)
    
    #filters from the dataframe only trials where the cell was recorded
    file_number=(behaviour_session.iloc[:]['filename']).tolist()
    file_inx=[ inx for inx,x in enumerate(file_number) if int(x[-4:])>=fb_cell and int(x[-4:])<=fe_cell]
    cur_behaviour_block=behaviour_session.iloc[file_inx,:] #df with only the trials where the cell was recorded
    cur_behaviour_block=cur_behaviour_block[cur_behaviour_block['fail']==0]
    
    #Trial type: active or passive
    trial_type='v20NS'
    saccade_motion_begin=0 #after motion onset
    saccade_motion_end=300 #after motion onset
    

    
    if saccade_option==1:
        
        #remove trial with saccades that begin around MO
        cur_behaviour_block.loc[:,'saccade_onset'] = cur_behaviour_block.apply(lambda row: [saccade[0] if type(saccade)==list else row.saccades[0]  for saccade in row.saccades],axis=1)
        cur_behaviour_block.loc[:,'saccade_motion']= cur_behaviour_block.apply(lambda row: any([saccade_onset>row.motion_onset+saccade_motion_begin and saccade_onset<row.motion_onset+saccade_motion_end   for saccade_onset in row.saccade_onset]) ,axis=1)
        cur_behaviour_block.loc[:,'saccade_motion']= cur_behaviour_block.apply(lambda row: not(row.saccade_motion),axis=1)
        #same for blinks
        cur_behaviour_block.loc[:,'blink_onset'] = cur_behaviour_block.apply(lambda row: [blink[0] if type(blink)==list else row.blinks[0]  for blink in row.blinks],axis=1)
        cur_behaviour_block.loc[:,'blink_motion']= cur_behaviour_block.apply(lambda row: any([blink_onset>row.motion_onset+saccade_motion_begin and blink_onset<row.motion_onset+saccade_motion_end   for blink_onset in row.blink_onset]) ,axis=1)
        cur_behaviour_block.loc[:,'blink_motion']= cur_behaviour_block.apply(lambda row: not(row.blink_motion),axis=1)
        #Remove trial with saccades fron behaviour block
        no_sac_index=cur_behaviour_block[(cur_behaviour_block.loc[:,'saccade_motion']==1) & (cur_behaviour_block.loc[:,'blink_motion']==1)].index 
        sac_index=cur_behaviour_block[(cur_behaviour_block.loc[:,'saccade_motion']==0) | (cur_behaviour_block.loc[:,'blink_motion']==0)].index 
        cur_behaviour_block=cur_behaviour_block.loc[no_sac_index,:]
        
    
    #extract the position (hor and vel) and the pupil  from columns relevant for the current cell task
    hPos_serie=cur_behaviour_block.loc[cur_behaviour_block.trialname.str.contains(trial_type)]['hPos']
    vPos_serie=cur_behaviour_block.loc[cur_behaviour_block.trialname.str.contains(trial_type)]['vPos']
    pupil_serie=cur_behaviour_block.loc[cur_behaviour_block.trialname.str.contains(trial_type)]['pupil']
    MO_serie=cur_behaviour_block.loc[cur_behaviour_block.trialname.str.contains(trial_type)]['motion_onset']
    hPos_serie=hPos_serie.reset_index(drop=True)#reset indexes from 0
    vPos_serie=vPos_serie.reset_index(drop=True)
    pupil_serie=pupil_serie.reset_index(drop=True)
    MO_serie=MO_serie.reset_index(drop=True)
        
    #extract motion onset:
    motion_onset_list=MO_serie.to_list()
    motion_onset_shift=[-x for x in motion_onset_list]
    motion_onset_shift=np.array(motion_onset_shift)
    
    #convert hPos to rectangular np array and shift such tat motion onset is at index 0
    hPos_list=hPos_serie.tolist()
    hPos_list=[np.append(hPos_array,float('nan')) for hPos_array in hPos_list] #append nan to end of vectors (up to the longest one)
    hPos_np=pd.DataFrame(hPos_list).to_numpy() 
    hPos_np_shifted=circshift_row_independently(hPos_np,motion_onset_shift)#shift such that index 0 will be motion onset
    
    vPos_list=vPos_serie.tolist()
    vPos_list=[np.append(vPos_array,float('nan')) for vPos_array in vPos_list  ]
    vPos_np=pd.DataFrame(vPos_list).to_numpy()  
    vPos_np_shifted=circshift_row_independently(vPos_np,motion_onset_shift)#shift such that index 0 will be motion onset
    
    pupil_list=pupil_serie.tolist()
    pupil_list=[np.append(pupil_array,float('nan')) for pupil_array in pupil_list  ]
    pupil_np=pd.DataFrame(pupil_list).to_numpy()  
    pupil_np_shifted=circshift_row_independently(pupil_np,motion_onset_shift)#shift such that index 0 will be motion onset
    
    #cancel pupil bias
    cur_slope=pupil_dict[session][0]
    cur_intercept=pupil_dict[session][1]
    vPos_np_shifted=vPos_np_shifted-(cur_slope*pupil_np_shifted+cur_intercept)
    
    #Base direction
    base_dir=cur_behaviour_block['screen_rotation'].iloc[0]
       
    #Learned direction
    if 'CCW' in cur_behaviour_block.iloc[0]['task']:
        learned_dir=(base_dir+90)%360
    elif 'CW' in cur_behaviour_block.iloc[0]['task']:   
        learned_dir=(base_dir-90)%360
    
        
        
    if learned_dir==0:
        learned_pos_shifted=hPos_np_shifted
    elif learned_dir==90:
        learned_pos_shifted=vPos_np_shifted
    elif learned_dir==180:
        learned_pos_shifted=-hPos_np_shifted
    elif learned_dir==270:
        learned_pos_shifted=-vPos_np_shifted

    
    if base_dir==0:
        base_pos_shifted=hPos_np_shifted
    elif base_dir==90:
        base_pos_shifted=vPos_np_shifted
    elif base_dir==180:
        base_pos_shifted=-hPos_np_shifted
    elif base_dir==270:
        base_pos_shifted=-vPos_np_shifted
    
    learned_vel_begin=100
    learned_vel_end=300
    
    learned_vel_shifted=np.diff(learned_pos_shifted,axis=1)*1000
    learned_vel=np.nanmean(learned_vel_shifted[:,learned_vel_begin:learned_vel_end],1)

    base_vel_shifted=np.diff(base_pos_shifted,axis=1)*1000
    
    #FR in cell during learning period
    window_begin=100
    window_end=300
    #active base
    dictFilterTrials={'dir':'filterOff', 'trial_name':'v20NS', 'fail':0, 'after_fail':'filterOff','saccade_motion':saccade_option,'blink_motion':saccade_option}    
    FR_learning=cur_cell_task.get_mean_FR_event(dictFilterTrials,'motion_onset',window_pre=window_begin,window_post=window_end)
    
    
    # mean_FR_np=mean_FR.to_numpy()
    # total_movement_np=np.array(total_movement)
    sort_inxs=learned_vel.argsort()
    FR_learning_sorted=FR_learning[sort_inxs]        
    learned_vel_sorted=learned_vel[sort_inxs]
    
    
    FR_learning_grouped=np.array_split(FR_learning_sorted, n_group)
    learned_vel_grouped=np.array_split(learned_vel_sorted, n_group)
    
    FR_learning_grouped_mean=[np.nanmean(x) for x in FR_learning_grouped]
    learned_vel_grouped_mean=[np.nanmean(x) for x in learned_vel_grouped]
    
    FR_array[cell_inx,:]=FR_learning_grouped_mean
    learned_vel_array[cell_inx,:]=learned_vel_grouped_mean

    inx_group=np.array_split(sort_inxs, n_group)
    for grp_inx in np.arange(0,n_group):
        cur_inxs=inx_group[grp_inx]
        dictFilterTrials_inx={'dir':'filterOff', 'trial_name':'v20NS', 'fail':0, 'after_fail':'filterOff','saccade_motion':saccade_option,'blink_motion':saccade_option, 'trial_inxs':list(cur_inxs)}  
        trials_df=cur_cell_task.filtTrials(dictFilterTrials_inx)
        PSTH=cur_cell_task.PSTH(window,dictFilterTrials_inx,plot_option=0,smooth_option=1) 
        PSTH=PSTH[filter_margin:-filter_margin]
        PSTH_dict[grp_inx][cell_inx,:]=PSTH
        
        learnedVel_dict[grp_inx][cell_inx,:]=smooth_data(np.nanmean(learned_vel_shifted[cur_inxs,0:time_dynamics_length],axis=0),30)
        baseVel_dict[grp_inx][cell_inx,:]=smooth_data(np.nanmean(base_vel_shifted[cur_inxs,0:time_dynamics_length],axis=0),30)
        

        

plt.plot(np.nanmean(learned_vel_array,axis=0),color='tab:orange')
plt.ylabel('learned velocity (deg/s)')
plt.xlabel('# group')
plt.title('Average learned velocity')
plt.show()

plt.plot(np.nanmean(FR_array,axis=0))
plt.ylabel('FR')
plt.xlabel('# group')
plt.title('Average FR')
plt.show()

#Plot psth
for grp_inx in np.arange(0,n_group):
    cur_gp_PSTH=np.nanmean(PSTH_dict[grp_inx],axis=0)
    plt.plot(time_course,cur_gp_PSTH)
plt.axvline(x=100,color='black')
plt.axvline(x=300,color='black')
plt.legend(np.arange(0,n_group))
plt.xlabel('time from MO')
plt.ylabel('FR')
plt.title('PSTH')
plt.show()
    
#plot learned velocity and base velocity
for grp_inx in np.arange(0,n_group):
    cur_gp_learnedVel=np.nanmean(learnedVel_dict[grp_inx],axis=0)

    plt.plot(np.arange(time_dynamics_length),cur_gp_learnedVel)
plt.axvline(x=100,color='black')
plt.axvline(x=300,color='black')
plt.legend(np.arange(0,n_group))
plt.xlabel('time from MO')
plt.ylabel('vel (deg/s)')
plt.title('learned velocity')
plt.show()


    
for grp_inx in np.arange(0,n_group):
    cur_gp_baseVel=np.nanmean(baseVel_dict[grp_inx],axis=0)

    plt.plot(np.arange(time_dynamics_length),cur_gp_baseVel)
plt.axvline(x=100,color='black')
plt.axvline(x=300,color='black')
plt.legend(np.arange(0,n_group))
plt.xlabel('time from MO')
plt.ylabel('vel (deg/s)')
plt.title('base velocity')
plt.show()


