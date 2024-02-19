#this script shows basic behavioural data 
#It shows either for active or passive trials the base and learned velocity for motor learning, congruent, incongruent and dishabituation

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
path="C:/Users/Owner/Documents/Matan/Mati's Lab/FEF learning project/Data_Analyses/Code analyzes/basic classes"
os.chdir(path) 
from behavior_class import session_task

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
behaviour_py_folder="C:/Users/Owner/Documents/Matan/Mati's Lab/FEF learning project/Data_Analyses/behaviour_python_two_monkeys/"

behaviour_db_excel="C:/Users/Owner/Documents/Matan/Mati's Lab/FEF learning project/Data_Analyses/fiona_behaviour_db.xlsx"
behaviour_db=pd.read_excel(behaviour_db_excel)
    
########################################################
def load_behaviour_session(behaviour_py_folder,task,session):
    filename=behaviour_py_folder+task+'/'+str(session)
    infile = open(filename,'rb')
    cur_session = pickle.load(infile)
    infile.close()
    return cur_session
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
#This function receives a row (cur_trial, np_array) with three elements: the x coordinate, the y and a direction in degrees.
#The function then creates a rotation matrix and rotates the coordinate according to the direction 
# the function returns the x and the y in the new direction
def rotation_vector(cur_trial):
    #extract data
    delta_hor=cur_trial[0]
    delta_ver=cur_trial[1]
    theta = np.radians(cur_trial[2]) #angles in radians
    cur_vector=np.array((delta_hor,delta_ver))
    #create rotation matrix
    rotation_matrix = np.array(( (np.cos(theta), np.sin(theta)),
                   (-np.sin(theta),  np.cos(theta)) ))
    #return rotated vector
    rotated_vector=rotation_matrix.dot(cur_vector)
    return rotated_vector


congruent_tasks=['fixation_right_probes_CW_100_25_cue','fixation_right_probes_CCW_100_25_cue','fixation_right_probes_CW','fixation_right_probes_CCW']
incongruent_tasks=['fixation_wrong_probes_CW_100_25_cue','fixation_wrong_probes_CCW_100_25_cue','fixation_wrong_probes_CW','fixation_wrong_probes_CCW']
motor_tasks=['Motor_learning_CW_100_25_cue','Motor_learning_CCW_100_25_cue','Motor_learning_CW','Motor_learning_CCW']
dishabituation_tasks=['Dishabituation_100_25_cue','Dishabituation']

# congruent_tasks=['fixation_right_probes_CW','fixation_right_probes_CCW']
# incongruent_tasks=['fixation_wrong_probes_CW','fixation_wrong_probes_CCW']
# motor_tasks=['Motor_learning_CW','Motor_learning_CCW']
# dishabituation_tasks=['Dishabituation']




Task_lists_general=[motor_tasks,congruent_tasks,incongruent_tasks,dishabituation_tasks]

pupil_dict_list_general=[]
for task_list in Task_lists_general:    
    pupil_dict=get_pupil_dict(task_list)
    pupil_dict_list_general.append(pupil_dict)

    
#%%    

#Trial type: active or passive
trial_type='v20S'
saccade_motion_begin=0 #after motion onset
saccade_motion_end=300 #after motion onset
trial_length=1000

###
#Remove motor tasks if passive trials are analyzed
Task_lists_active=Task_lists_general
pupil_dict_list_active=pupil_dict_list_general

Task_lists_passive=Task_lists_general[1:]
pupil_dict_list_passive=pupil_dict_list_general[1:]

if trial_type=='v20NS':
    Task_lists=Task_lists_active
    pupil_dict_list=pupil_dict_list_active
elif trial_type=='v20S':   
    Task_lists=Task_lists_passive
    pupil_dict_list=pupil_dict_list_passive
###

#Array initialization
learned_vel_array_task=[]
base_vel_array_task=[]

for task_list_inx,task_list in enumerate(Task_lists): #for each group of tasks
    learned_vel_array=np.empty([0,trial_length]) #each row is the learned velocity for a given block
    base_vel_array=np.empty([0,trial_length]) #each row is the learned velocity for a given block
    pupil_dict=pupil_dict_list[task_list_inx]
    for cur_task in task_list: #for each task in the current group of tasl
        session_list=os.listdir(os.path.join(behaviour_py_folder,cur_task))
        for session in session_list:   
            behaviour_session=load_behaviour_session(behaviour_py_folder,cur_task,session)
            
            #Remove fail trials
            behaviour_session=behaviour_session.loc[behaviour_session['fail']==0]
            
            #SEPARATE TO THE DIFFERENT BLOCKS of the same day
            # Separate the cur_behaviour_session df to different dishabituation block (each session includes all the dishabitaution blocks recorded during a given day)
            trials_list=behaviour_session.loc[:]['filename'].tolist()
            trial_number_np=np.array([int(x.split('.',1)[1]) for x in trials_list ])
            first_trial_block_indices=np.where(np.diff(trial_number_np)>=80)[0]
            
            block_begin_indexes=np.append(np.array(0),first_trial_block_indices+1)
            block_end_indexes=np.append(first_trial_block_indices,len(behaviour_session))
            
            #Run across blocks within the day
            for block_index,(cur_begin,cur_end) in enumerate(zip(block_begin_indexes,block_end_indexes)):
                cur_behaviour_block=behaviour_session.iloc[cur_begin:cur_end,:] #exctract a sub df from the data frame
                
                #find filename of last trial in block (for learned direction of washout blocks)
                last_filename=cur_behaviour_block.filename.iloc[-1]
                last_trial= int(re.findall("a.[0-9]{4}", last_filename)[0][2:]) #file end of the current block
                
                
                #remove trial with saccades that begin around MO
                cur_behaviour_block.loc[:,'saccade_onset'] = cur_behaviour_block.apply(lambda row: [saccade[0] if type(saccade)==list else row.saccades[0]  for saccade in row.saccades],axis=1)
                cur_behaviour_block.loc[:,'saccade_motion']= cur_behaviour_block.apply(lambda row: any([saccade_onset>row.motion_onset+saccade_motion_begin and saccade_onset<row.motion_onset+saccade_motion_end   for saccade_onset in row.saccade_onset]) ,axis=1)
                cur_behaviour_block.loc[:,'saccade_motion']= cur_behaviour_block.apply(lambda row: not(row.saccade_motion),axis=1)
                #same for blinks
                cur_behaviour_block.loc[:,'blink_onset'] = cur_behaviour_block.apply(lambda row: [blink[0] if type(blink)==list else row.blinks[0]  for blink in row.blinks],axis=1)
                cur_behaviour_block.loc[:,'blink_motion']= cur_behaviour_block.apply(lambda row: any([blink_onset>row.motion_onset+saccade_motion_begin and blink_onset<row.motion_onset+saccade_motion_end   for blink_onset in row.blink_onset]) ,axis=1)
                cur_behaviour_block.loc[:,'blink_motion']= cur_behaviour_block.apply(lambda row: not(row.blink_motion),axis=1)
                #Remove trial with saccades from behaviour block
                no_sac_index=cur_behaviour_block[(cur_behaviour_block.loc[:,'saccade_motion']==1) & (cur_behaviour_block.loc[:,'blink_motion']==1)].index 
                sac_index=cur_behaviour_block[(cur_behaviour_block.loc[:,'saccade_motion']==0) | (cur_behaviour_block.loc[:,'blink_motion']==0)].index 
                cur_behaviour_block=cur_behaviour_block.loc[no_sac_index,:]
                
                #remove the n_first_trials of dishabituation block:
                n_first_trials=10
                if task_list==dishabituation_tasks:
                    cur_behaviour_block = cur_behaviour_block.iloc[n_first_trials: , :]
                
                
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
                #for non washout
                if task_list in [motor_tasks,congruent_tasks,incongruent_tasks]:
                    if 'CCW' in cur_behaviour_block.iloc[0]['task']:
                        learned_dir=(base_dir+90)%360
                    elif 'CW' in cur_behaviour_block.iloc[0]['task']:   
                        learned_dir=(base_dir-90)%360
                #for washout: find the next row in the database and find the learned direction according to it
                elif task_list==dishabituation_tasks:
                    #Find row of current dishabituation block in database
                    behaviour_db_sess=behaviour_db.loc[behaviour_db['behaviour_session']==session,:]
                    #Find next row based on filename of last trial in washout block
                    behaviour_db_sess2= behaviour_db_sess.loc[behaviour_db_sess['file_begin']==last_trial+1,:]
                    if len(behaviour_db_sess2.index)==0:
                        continue
                    else:
                        if 'CCW' in behaviour_db_sess2.Task.iloc[0]:
                            learned_dir=(base_dir+90)%360
                        elif 'CW' in behaviour_db_sess2.Task.iloc[0]:   
                            learned_dir=(base_dir-90)%360
                        
                    
                if learned_dir==0:
                    learned_pos_shifted=hPos_np_shifted
                elif learned_dir==90:
                    learned_pos_shifted=vPos_np_shifted
                elif learned_dir==180:
                    learned_pos_shifted=-hPos_np_shifted
                elif learned_dir==270:
                    learned_pos_shifted=-vPos_np_shifted
                #From position to velocity
                pos_learned_dir=np.nanmean(learned_pos_shifted,axis=0)
                vel_learned_dir=np.diff(pos_learned_dir[:trial_length+1])*1000 #Average velocity in learned direction for current block
            
                if base_dir==0:
                    base_pos_shifted=hPos_np_shifted
                elif base_dir==90:
                    base_pos_shifted=vPos_np_shifted
                elif base_dir==180:
                    base_pos_shifted=-hPos_np_shifted
                elif base_dir==270:
                    base_pos_shifted=-vPos_np_shifted
                #From position to velocity                
                pos_base_dir=np.nanmean(base_pos_shifted,axis=0)
                vel_base_dir=np.diff(pos_base_dir[:trial_length+1])*1000 #Average velocity in base direction for current block
    
                #add the timecourse of the current block to other block of the current task
                learned_vel_array=np.vstack([learned_vel_array,vel_learned_dir])
                base_vel_array=np.vstack([base_vel_array,vel_base_dir])
    #Add the relative matrix (block*time) to the current group of tasks
    learned_vel_array_task.append(learned_vel_array)
    base_vel_array_task.append(base_vel_array)
            



#%%
#Figures
end_time_graph=400
filter_window=31
#Average across blocks
learned_vel_array_task_mean=[np.nanmean(x[:,0:end_time_graph],axis=0) for x in learned_vel_array_task]
base_vel_array_task_mean=[np.nanmean(x[:,0:end_time_graph],axis=0) for x in base_vel_array_task]
#Smooth data
learned_vel_array_task_mean=[savgol_filter(x, filter_window, 3) for x in learned_vel_array_task_mean]
base_vel_array_task_mean=[savgol_filter(x, filter_window, 3) for x in base_vel_array_task_mean]

title_array=['motor','cong','incong','washout']
color_array=['tab:blue','tab:orange','tab:green','tab:red']

#Remove motor task if passive trials are plotted
if trial_type=='v20S':
    title_array.pop(0)
    color_array.pop(0)

#Base velocity
for task_inx,task in enumerate(base_vel_array_task_mean):
    plt.plot(task,color=color_array[task_inx])
plt.xlabel('time from MO (ms)')
plt.ylabel('vel deg/s')
plt.title('base direction')
plt.legend(title_array)
plt.axvline(x=250,color='black')
plt.show()

#Learned velocity
for task_inx,task in enumerate(learned_vel_array_task_mean):
    plt.plot(task,color=color_array[task_inx])
plt.xlabel('time from MO (ms)')
plt.ylabel('vel deg/s')
plt.title('learned direction')
plt.legend(title_array)

plt.axvline(x=250,color='black')
plt.legend(title_array)
plt.show()


