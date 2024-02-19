# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 12:59:37 2021

@author: Owner
"""
from __future__ import print_function
import os.path
import sys


from glob import glob
import pickle
import os
import pandas as pd
import numpy as np
import re
import math
import matplotlib.pyplot as plt
import scipy.signal as signal
from mat4py import loadmat
import sys
from os.path import isfile,join
from scipy.io import savemat


## OBJECT DEFINIION
##########################################
class session_task:
   
    def __init__(self,session_task_trials):
        self.trials_df = session_task_trials # a data frame with all data regarding this behavioral session during a specific task
        
    #this method filters the trials_df according to parameters in dictFilterTrials.
    #To not filter add 'filterOff' as value to the key parameter.
    
    #dictFilterTrials  = {'dir':'filterOff', 'trial_name':'v20a', 'fail':0,'after_fail':0}
    #dir: keeps all the direction X when trialname is dXv20...
    #for trial_name it keeps all trials that contain the pattern in dictFilterTrials['trial_name']
    # n_trials extract n random trials - caution: trials are not kept in order
    #trial_begin_end: a list that extract all trials from rows first to second element or an int and extract only  this row
    #files_begin_end: a list where each element is a number of the maestro files. keep files from first to second element
    #trial_inxs: a list where is element is an index of a trial we want   
    def filtTrials(self, dictFilterTrials):
        
        trials_df = self.trials_df
        
        if  ('files_begin_end' in dictFilterTrials) and (dictFilterTrials['files_begin_end'] !='filterOff'):
            file_begin=dictFilterTrials['files_begin_end'][0]
            file_end=dictFilterTrials['files_begin_end'][1]
            trials_df['file_number']=trials_df['filename'].str.split(".",1)   
            trials_df.loc[:, 'file_number']=trials_df.file_number.map(lambda x: int(x[1]))  
            trials_df=trials_df[(trials_df['file_number']>=file_begin) & (trials_df['file_number']<=file_end)]
        
        if 'fail' in dictFilterTrials and dictFilterTrials['fail'] !='filterOff': 
            trials_df = trials_df[trials_df['fail'] == dictFilterTrials['fail']]
            
        if 'trial_name' in dictFilterTrials and  dictFilterTrials['trial_name'] !='filterOff': 
            trials_df = trials_df[trials_df['trialname'].str.contains(dictFilterTrials['trial_name'],regex=True)]    
            
    
        if 'after_fail' in dictFilterTrials and  dictFilterTrials['after_fail'] !='filterOff':
            #0 will remove all trials AFTER a failure
            #1 will keep only trials after failure
            fail_inx=trials_df[trials_df['fail']==dictFilterTrials['after_fail'] ].index
            trials_df=trials_df.loc[fail_inx[0:-1]+1,:]  
        
        if 'dir' in dictFilterTrials and dictFilterTrials['dir'] !='filterOff':
            trials_df['dir']=trials_df.apply(lambda row: int(re.search('d(.*)v',row['trialname'])[0][1:-1]), axis=1)
            if type (dictFilterTrials['dir'])==list:
                trials_df=trials_df[trials_df['dir'].isin(dictFilterTrials['dir'])]
            else: #if its int or numpy or float...
                trials_df = trials_df[trials_df['dir'] == dictFilterTrials['dir']]
                
                
        if 'screen_rot' in dictFilterTrials and dictFilterTrials['screen_rot'] !='filterOff':
            if type (dictFilterTrials['screen_rot'])==list:
                trials_df=trials_df[trials_df['screen_rotation'].isin(dictFilterTrials['screen_rot'])]
            else: #if its int or numpy or float...
                trials_df = trials_df[trials_df['screen_rotation'] == dictFilterTrials['screen_rot']]                


        if  'saccade_motion' in dictFilterTrials and dictFilterTrials['saccade_motion'] !='filterOff':
            saccade_motion_begin=-200 #after motion onset
            saccade_motion_end=300 #after motion onset
            trials_df.loc[:,'saccade_onset'] = trials_df.apply(lambda row: [saccade[0] if type(saccade)==list else row.saccades[0]  for saccade in row.saccades],axis=1)
            trials_df.loc[:,'saccade_motion']= trials_df.apply(lambda row: any([saccade_onset>row.motion_onset+saccade_motion_begin and saccade_onset<row.motion_onset+saccade_motion_end   for saccade_onset in row.saccade_onset]) ,axis=1)
            trials_df.loc[:,'saccade_motion']= trials_df.apply(lambda row: not(row.saccade_motion),axis=1)
            trials_df=trials_df[trials_df['saccade_motion']]
        
        if  'blink_motion' in dictFilterTrials and dictFilterTrials['blink_motion'] !='filterOff': #delete trials with blinks that begin in the relevant window after motion onset
            blink_window_begin=-200 #after motion onset
            blink_window_end=300 #after motion onset
            trials_df.loc[:,'blink_onset'] = trials_df.apply(lambda row: [blink[0] if type(blink)==list else row.blinks[0]  for blink in row.blinks],axis=1)
            trials_df.loc[:,'blink_motion']= trials_df.apply(lambda row: any([blink_onset>row.motion_onset+blink_window_begin and blink_onset<row.motion_onset+blink_window_end   for blink_onset in row.blink_onset]) ,axis=1)
            trials_df.loc[:,'blink_motion']= trials_df.apply(lambda row: not(row.blink_motion),axis=1)
            trials_df=trials_df[trials_df['blink_motion']]
        
        if  'n_trials' in dictFilterTrials and dictFilterTrials['n_trials'] !='filterOff':
            if dictFilterTrials['n_trials']< trials_df.shape[0]:
                trials_df=trials_df.sample(n=dictFilterTrials['n_trials'])
                
        if  ('trial_begin_end' in dictFilterTrials) and dictFilterTrials['trial_begin_end'] !='filterOff':
            if type(dictFilterTrials['trial_begin_end'])==list:
                trials_df=trials_df.iloc[dictFilterTrials['trial_begin_end'][0]:dictFilterTrials['trial_begin_end'][1],:]
            if type(dictFilterTrials['trial_begin_end'])==int:
                trials_df=trials_df.iloc[[dictFilterTrials['trial_begin_end']],:]
            if type(dictFilterTrials['trial_begin_end'])==np.ndarray:
                trials_df=trials_df.iloc[dictFilterTrials['trial_begin_end'],:]        
        
        if ('trial_inxs' in dictFilterTrials) and dictFilterTrials['trial_inxs'] !='filterOff':
            trials_df=trials_df.iloc[dictFilterTrials['trial_inxs']]
        return trials_df
 
    
########################################################
def load_behaviour_session(behaviour_py_folder,task,session):
    filename=behaviour_py_folder+task+'/'+str(session)
    infile = open(filename,'rb')
    cur_session = pickle.load(infile)
    infile.close()
    return cur_session
########################################################

########################################################
#This function takes as input a behaviour_session object, a kinematic feature (e.g:'hPos') and return the corresponding serie with nan during saccades 
#Inner function:
def nan_saccades_vector(vector,saccade_matrix):
    if not(saccade_matrix): #if there is no saccade
        return vector
    else:
        if type(saccade_matrix[0])==int:
                temp=np.empty([saccade_matrix[-1]-saccade_matrix[0]])
                temp[:]=np.nan
                vector[saccade_matrix[0]:saccade_matrix[-1]]=temp
        elif type(saccade_matrix[0])==list:    
            for row in np.array(saccade_matrix):
                temp=np.empty([row[-1]-row[0]])
                temp[:]=np.nan
                vector[row[0]:row[-1]]=temp
        return vector
         
def nan_saccades_serie(trials_df,kinematic_feature): 
    kinematic_serie=trials_df.apply(lambda row : nan_saccades_vector(row[kinematic_feature],row['saccades']),axis=1)
    trials_df[kinematic_feature]=kinematic_serie
    return trials_df
######################################################## 
    
########################################################
#This function takes as input a behaviour_session object, a kinematic feature (e.g:'hPos') and return the corresponding serie with nan during blinks 
#Inner function:
def nan_blink_vector(vector,blink_matrix):
    if not(blink_matrix): #if there is no blink
        return vector
    else:    
        if type(blink_matrix[0])==int:
                temp=np.empty([blink_matrix[-1]-blink_matrix[0]])
                temp[:]=np.nan
                vector[blink_matrix[0]:blink_matrix[-1]]=temp
        elif type(blink_matrix[0])==list:    
            for row in np.array(blink_matrix):
                temp=np.empty([row[-1]-row[0]])
                temp[:]=np.nan
                vector[row[0]:row[-1]]=temp
        return vector
     
def nan_blinks_serie(trials_df,kinematic_feature): 
    kinematic_serie=trials_df.apply(lambda row : nan_blink_vector(row[kinematic_feature],row['blinks']),axis=1)
    trials_df[kinematic_feature]=kinematic_serie
    return trials_df
######################################################## 

######################################################## 
#this function extract a kinamtic serie around an event
#window pre needs to be negactive to get time before event
def align_serie_event(window,trials_df,kinematic_feature):
    event=window['event']
    window_pre=window['window_pre']
    window_post=window['window_post']
    trials_df[kinematic_feature]=trials_df.apply(lambda row : row[kinematic_feature][int(row[event])+window_pre:int(row[event])+window_post],axis=1)
    return trials_df
######################################################## 

########################################################
#This set of function is used to rotate the hvel to direction of the targret and vvel as the bertical direction.
#This is for task where angle was decided by trial name rather than scren rotation
def create_rotation_matrix(angle_degree):
    theta=np.radians(angle_degree)
    c, s = np.cos(theta), np.sin(theta)
    rotation_matrix =np.array(((c, s), (-s, c)))
    return rotation_matrix

def rotate_vel(trials_df):
    rotated_vel_serie=trials_df.apply(lambda row : np.matmul(create_rotation_matrix(row['dir']),np.vstack([np.array(row['hVel']),np.array(row['vVel'])])),axis=1)
    return rotated_vel_serie
 ########################################################   
 
#########       SAVE BEHAVIOURAL DATA IN PYTHON #####

#This script converts all the behavioral in the matlab folder into python session_task object save them in the python folder
# behaviour_matlab_folder="C:/Users/Owner/Documents/Matan/Mati's Lab/FEF learning project/Data_Analyses/behaviour_matlab_two_monkeys/"
# behaviour_py_folder="C:/Users/Owner/Documents/Matan/Mati's Lab/FEF learning project/Data_Analyses/behaviour_python_two_monkeys/"

# task_list=os.listdir(behaviour_matlab_folder) #for all task
# #task_list=Tasks=['8dir_saccade'] #for specific task(s)
# for cur_task in task_list:
#     session_list=os.listdir(behaviour_matlab_folder+cur_task)
#     session_list=[item[0:-4] for item in session_list] #remove '.mat'
#     # Continue to next task if directory already exists
#     if os.path.isdir(behaviour_py_folder+cur_task):
#          continue
   
#     for cur_session in session_list:
#         matlab_file=(behaviour_matlab_folder+cur_task+'/'+cur_session+'.mat')
#         data=loadmat(matlab_file)
#         #convert dict to df
#         cur_sess=data['session_task']['trials'] 
#         sess_df=pd.DataFrame.from_dict(cur_sess, orient='columns', dtype=None, columns=None)
#         sess_df['dir']=sess_df.apply(lambda row: int(re.search('d[0-9]+',row['trialname'])[0][1:]), axis=1)

#         cur_behaviour_session=session_task(sess_df)
#         filename=behaviour_py_folder+cur_task+'/'+cur_session

#         #create a directory if it does not exist
#         if not os.path.isdir(behaviour_py_folder+cur_task):
#             os.makedirs(behaviour_py_folder+cur_task)

#         outfile = open(filename,'wb')
#         pickle.dump(cur_behaviour_session,outfile)
#         outfile.close()
######################################################## 




######################################################## 

#Basic Behavior plot

# #Paramteres that needs to be filled
# task='8dir_saccade_100_25'
# window={'event':'motion_onset','window_pre':-200,'window_post':1000}
# dictFilterTrials={'fail':0}

# path="C:/Users/Owner/Documents/Matan/Mati's Lab/FEF learning project/Data_Analyses"
# os.chdir(path) 
# #For all this program:
# behaviour_py_folder="behaviour_python/"
# session_list=os.listdir(behaviour_py_folder+task)

# #initialize arrays
# absVel_array=np.empty([len(session_list),window['window_post']-window['window_pre']])
# absVel_array[:]=np.nan
# for sess_inx,session in enumerate(session_list):
#     cur_behaviour_session=load_behaviour_session(behaviour_py_folder,task,session)
#     trials_df=cur_behaviour_session.filtTrials(dictFilterTrials)

#     for kinematic_feature in ['hVel','vVel']:#nan saccades and blinks and extract trials df with relevant trials
#         #trials_df=nan_saccades_serie(trials_df,kinematic_feature)
#         trials_df=nan_blinks_serie(trials_df,kinematic_feature)
#         trials_df=align_serie_event(window,trials_df,kinematic_feature)
    
#     #rotate velocity in direction of movement (hVel is the direction of the target)
#     rotated_vel_serie= rotate_vel(trials_df)
    
#     hVel_rotated=rotated_vel_serie.apply(lambda x: x[0,:]) #extract rotated hVel
#     hVel_array=np.array([np.array(xi) for xi in hVel_rotated]) #concatenate all the vectors from the serie
#     hVel=np.nanmean(hVel_array,axis=0) #average
#     absVel_smoothed=signal.savgol_filter(hVel,51,3) #smooth  
#     absVel_array[sess_inx,:]=absVel_smoothed #fill the relevant row in the array
    
# absVel_mean=np.nanmean(absVel_array,axis=0)#average across sessions

# #Figure    
# timecourse=np.arange(window['window_pre'],window['window_post'])
# plt.plot(timecourse,absVel_mean)
# plt.xlabel('time from onset')
# plt.ylabel('velocity (deg/s)')
# plt.title('velocity of mapping trials')
# plt.axvline(x=0,color='red')
# plt.show()
######################################################## 

#script to load sessions of a specific task
# behaviour_py_folder="C:/Users/Owner/Documents/Matan/Mati's Lab/FEF learning project/Data_Analyses/behaviour_python_two_monkeys/"
# task='Dishabituation_100_25_cue'
# session_list=os.listdir(join(behaviour_py_folder,task))
# for session in session_list: #all sessions recorded during dishabituation taskjs
#     cur_behaviour_session=load_behaviour_session(behaviour_py_folder,task,session)
