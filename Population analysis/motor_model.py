
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
########################################################


#liste of cells recorded during dishabituation
Tasks=['Dishabituation_100_25_cue','Dishabituation']
dishabituation_cells_list=[]
for cur_task in Tasks:
    dishabituation_cells_list=dishabituation_cells_list+os.listdir(join(cell_task_py_folder,cur_task))
    dishabituation_cells_list=[int(x) for x in dishabituation_cells_list]
cutoff_cell=8229 #cutoff between yasmin and fiona
yasmin_cell_list=[x for x in dishabituation_cells_list if x>cutoff_cell]
fiona_cell_list=[x for x in dishabituation_cells_list if x<=cutoff_cell]
cur_cell_list=dishabituation_cells_list
    
window_post=250
window_pre=0
SMOOTH_EDGE=200 #for cutting the tail due to smoothing
win_before=window_pre-SMOOTH_EDGE
win_after=window_post+SMOOTH_EDGE

window={"timePoint": 'motion_onset',"timeBefore":win_before,"timeAfter":win_after}
saccade_motion_parameter='filterOff'

n_min_trials=30
smooth_parameter_velocity=30    
smooth_parameter_acceleration=50    
plot_option=0

FR_passive_array=[]
FR_pred_passive_array=[]
FR_pred_passive_array2=[]

FR_active_array=[]
FR_pred_active_array=[]
FR_pred_active_array2=[]

score_active_array=[]
score_passive_array=[]    
for cell_ID in cur_cell_list:
    for cur_task in Tasks:
        try:
            cur_cell_task=load_cell_task(cell_task_py_folder,cur_task,cell_ID)
        except:
            continue
    try:
        dictFilterTrials_pre={'trial_name':'v20S|v20NS', 'fail':0,'saccade_motion':saccade_motion_parameter}
        trials_df=cur_cell_task.filtTrials(dictFilterTrials_pre)
    except:
        continue
    
    #division into blocks
    trials_list=trials_df.loc[:]['filename_name'].tolist()
    trial_number_np=np.array([int(x.split('.',1)[1]) for x in trials_list ])
    first_trial_block_indices=np.where(np.diff(trial_number_np)>=80)[0]
    
    block_begin_indexes= trials_df.index[np.append(np.array(0),first_trial_block_indices+1)]
    block_end_indexes=trials_df.index[np.append(first_trial_block_indices,len(trials_df)-1)]
    
    file_begin_indexes=[trial_number_np[0]]+list(trial_number_np[first_trial_block_indices+1])
    file_end_indexes=list(trial_number_np[first_trial_block_indices])+[trial_number_np[-1]]

    for block_index,(cur_begin,cur_end) in enumerate(zip(file_begin_indexes,file_end_indexes)):
        try:
            dictFilterTrials_active={ 'trial_name':'v20NS', 'fail':0 ,'saccade_motion':saccade_motion_parameter,'files_begin_end':[cur_begin,cur_end]}
            trials_df_active=cur_cell_task.filtTrials(dictFilterTrials_active)  

            dictFilterTrials_passive={ 'trial_name':'v20S', 'fail':0 ,'saccade_motion':saccade_motion_parameter,'files_begin_end':[cur_begin,cur_end]}
            trials_df_passive=cur_cell_task.filtTrials(dictFilterTrials_passive)          
        except:
            continue
        if len(trials_df_active)<n_min_trials or len(trials_df_passive)<n_min_trials:
            print(cell_ID)
            continue
        
        
        #kinematics     
        #calculate velocity based on position in ative
        hPosActive= np.array(trials_df_active.apply(lambda row: [(row.hPos[win_before+row.motion_onset:win_after+row.motion_onset])],axis=1).to_list())
        hPosActive=np.squeeze(hPosActive,axis=1)
        #velocity
        hVelActive=np.diff(hPosActive,axis=1)*1000
        hVelActiveMean=np.nanmean(hVelActive,axis=0)
        hVelActiveMean=smooth_data(hVelActiveMean,smooth_parameter_velocity)
        #acceleration
        hAccActiveMean=np.diff(hVelActiveMean)*1000
        hAccActiveMean=smooth_data(hAccActiveMean,smooth_parameter_acceleration)
        #remove edges    
        hVelActiveMean=hVelActiveMean[SMOOTH_EDGE:-SMOOTH_EDGE+1] #The +1 compensates for the diff that cuts one element
        hAccActiveMean=hAccActiveMean[SMOOTH_EDGE:-SMOOTH_EDGE+2]
        
        #calculate velocity based on position
        vPosActive= np.array(trials_df_active.apply(lambda row: [(row.vPos[win_before+row.motion_onset:win_after+row.motion_onset])],axis=1).to_list())
        vPosActive=np.squeeze(vPosActive,axis=1)
        #velocity
        vVelActive=np.diff(vPosActive,axis=1)*1000
        vVelActiveMean=np.nanmean(vVelActive,axis=0)
        vVelActiveMean=smooth_data(vVelActiveMean,smooth_parameter_velocity)
        #acceleration
        vAccActiveMean=np.diff(vVelActiveMean)*1000
        vAccActiveMean=smooth_data(vAccActiveMean,smooth_parameter_acceleration)
        #remove edges    
        vVelActiveMean=vVelActiveMean[SMOOTH_EDGE:-SMOOTH_EDGE+1] #The +1 compensates for the diff that cuts one element
        vAccActiveMean=vAccActiveMean[SMOOTH_EDGE:-SMOOTH_EDGE+2]
        
        
        #calculate velocity based on position in passive
        hPosPassive= np.array(trials_df_passive.apply(lambda row: [(row.hPos[win_before+row.motion_onset:win_after+row.motion_onset])],axis=1).to_list())
        hPosPassive=np.squeeze(hPosPassive,axis=1)
        #velocity
        hVelPassive=np.diff(hPosPassive,axis=1)*1000
        hVelPassiveMean=np.nanmean(hVelPassive,axis=0)
        hVelPassiveMean=smooth_data(hVelPassiveMean,smooth_parameter_velocity)
        #acceleration
        hAccPassiveMean=np.diff(hVelPassiveMean)*1000
        hAccPassiveMean=smooth_data(hAccPassiveMean,smooth_parameter_acceleration)
        #remove edges    
        hVelPassiveMean=hVelPassiveMean[SMOOTH_EDGE:-SMOOTH_EDGE+1] #The +1 compensates for the diff that cuts one element
        hAccPassiveMean=hAccPassiveMean[SMOOTH_EDGE:-SMOOTH_EDGE+2]
        
        #calculate velocity based on position
        vPosPassive= np.array(trials_df_passive.apply(lambda row: [(row.vPos[win_before+row.motion_onset:win_after+row.motion_onset])],axis=1).to_list())
        vPosPassive=np.squeeze(vPosPassive,axis=1)
        #velocity
        vVelPassive=np.diff(vPosPassive,axis=1)*1000
        vVelPassiveMean=np.nanmean(vVelPassive,axis=0)
        vVelPassiveMean=smooth_data(vVelPassiveMean,smooth_parameter_velocity)
        #acceleration
        vAccPassiveMean=np.diff(vVelPassiveMean)*1000
        vAccPassiveMean=smooth_data(vAccPassiveMean,smooth_parameter_acceleration)
        #remove edges    
        vVelPassiveMean=vVelPassiveMean[SMOOTH_EDGE:-SMOOTH_EDGE+1] #The +1 compensates for the diff that cuts one element
        vAccPassiveMean=vAccPassiveMean[SMOOTH_EDGE:-SMOOTH_EDGE+2]

        #Rotation
        #Find relevant direction
        cur_dir=trials_df_active.iloc[0]['dir']
        cur_dir_rad=math.radians(cur_dir)
        #active
        rotation_matrix=np.array([[math.cos(cur_dir_rad),-math.sin(cur_dir_rad)],[math.sin(cur_dir_rad),math.cos(cur_dir_rad)]])
        Vel_mean_active=np.matmul(rotation_matrix,np.vstack([hVelActiveMean,vVelActiveMean]))[0,:]
        Acc_mean_active=np.matmul(rotation_matrix,np.vstack([hAccActiveMean,vAccActiveMean]))[0,:]
        #passive
        Vel_mean_passive=np.matmul(rotation_matrix,np.vstack([hVelPassiveMean,vVelPassiveMean]))[0,:]
        Acc_mean_passive=np.matmul(rotation_matrix,np.vstack([hAccPassiveMean,vAccPassiveMean]))[0,:]
        
        #calculate PSTH - baseline
        #baseline calculation 
        dictFilterTrials_both={'fail':0 ,'saccade_motion':saccade_motion_parameter,'files_begin_end':[cur_begin,cur_end]}
        FR_BL=np.nanmean(cur_cell_task.get_mean_FR_event(dictFilterTrials_both,window['timePoint'],window_pre=-300,window_post=-100))
        #PSTH
        PSTH_active=cur_cell_task.PSTH(window,dictFilterTrials_active,smooth_option=1)
        PSTH_active=PSTH_active[SMOOTH_EDGE:-SMOOTH_EDGE]
        PSTH_active=PSTH_active-FR_BL
        
        PSTH_passive=cur_cell_task.PSTH(window,dictFilterTrials_passive,smooth_option=1)
        PSTH_passive=PSTH_passive[SMOOTH_EDGE:-SMOOTH_EDGE]
        PSTH_passive=PSTH_passive-FR_BL
                      

        #prepare the data for the regression
        x_active=np.transpose(np.vstack([Vel_mean_active,Acc_mean_active]))
        y_active=PSTH_active
        model_active = LinearRegression().fit(x_active, y_active)
        
        x_passive=np.transpose(np.vstack([Vel_mean_passive,Acc_mean_passive]))
        y_passive=PSTH_passive
        model_passive = LinearRegression().fit(x_passive, y_passive)
        r_sq_active = round(model_active.score(x_active, y_active),3)
        r_sq_passive = round(model_passive.score(x_passive, y_passive),3)
        #r_sq_passive2 = round(model_active.score(x_passive, y_passive),3)
        #r_sq_active2 = round(model_passive.score(x_active, y_active),3)

        
        #Plot the actual and predicted FR of the cell
        y_pred_active=model_active.predict(x_active)
        y_pred_active2=model_passive.predict(x_active)
        y_pred_passive=model_passive.predict(x_passive)
        y_pred_passive2 = model_active.predict(x_passive)


        
        if plot_option:
            if r_sq_active>0.8:
                fig, ax = plt.subplots(3)
                ax[0].plot(y_active)
                ax[0].plot(y_pred_active)
                ax[0].set_xlabel('time from MO')
                ax[0].set_ylabel('FR')
                ax[0].legend(['FR','model_active'])
                ax[0].set_title('active-active '+str(cell_ID)+' r^2='+str(r_sq_active) )
                
                ax[1].plot(y_passive)
                ax[1].plot(y_pred_passive)
                ax[1].set_xlabel('time from MO')
                ax[1].set_ylabel('FR')
                ax[1].legend(['FR','model_passive'])
                ax[1].set_title('passive-active '+str(cell_ID)+' r^2='+str(r_sq_passive) )
        
                # ax[2].plot(y_passive)
                # ax[2].plot(y_pred_passive2)
                # ax[2].set_xlabel('time from MO')
                # ax[2].set_ylabel('FR')
                # ax[2].legend(['FR','model_active'])
                # ax[2].set_title('passive-active '+ str(cell_ID)+' r^2='+str(r_sq_passive2) )
                
                fig.tight_layout()
                plt.show()
        score_active_array.append(r_sq_active)   
        score_passive_array.append(r_sq_passive)   

        FR_passive_array.append(np.nanmean(PSTH_passive)) 
        FR_pred_passive_array.append(np.nanmean(y_pred_passive)) #model passive 
        FR_pred_passive_array2.append(np.nanmean(y_pred_passive2)) #model active 

        FR_active_array.append(np.nanmean(PSTH_active)) 
        FR_pred_active_array.append(np.nanmean(y_pred_active)) #model passive 
        FR_pred_active_array2.append(np.nanmean(y_pred_active2)) #model active         
#%% predict FR passive from active model

n_cells=round(len(cur_cell_list)*0.2)
high_scores_inxs=np.argsort(score_active_array)[-n_cells:]
FR_passive_array=np.array(FR_passive_array)
FR_pred_passive_array=np.array(FR_pred_passive_array)
FR_pred_passive_array2=np.array(FR_pred_passive_array2)

pval=round(scipy.stats.wilcoxon(FR_passive_array[high_scores_inxs],FR_pred_passive_array2[high_scores_inxs])[1],3)
plt.scatter(FR_passive_array[high_scores_inxs],FR_pred_passive_array2[high_scores_inxs])
plt.xlabel('FR in passive trials')
plt.ylabel(' predicted FR in passive trials')
plt.title('Predicted vs actual FR passive - p=' + str(pval))
plt.axline((0, 0), slope=1, color='black')
plt.show()

#%% predict FR active from passive model

n_cells=round(len(cur_cell_list)*0.2)
high_scores_inxs=np.argsort(score_passive_array)[-n_cells:]
FR_active_array=np.array(FR_active_array)
FR_pred_active_array=np.array(FR_pred_active_array)
FR_pred_active_array2=np.array(FR_pred_active_array2)

pval=round(scipy.stats.wilcoxon(FR_active_array[high_scores_inxs],FR_pred_active_array2[high_scores_inxs])[1],3)
plt.scatter(FR_active_array[high_scores_inxs],FR_pred_active_array2[high_scores_inxs])
plt.xlabel('FR in active trials')
plt.ylabel(' predicted FR in active trials')
plt.title('Predicted vs actual FR active - p=' + str(pval))
plt.axline((0, 0), slope=1, color='black')
plt.show()