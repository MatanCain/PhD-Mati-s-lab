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

congruent_tasks=['fixation_right_probes_CW_100_25_cue','fixation_right_probes_CCW_100_25_cue','fixation_right_probes_CW','fixation_right_probes_CCW']
incongruent_tasks=['fixation_wrong_probes_CW_100_25_cue','fixation_wrong_probes_CCW_100_25_cue','fixation_wrong_probes_CW','fixation_wrong_probes_CCW']
dishabituation_tasks=['Dishabituation_100_25_cue','Dishabituation']        
           
#####################################        
# %%
# This script display psth of motor, congruent and incongruent blocks (row 1 to 3, active and passive trials apart) as well as learning curves and dishabituation  blocks (third column, in motor row active and passive trials are shown in the same subplot- dashed vs continous).
plot_option=0
#PSTH of cells that were recorded during complete sessions
interesting_cells=[7691,7693,7867,7951,7395,7568,7622,8053,7250,7251,7312,7335]
#you can choose the complete_cells list to see cells that have been recorded in full congruent and incongruent block
sig_cong_slope_cell=[]
sig_incong_slope_cell=[]
for cell_ID in unstable_cells:
    #PSTH paramters
    filter_margin=100 #for Psth margin
    window={"timePoint": 'motion_onset',"timeBefore":-300,"timeAfter":500}
    time_course=np.arange(window['timeBefore']+filter_margin,window['timeAfter']-filter_margin)
    dictFilterTrials_active={'dir':'filterOff', 'trial_name':'v20NS', 'fail':0, 'after_fail':'filterOff','saccade_motion':'filterOff'}
    dictFilterTrials_passive={'dir':'filterOff', 'trial_name':'v20S', 'fail':0, 'after_fail':'filterOff','saccade_motion':'filterOff'}
    plot_option=0
    

    #Congruent learning
    for cur_task in congruent_tasks:
        try:
            cur_cell_task=load_cell_task(cell_task_py_folder,cur_task,cell_ID)
        except:
            continue
    #PSTHs
    #passive trials
    PSTH_cong_passive=cur_cell_task.PSTH(window,dictFilterTrials_passive,plot_option=0,smooth_option=1) 
    PSTH_cong_passive=PSTH_cong_passive[filter_margin:-filter_margin]
    # Scatter and linear fit
    #passive
    trials_FR_passive_cong=cur_cell_task.get_mean_FR_event(dictFilterTrials_passive,'motion_onset',window_pre=150,window_post=350)
    trial_inx_passive_cong=np.arange(1,trials_FR_passive_cong.size+1)
    slope_passive_cong, inter_passive_cong = np.polyfit(trial_inx_passive_cong,trials_FR_passive_cong, 1)

      
    #Incongruent learning
    for cur_task in incongruent_tasks:
        try:
            cur_cell_task=load_cell_task(cell_task_py_folder,cur_task,cell_ID)
        except:
            continue
    #PSTHs

    #passive trials    
    PSTH_incong_passive=cur_cell_task.PSTH(window,dictFilterTrials_passive,plot_option=0,smooth_option=1) 
    PSTH_incong_passive=PSTH_incong_passive[filter_margin:-filter_margin]
    # Scatter and linear fit     
    #passive
    trials_FR_passive_incong=cur_cell_task.get_mean_FR_event(dictFilterTrials_passive,'motion_onset',window_pre=150,window_post=350)
    trial_inx_passive_incong=np.arange(1,trials_FR_passive_incong.size+1)
    slope_passive_incong, inter_passive_incong = np.polyfit(trial_inx_passive_incong,trials_FR_passive_incong, 1)  

    #Dishabituation
    #According to dbs connect between a washout block and th relevant learning block to plot them in the same row
   
    #cell db :find session, File begin and file end of current cell
    fb=cells_db.loc[cells_db['cell_ID']==cell_ID,'fb_after_stablility'].item()
    fe=cells_db.loc[cells_db['cell_ID']==cell_ID,'fe_after_stability'].item()
    session=cells_db.loc[cells_db['cell_ID']==cell_ID,'session'].item()
   
    #behaviour db: find the order of tasks in the relevant behaviour session based on information taken in cell db
    cur_behaviour_db=behaviour_db.loc[behaviour_db['behaviour_session']==session ,: ] #keep only relevant day
    cur_behaviour_db=cur_behaviour_db.loc[cur_behaviour_db['file_begin']>=fb ,: ] #keep only relevant session
    cur_behaviour_db=cur_behaviour_db.loc[cur_behaviour_db['file_end']<=fe ,: ]
    cur_task_order=cur_behaviour_db.loc[cur_behaviour_db['Task'].str.contains('CW') ,'Task' ] #remove dishabituations rows
   
   
    for cur_dis_task in dishabituation_tasks:
        try:
            cur_cell_task=load_cell_task(cell_task_py_folder,cur_dis_task,cell_ID)
        except:
            continue
    
    # Separate the trials_df for different dishabituation block
    dictFilterTrials_pre={'dir':'filterOff', 'trial_name':'v20S|v20NS', 'fail':0, 'after_fail':'filterOff','saccade_motion':'filterOff'}
    trials_df=cur_cell_task.filtTrials(dictFilterTrials_pre)
   
    trials_list=trials_df.loc[:]['filename_name'].tolist()
    trial_number_np=np.array([int(x.split('.',1)[1]) for x in trials_list ])
    first_trial_block_indices=np.where(np.diff(trial_number_np)>=80)[0]
   
    block_begin_indexes= trials_df.index[np.append(np.array(0),first_trial_block_indices+1)]
    block_end_indexes=trials_df.index[np.append(first_trial_block_indices,len(trials_df)-1)]
    
    #For each dishabituation block recorded for this cell
    for block_index,(cur_begin,cur_end) in enumerate(zip(block_begin_indexes,block_end_indexes)):
        dictFilterTrials_passive={'dir':'filterOff', 'trial_name':'v20S', 'fail':0, 'after_fail':'filterOff','saccade_motion':'filterOff','trial_begin_end':[cur_begin,cur_end]}
    
        #PSTHs
        #passive trials of current block
        cur_PSTH_washout_passive=cur_cell_task.PSTH(window,dictFilterTrials_passive,plot_option=0,smooth_option=1) 
        cur_PSTH_washout_passive=cur_PSTH_washout_passive[filter_margin:-filter_margin]
   
        plt.plot(time_course,cur_PSTH_washout_passive,color='tab:brown')
    plt.plot(time_course,PSTH_cong_passive,color='tab:blue')
    plt.plot(time_course,PSTH_incong_passive,color='tab:orange')

    plt.axvline(x=250,color='black')
    plt.xlabel('time from MO (ms)')
    plt.ylabel('FR (Hz)')
    plt.legend(['dis1','dis2','dis3','cong passive','incong passive'])
    plt.title(cell_ID)
    plt.show()
    
        
        