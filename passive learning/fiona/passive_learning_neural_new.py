# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 14:48:12 2022

@author: Owner
"""

import os
import numpy as np
import math
import matplotlib.pyplot as plt
os.chdir("C:/Users/Owner/Documents/Matan/Mati's Lab/FEF learning project/Data_Analyses/Code analyzes/basic classes")

import neuron_class
from neuron_class import cell_task,filtCells,load_cell_task,smooth_data,PSTH_across_cells
from scipy.io import savemat
import scipy.stats as stats
from scipy.ndimage import gaussian_filter1d

# General parameters
path="C:/Users/Owner/Documents/Matan/Mati's Lab/FEF learning project/Data_Analyses"
os.chdir(path)
cell_db_file='fiona_cells_db.xlsx'
#get list of cells
cell_task_py_folder="units_task_python/"
save_path="C:/Users/Owner/Documents/Matan/Mati's Lab/stage B/"


#Find cells recorded in dishabituation, motor and fixation congruent
#cell list
cell_dishabituation_list=os.listdir(cell_task_py_folder+'Dishabituation_100_25_cue') #list of strings
cell_motor_list=os.listdir(cell_task_py_folder+'Motor_learning_CCW_100_25_cue') #list of strings
cell_cong_list=os.listdir(cell_task_py_folder+'fixation_right_probes_CCW_100_25_cue') #list of strings
cell_incong_list=os.listdir(cell_task_py_folder+'fixation_wrong_probes_CCW_100_25_cue') #list of strings


cell_list=[x for x in cell_dishabituation_list if x in cell_motor_list and x in cell_cong_list and x in cell_incong_list]
cell_list=[int(item) for item in cell_list] #list of ints

#%%

# cells that increase/decrease FR in MO compared to BL in active/passive trials during washout
Window_pre_MO=250
Window_post_MO=750


Window_pre_BL=-0
Window_post_BL=250
dictFilterTrials_passive={'trial_name':'v20S', 'fail':0}
dictFilterTrials_active={'trial_name':'v20NS', 'fail':0}
decrease_cells_active=[]
increase_cells_active=[]
other_cells_active=[]
decrease_cells_passive=[]
increase_cells_passive=[]
other_cells_passive=[]
task='Motor_learning_CCW_100_25_cue'
for cell_ID in cell_list:
    cur_cell_task=load_cell_task(cell_task_py_folder,task,cell_ID)
    FR_MO=cur_cell_task.get_mean_FR_event(dictFilterTrials_active,'motion_onset',window_pre=Window_pre_MO,window_post=Window_post_MO)
    FR_BL=cur_cell_task.get_mean_FR_event(dictFilterTrials_active,'motion_onset',window_pre=Window_pre_BL,window_post=Window_post_BL)
    try:
        [stat1,pval1]=stats.wilcoxon(FR_MO,FR_BL,alternative='greater')
        [stat2,pval2]=stats.wilcoxon(FR_MO,FR_BL,alternative='less')
    except:
        print('no spike in cur cell ')
        pval1=1
        pval2=1
    
    if np.nanmean(FR_MO)>np.nanmean(FR_BL):
        increase_cells_active.append(cell_ID)
    else:
        decrease_cells_active.append(cell_ID)


    # FR_MO=cur_cell_task.get_mean_FR_event(dictFilterTrials_passive,'motion_onset',window_pre=Window_pre_MO,window_post=Window_post_MO)
    # FR_BL=cur_cell_task.get_mean_FR_event(dictFilterTrials_passive,'motion_onset',window_pre=Window_pre_BL,window_post=Window_post_BL)
    # try:
    #     [stat1,pval1]=stats.wilcoxon(FR_MO,FR_BL,alternative='greater')
    #     [stat2,pval2]=stats.wilcoxon(FR_MO,FR_BL,alternative='less')
    # except:
    #     print('no spike in cur cell ')
    #     pval1=1
    #     pval2=1

    # if np.nanmean(FR_MO)>np.nanmean(FR_BL):
    #     increase_cells_passive.append(cell_ID)
    # else:
    #     decrease_cells_passive.append(cell_ID)

#%%
task_array=['Dishabituation_100_25_cue','fixation_right_probes_CCW_100_25_cue','fixation_wrong_probes_CCW_100_25_cue']
trial_type='v20S'
event='motion_onset'

#window PSTH
win_before=-400
win_after=1000
PSTH_length=win_after-win_before
fail=0

PSTH_type='merged_dir' 

plot_option=1

cur_cell_list=increase_cells_active

PSTH_average_list=[]
for psth_inx in np.arange(len(task_array)):
    task=task_array[psth_inx]
    window={"timePoint": event,"timeBefore": win_before,"timeAfter":win_after}
    dictFilterTrials = {'dir':'filterOff', 'trial_name':trial_type, 'fail':fail, 'after_fail':'filterOff','saccade_motion':'filterOff'}    
    PSTH_array_mean,PSTH_array=PSTH_across_cells(cur_cell_list,task,event,trial_type,window,dictFilterTrials,PSTH_type,PD_dict=[],plot_option=0)
    PSTH_average_list.append(PSTH_array_mean)
    plt.plot()

win_before=-400
win_after=1000
x_axis=np.arange(win_before,win_after)
for PSTH in PSTH_average_list:
    plt.plot(x_axis,PSTH)
plt.legend(['dishabituation','cong','incong'])    
plt.axvline(x=0,color='red')
plt.axvline(x=250,color='red')
plt.ylim([0, 25])
plt.xlabel('Time from '+event+' (ms)')
plt.ylabel('FR (Hz)')
plt.show()


