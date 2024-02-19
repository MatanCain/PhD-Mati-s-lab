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
cell_dishabituation_list=[]
dishabituation_tasks=['Dishabituation_100_25_cue','Dishabituation',]
#motor_tasks=['Motor_learning_CW_100_25_cue','Motor_learning_CCW_100_25_cue']
for dishabituation_task in dishabituation_tasks:
    cell_dishabituation_list=cell_dishabituation_list+os.listdir(cell_task_py_folder+dishabituation_task) #list of strings

cell_motor_list=[]
motor_tasks=['Motor_learning_CW_100_25_cue','Motor_learning_CCW_100_25_cue','Motor_learning_CW','Motor_learning_CCW']
#motor_tasks=['Motor_learning_CW_100_25_cue','Motor_learning_CCW_100_25_cue']
for motor_task in motor_tasks:
    cell_motor_list=cell_motor_list+os.listdir(cell_task_py_folder+motor_task) #list of strings
    
cell_cong_list=[]
cong_tasks=['fixation_right_probes_CW_100_25_cue','fixation_right_probes_CCW_100_25_cue','fixation_right_probes_CW','fixation_right_probes_CCW']
#cong_tasks=['fixation_right_probes_CW_100_25_cue','fixation_right_probes_CCW_100_25_cue']
for cong_task in cong_tasks:
    cell_cong_list=cell_cong_list+os.listdir(cell_task_py_folder+cong_task) #list of strings
    
cell_incong_list=[]
#incong_tasks=['fixation_wrong_probes_CW_100_25_cue','fixation_wrong_probes_CCW_100_25_cue']
incong_tasks=['fixation_wrong_probes_CW_100_25_cue','fixation_wrong_probes_CCW_100_25_cue','fixation_wrong_probes_CW','fixation_wrong_probes_CCW']
for incong_task in incong_tasks:
    cell_incong_list=cell_incong_list+os.listdir(cell_task_py_folder+incong_task) #list of strings

cell_list=[x for x in cell_dishabituation_list if x in cell_motor_list and x in cell_cong_list and x in cell_incong_list]
cell_list=[int(item) for item in cell_list] #list of ints

#%%

# cells that increase/decrease FR in MO compared to BL in active/passive trials during washout
Window_pre_MO=250
Window_post_MO=500

#BL is th eperiod before change in direction
Window_pre_BL=0
Window_post_BL=250
dictFilterTrials_passive={'trial_name':'v20S', 'fail':0}
dictFilterTrials_active={'trial_name':'v20NS', 'fail':0}
learned_cells=[]
base_cells=[]
task_list=motor_tasks
for cell_ID in cell_list:
    for task in task_list:
        try:    
            cur_cell_task=load_cell_task(cell_task_py_folder,task,cell_ID)
        except:
            continue
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
            learned_cells.append(cell_ID)
        else:
            base_cells.append(cell_ID)



#%%
trial_type='v20NS'
event='motion_onset'

#window PSTH
win_before=-300
win_after=800
PSTH_length=win_after-win_before
fail=0

PSTH_type='merged_dir' 
plot_option=1
cur_cell_list=base_cells

Window_pre_MO=250
Window_post_MO=500
Window_pre_BL=0
Window_post_BL=250

task=incong_tasks
window={"timePoint": event,"timeBefore": win_before,"timeAfter":win_after}
dictFilterTrials = {'dir':'filterOff', 'trial_name':trial_type, 'fail':fail, 'after_fail':'filterOff','saccade_motion':'filterOff'}    
PSTH_array_mean_incong,PSTH_incong_array=PSTH_across_cells(cur_cell_list,task,event,trial_type,window,dictFilterTrials,PSTH_type,PD_dict=[],plot_option=0)
PSTH_incong_BL=np.nanmean(PSTH_incong_array[:,-250-win_before:0-win_before],axis=1)
#remove BL from PSTH
PSTH_incong_array=PSTH_incong_array-np.transpose(np.matlib.repmat(PSTH_incong_BL, win_after-win_before,1))


task=cong_tasks
window={"timePoint": event,"timeBefore": win_before,"timeAfter":win_after}
dictFilterTrials = {'dir':'filterOff', 'trial_name':trial_type, 'fail':fail, 'after_fail':'filterOff','saccade_motion':'filterOff'}    
PSTH_array_mean_cong,PSTH_cong_array=PSTH_across_cells(cur_cell_list,task,event,trial_type,window,dictFilterTrials,PSTH_type,PD_dict=[],plot_option=0)
PSTH_cong_BL=np.nanmean(PSTH_cong_array[:,-250-win_before:0-win_before],axis=1)
#remove BL from PSTH
PSTH_cong_array=PSTH_cong_array-np.transpose(np.matlib.repmat(PSTH_cong_BL, win_after-win_before,1))

task=motor_tasks
window={"timePoint": event,"timeBefore": win_before,"timeAfter":win_after}
dictFilterTrials = {'dir':'filterOff', 'trial_name':trial_type, 'fail':fail, 'after_fail':'filterOff','saccade_motion':'filterOff'}    
PSTH_array_mean_motor,PSTH_motor_array=PSTH_across_cells(cur_cell_list,task,event,trial_type,window,dictFilterTrials,PSTH_type,PD_dict=[],plot_option=0)
PSTH_motor_BL=np.nanmean(PSTH_motor_array[:,-250-win_before:0-win_before],axis=1)
#remove BL from PSTH
PSTH_motor_array=PSTH_motor_array-np.transpose(np.matlib.repmat(PSTH_motor_BL, win_after-win_before,1))

task=dishabituation_tasks
window={"timePoint": event,"timeBefore": win_before,"timeAfter":win_after}
dictFilterTrials = {'dir':'filterOff', 'trial_name':trial_type, 'fail':fail, 'after_fail':'filterOff','saccade_motion':'filterOff'}    
PSTH_array_mean_dis,PSTH_dis_array=PSTH_across_cells(cur_cell_list,task,event,trial_type,window,dictFilterTrials,PSTH_type,PD_dict=[],plot_option=0)
PSTH_dis_BL=np.nanmean(PSTH_dis_array[:,-250-win_before:0-win_before],axis=1)
#remove BL from PSTH
PSTH_dis_array=PSTH_dis_array-np.transpose(np.matlib.repmat(PSTH_dis_BL, win_after-win_before,1))

#PSTH
win_before=-300
win_after=800
x_axis=np.arange(win_before,win_after)
plt.plot(x_axis,np.nanmean(PSTH_cong_array,axis=0))
plt.plot(x_axis,np.nanmean(PSTH_incong_array,axis=0))
plt.plot(x_axis,np.nanmean(PSTH_dis_array,axis=0))
plt.plot(x_axis,np.nanmean(PSTH_motor_array,axis=0))
plt.legend(['cong','incong','washout','motor'])    
plt.axvline(x=0,color='red')
plt.axvline(x=250,color='red')
plt.ylim([-5, 10])
plt.xlabel('Time from '+event+' (ms)')
plt.ylabel('FR (Hz)')
plt.show()

#Scatter plot
EDGE=40
x=np.arange(-20,EDGE)
cong_scatter=np.nanmean(PSTH_cong_array[:,100-win_before:300-win_before],axis=1)
incong_scatter=np.nanmean(PSTH_incong_array[:,100-win_before:300-win_before],axis=1)
res=stats.wilcoxon(cong_scatter,incong_scatter)
pval=res.pvalue
plt.scatter(cong_scatter,incong_scatter)
plt.plot(x,x,color='red')
plt.xlabel('cong')
plt.ylabel('incong')
plt.title('scatter cong vs incong (100-300), pval:'+str(round(pval,2)))
#plt.xlim([-EDGE,EDGE])
#plt.ylim([-EDGE,EDGE])
plt.show()


# save_path="C:/Users/Owner/Documents/Matan/Mati's Lab/stage B/"
# mdic = {"PSTH_arrays":[PSTH_incong_array,PSTH_cong_array,PSTH_motor_array,PSTH_dis_array]}
# savemat(save_path+"PSTH_base_cells_active"+ ".mat", mdic)
