# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 14:48:12 2022

@author: Owner
"""

import os
os.chdir("C:/Users/Owner/Documents/Matan/Mati's Lab/FEF learning project/Data_Analyses/Code analyzes/basic classes")
import numpy as np
import math
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300
import matplotlib.pyplot as plt
import neuron_class
from neuron_class import cell_task,filtCells,load_cell_task,smooth_data,PSTH_across_cells
from scipy.io import savemat
from scipy.stats.stats import pearsonr
import scipy.stats as stats
from scipy.ndimage import gaussian_filter1d
import seaborn as sns
from sklearn.decomposition import PCA

# General parameters
path="C:/Users/Owner/Documents/Matan/Mati's Lab/FEF learning project/Data_Analyses"
os.chdir(path)
cell_db_file='fiona_cells_db.xlsx'
#get list of cells
cell_task_py_folder="units_task_python_two_monkeys/"

#%% In this part of the script, we create population_tuning_curve_noNan a 3D array (n_conds*n_dirs*n_cells) with the tuning curve of all the cells recorded during mapping

cell_pursuit_list=os.listdir(cell_task_py_folder+'8dir_active_passive_interleaved_100_25') #list of strings
cell_saccade_list=os.listdir(cell_task_py_folder+'8dir_saccade_100_25') #list of strings
cell_list=[x for x in cell_saccade_list if x in cell_pursuit_list]
cell_list=[int(item) for item in cell_list] #list of ints
cutoff_cell=8229 #cutoff between yasmin and fiona
yasmin_cell_list=[x for x in cell_list if x>cutoff_cell]
fiona_cell_list=[x for x in cell_list if x<=cutoff_cell]
cell_list=cell_list

task_array=['8dir_saccade_100_25','8dir_active_passive_interleaved_100_25','8dir_active_passive_interleaved_100_25']
trial_type_array=['filterOff','v20a','v20p']
event_array=['motion_onset','motion_onset','motion_onset']
condition_array=['saccade','active','passive']

#Window for PSTH calculation
win_begin_PSTH=-300 
win_end_PSTH=800
PSTH_length=win_end_PSTH-win_begin_PSTH

#window for PCA analysis (we dont want to take time before the event)
win_begin_PCA=0
win_end_PCA=350
PSTH_PCA_length=win_end_PCA-win_begin_PCA

SMOOTH_PSTH_EDGE=200

win_begin_baseline=-300 #for the baseline calculation of the cell
win_end_baseline=-100

directions=[0,45,90,135,180,225,270,315]


sig_pop_array=np.empty([len(task_array),len(cell_list)])
sig_pop_array[:]=np.nan

population_flat_PSTH=[np.zeros([len(cell_list),PSTH_length*len(directions)]) for cur_task in task_array]#each element of the list is for a condition. Each element is a numpy array (n_cells*PSTH_length)
for cur_np_array in population_flat_PSTH:
    cur_np_array[:]=np.nan

population_flat_PSTH_PCA=[np.zeros([len(cell_list),PSTH_PCA_length*len(directions)]) for cur_task in task_array]#each element of the list is for a condition. Each element is a numpy array (n_cells*PSTH_length)
for cur_np_array in population_flat_PSTH_PCA:
    cur_np_array[:]=np.nan
    
for cell_inx,cell_ID in enumerate(cell_list):
 
    for condition_inx in np.arange(len(task_array)):
        cur_task=task_array[condition_inx]
        cur_trial_type=trial_type_array[condition_inx]
        cur_event=event_array[condition_inx]

        cur_cell_task=load_cell_task(cell_task_py_folder,cur_task,cell_ID)
        # if int(cur_cell_task.grade)>7:
        #     continue
        # Check whether cell is significant for the event relative to baseline
        dictFilterTrials_test = {'dir':'filterOff', 'trial_name':cur_trial_type, 'fail':0}
        try:
            check_sig_cell_cond=cur_cell_task.check_main_effect_motion_vs_baseline(cur_event,dictFilterTrials_test,crit_value=0.01)
        except:
            check_sig_cell_cond=False
        sig_pop_array[condition_inx,cell_inx]=check_sig_cell_cond

                
        # Create a flatPSTH (משורשר)
        dictFilterTrials_PSTH = {'dir':'filterOff', 'trial_name':cur_trial_type, 'fail':0}

        window_PSTH={"timePoint":cur_event,'timeBefore':win_begin_PSTH-SMOOTH_PSTH_EDGE,'timeAfter':win_end_PSTH+SMOOTH_PSTH_EDGE}
        window_PCA={"timePoint":cur_event,'timeBefore':win_begin_PCA-SMOOTH_PSTH_EDGE,'timeAfter':win_end_PCA+SMOOTH_PSTH_EDGE}
        cell_cond_flat_PSTH=np.array([])
        cell_cond_flat_PSTH_PCA=np.array([])
        
        #calculate average PSTH
        try:
            average_PSTH=cur_cell_task.PSTH(window_PSTH,dictFilterTrials_PSTH) #PSTH of cell for given condition and direction
            average_PSTH=average_PSTH[SMOOTH_PSTH_EDGE:-SMOOTH_PSTH_EDGE]
            average_PSTH_PCA=cur_cell_task.PSTH(window_PCA,dictFilterTrials_PSTH) #PSTH of cell for given condition and direction
            average_PSTH_PCA=average_PSTH_PCA[SMOOTH_PSTH_EDGE:-SMOOTH_PSTH_EDGE]
        except:
            print(cell_ID)
                        
        for cur_direction in directions:
            dictFilterTrials_PSTH['dir']=cur_direction
            cur_PSTH=np.empty([PSTH_length])
            cur_PSTH[:]=np.nan
            cur_PSTH_PCA=np.empty([PSTH_PCA_length])
            cur_PSTH_PCA[:]=np.nan
            try:
                cur_PSTH=cur_cell_task.PSTH(window_PSTH,dictFilterTrials_PSTH) #PSTH of cell for given condition and direction
                cur_PSTH=cur_PSTH[SMOOTH_PSTH_EDGE:-SMOOTH_PSTH_EDGE]-average_PSTH

                cur_PSTH_PCA=cur_cell_task.PSTH(window_PCA,dictFilterTrials_PSTH) #PSTH of cell for given condition and direction
                cur_PSTH_PCA=cur_PSTH_PCA[SMOOTH_PSTH_EDGE:-SMOOTH_PSTH_EDGE]-average_PSTH_PCA                
            except:
                print(cell_ID)
                
            cell_cond_flat_PSTH=np.hstack([cell_cond_flat_PSTH,cur_PSTH])
            cell_cond_flat_PSTH_PCA=np.hstack([cell_cond_flat_PSTH_PCA,cur_PSTH_PCA])

        population_flat_PSTH[condition_inx][cell_inx,:]=cell_cond_flat_PSTH
        population_flat_PSTH_PCA[condition_inx][cell_inx,:]=cell_cond_flat_PSTH_PCA
            
# if a cell is missing a direction in a given condition then the PSTH is an array of nan. ex: dir 0 in saccade for cell 7391            
         

#%% PCA for each condition
shuffle_option=0
#indexes of cell non significant for all the conditions
non_reactive_cells_inx=np.where(np.sum(sig_pop_array,axis=0)==0)[0]

#remove rows with nan
nan_inxs=np.array([])
nan_inxs=np.where(np.isnan(population_flat_PSTH_PCA).any(axis=0))[0]    
nan_inxs=np.hstack((nan_inxs,non_reactive_cells_inx))
nan_inxs=np.unique(nan_inxs)
nan_inxs=[int(x) for x  in nan_inxs]

#Finding PCS of each conditions
sum_trace=[]
sum_trace2=[]
for cond_inx in np.arange(len(task_array)):

    cond_flat_PSTH=population_flat_PSTH_PCA[cond_inx]
    cond_flat_PSTH=np.delete(cond_flat_PSTH,nan_inxs,axis=0)
        
    #remove the mean of each time point
    cond_flat_PSTH2=cond_flat_PSTH-np.mean(cond_flat_PSTH,axis=0)#zero-mean
    sum_trace.append(np.sum(np.trace(np.cov(cond_flat_PSTH))))
    sum_trace2.append(np.sum(np.trace(np.cov(cond_flat_PSTH2))))    

# Set position of bar on X axis 
barWidth = 0.25
br1 = np.arange(len(sum_trace)) 
br2 = [x + barWidth for x in br1] 
 
# Make the plot
plt.bar(br1, sum_trace, color ='r', width = barWidth, 
        edgecolor ='grey', label ='no sub') 
plt.bar(br2, sum_trace2, color ='g', width = barWidth, 
        edgecolor ='grey', label ='sub') 

# Adding Xticks 
plt.xlabel('Task', fontweight ='bold', fontsize = 15) 
plt.ylabel('Sum of trace', fontweight ='bold', fontsize = 15) 
plt.xticks([r + barWidth for r in range(len(sum_trace))], 
        ['sacccade', 'pursuit', 'sup'])
 
plt.legend()
plt.show() 