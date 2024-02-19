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

# task_array=['8dir_saccade_100_25','8dir_active_passive_interleaved_100_25','8dir_active_passive_interleaved_100_25','8dir_active_passive_interleaved_100_25']
# trial_type_array=['filterOff','v20a','v20p','v20a|v20p']
# event_array=['motion_onset','motion_onset','motion_onset','cue_onset']
# condition_array=['saccade','active','passive','cue']


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

population_flat_PSTH_PCA=[np.zeros([len(cell_list),PSTH_PCA_length]) for cur_task in task_array]#each element of the list is for a condition. Each element is a numpy array (n_cells*PSTH_length)
for cur_np_array in population_flat_PSTH_PCA:
    cur_np_array[:]=np.nan
    
for cell_inx,cell_ID in enumerate(cell_list):
 
    for condition_inx in np.arange(len(task_array)):
        cur_task=task_array[condition_inx]
        cur_trial_type=trial_type_array[condition_inx]
        cur_event=event_array[condition_inx]

        cur_cell_task=load_cell_task(cell_task_py_folder,cur_task,cell_ID)

        # Check whether cell is significant for the event relative to baseline
        dictFilterTrials_test = {'dir':'filterOff', 'trial_name':cur_trial_type, 'fail':0}
        # try:
        #     check_sig_cell_cond=cur_cell_task.check_main_effect_motion_vs_baseline(cur_event,dictFilterTrials_test,crit_value=0.01)
        # except:
        #     check_sig_cell_cond=False
        # sig_pop_array[condition_inx,cell_inx]=check_sig_cell_cond

        trial_df=cur_cell_task.filtTrials(dictFilterTrials_test)        
        if len(trial_df)<40:
            continue
                
        # Create a flatPSTH (משורשר)
        dictFilterTrials_PSTH = {'dir':'filterOff', 'trial_name':cur_trial_type, 'fail':0}

        window_PCA={"timePoint":cur_event,'timeBefore':win_begin_PCA-SMOOTH_PSTH_EDGE,'timeAfter':win_end_PCA+SMOOTH_PSTH_EDGE}
        cell_cond_flat_PSTH_PCA=np.array([])
        
        #calculate average PSTH
        try:

            average_PSTH_PCA=cur_cell_task.PSTH(window_PCA,dictFilterTrials_PSTH) #PSTH of cell for given condition and direction
            average_PSTH_PCA=average_PSTH_PCA[SMOOTH_PSTH_EDGE:-SMOOTH_PSTH_EDGE]
        except:
            print(cell_ID)
                        
        population_flat_PSTH_PCA[condition_inx][cell_inx,:]=average_PSTH_PCA-np.nanmean(average_PSTH_PCA)
            
# if a cell is missing a direction in a given condition then the PSTH is an array of nan. ex: dir 0 in saccade for cell 7391            
         
#%%
#indexes of cell non significant for all the conditions
#non_reactive_cells_inx=np.where(np.sum(sig_pop_array,axis=0)==0)[0]


#remove rows with nan
nan_inxs=np.array([])
nan_inxs=np.where(np.isnan(population_flat_PSTH_PCA).any(axis=0))[0]    
#nan_inxs=np.hstack((nan_inxs,non_reactive_cells_inx))
nan_inxs=np.unique(nan_inxs)
nan_inxs=[int(x) for x  in nan_inxs]

n_cells=len(cell_list)-len(nan_inxs)


saccade_psth=population_flat_PSTH_PCA[0]
saccade_psth=np.delete(saccade_psth,nan_inxs,axis=0)

pursuit_psth=population_flat_PSTH_PCA[1]
pursuit_psth=np.delete(pursuit_psth,nan_inxs,axis=0)

suppression_psth=population_flat_PSTH_PCA[2]
suppression_psth=np.delete(suppression_psth,nan_inxs,axis=0)

suppression_pursuit_array=[]
saccade_pursuit_array=[]
parameter='correlation'
for row_inx in np.arange(np.size(suppression_psth,0)) :
    if parameter=='correlation':
        r_saccade_pursuit=stats.pearsonr(saccade_psth[row_inx,:],pursuit_psth[row_inx,:])
        r_saccade_pursuit=r_saccade_pursuit[0]

        r_suppression_pursuit=stats.pearsonr(suppression_psth[row_inx,:],pursuit_psth[row_inx,:])
        r_suppression_pursuit=r_suppression_pursuit[0]
    elif parameter=='covariance':
        r_saccade_pursuit=np.cov(saccade_psth[row_inx,:],pursuit_psth[row_inx,:])
        r_saccade_pursuit=r_saccade_pursuit[0,1]

        r_suppression_pursuit=np.cov(suppression_psth[row_inx,:],pursuit_psth[row_inx,:])
        r_suppression_pursuit=r_suppression_pursuit[0,1]
        
    saccade_pursuit_array.append(r_saccade_pursuit)
    suppression_pursuit_array.append(r_suppression_pursuit)


saccade_pursuit_mean=np.nanmean(saccade_pursuit_array)
saccade_pursuit_sem=np.nanstd(saccade_pursuit_array)/(len(saccade_pursuit_array)**0.5)
suppression_pursuit_mean=np.nanmean(suppression_pursuit_array)
suppression_pursuit_sem=np.nanstd(suppression_pursuit_array)/(len(suppression_pursuit_array)**0.5)

plt.hist(saccade_pursuit_array,bins=np.linspace(-1, 1, 21),alpha=0.5)
plt.hist(suppression_pursuit_array,bins=np.linspace(-1, 1, 21),alpha=0.5)
plt.legend(['saccade','suppression'])
plt.show()

print('Average corr for saccade and pursuit:'+str(round(saccade_pursuit_mean,3))+' SEM:'+str(round(saccade_pursuit_sem,3)))
print('Average corr for suppression and pursuit:'+str(round(suppression_pursuit_mean,3))+' SEM:'+str(round(suppression_pursuit_sem,3)))

plt.show()
    
save_path="C:/Users/Owner/Documents/Matan/Mati's Lab/population analysis paper/figureR5_omega_time/"
mdic = {"psth_average_time_correlation":[saccade_pursuit_array,suppression_pursuit_array]}
savemat(save_path+"psth_average_time_correlation"+ ".mat", mdic)
