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
import random
from scipy.stats import kruskal

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
#%%

#check if cell is significant for cue before motion onset and if so remove it because correlation can be caused by cue
cue_cells=[]
directions=[0,45,90,135,180,225,270,315]
cur_task='8dir_active_passive_interleaved_100_25'
dictFilterTrials_TC = {'dir':'filterOff', 'trial_name':'v20a|v20p', 'fail':0}

for cell_ID in cell_list:
    cur_cell_task=load_cell_task(cell_task_py_folder,cur_task,cell_ID)
    try:
            #tuning curve of the cell within the bin
            tuning_curve_list=[]
            for cur_dir_inx,cur_direction in enumerate(directions):
                dictFilterTrials_TC['dir']=cur_direction
            
                try:
                    FR_cell_bin_cond=cur_cell_task.get_mean_FR_event(dictFilterTrials_TC,'motion_onset',window_pre=-400,window_post=0)
                    tuning_curve_list.append(FR_cell_bin_cond)
                except:
                    continue            
            #check if tuning is significant in current bin:
            try:
                test_result=kruskal(*tuning_curve_list)
                sig_cue=test_result[1]<0.01
            except:
                sig_tuning_bin=0    
    except:
        continue
    if sig_cue:
        cue_cells.append(cell_ID)

cell_list=[x for x in cell_list]        
        

task_array=['8dir_saccade_100_25','8dir_active_passive_interleaved_100_25','8dir_active_passive_interleaved_100_25','8dir_active_passive_interleaved_100_25']
trial_type_array=['filterOff','v20a','v20p','v20a|v20p']
event_array=['motion_onset','motion_onset','motion_onset','cue_onset']
condition_array=['saccade','active','passive','cue']

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

population_flat_PSTH=[np.zeros([len(cell_list),PSTH_length*len(directions)]) for ii in np.arange(len(task_array))]#each element of the list is for a condition. Each element is a numpy array (n_cells*PSTH_length)
for cur_np_array in population_flat_PSTH:
    cur_np_array[:]=np.nan

population_flat_PSTH_PCA=[np.zeros([len(cell_list),PSTH_PCA_length*len(directions)]) for ii in np.arange(len(task_array))]#each element of the list is for a condition. Each element is a numpy array (n_cells*PSTH_length)
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
#indexes of cell non significant for all the conditions
non_reactive_cells_inx=np.where(np.sum(sig_pop_array,axis=0)==0)[0]

#remove rows with nan
nan_inxs=np.array([])
nan_inxs=np.where(np.isnan(population_flat_PSTH_PCA).any(axis=0))[0]    
nan_inxs=np.hstack((nan_inxs,non_reactive_cells_inx))
nan_inxs=np.unique(nan_inxs)
nan_inxs=[int(x) for x  in nan_inxs]

#remove cells signififant for late cue cells
# late_cue_cells_inxs=[cell_inx for cell_inx,cell_ID in enumerate(cell_list) if (cell_ID) in cue_cells ]
# nan_inxs=np.hstack((nan_inxs,non_reactive_cells_inx,late_cue_cells_inxs)) 
# nan_inxs=np.unique(nan_inxs)
# nan_inxs=[int(x) for x  in nan_inxs]

n_cells=len(cell_list)-len(nan_inxs)
n_PCs=5#number of PCs

PCs_array=np.empty([len(task_array),n_PCs,n_cells])
exp_var_array=np.empty([len(task_array),n_PCs])
eigen_values_array=np.empty([len(task_array),n_PCs])

#Finding PCS of each conditions
for cond_inx in np.arange(len(task_array)):
    cond_flat_PSTH=population_flat_PSTH_PCA[cond_inx]
    cond_flat_PSTH=np.delete(cond_flat_PSTH,nan_inxs,axis=0)
        
    #remove the mean of each time point
    cond_flat_PSTH=cond_flat_PSTH-np.mean(cond_flat_PSTH,axis=0)#zero-mean
    pca = PCA(n_components=n_PCs)
    pca.fit(np.transpose(cond_flat_PSTH))
    exp_var_cond=pca.explained_variance_ratio_
    eigen_values=pca.explained_variance_
    PCs_cond=pca.components_
    PCs_array[cond_inx,:,:]=PCs_cond
    exp_var_array[cond_inx,:]=exp_var_cond
    eigen_values_array[cond_inx,:]=eigen_values
    



#%% Calculate the percentage of variance explained in one condition by the PCs  of other condition and ALIGNEMENT INDEX

exp_var_array_conds=np.empty([len(task_array),len(task_array)])#the explaine variance explained by PCs of other conditions

#prepare exp_var_array_conds array
#for PC_rank in np.arange(n_PCs):
PC_rank=0
for cond_inx_data in np.arange(len(task_array)):
    for cond_inx_PC in np.arange(len(task_array)):  
        cond_inx_data=0
        cond_inx_PC=0
        #rextract relevant neural data
        cond_flat_PSTH=population_flat_PSTH_PCA[cond_inx_data]
        cond_flat_PSTH=np.delete(cond_flat_PSTH,nan_inxs,axis=0)
        #subtract the mean of each variable (each timepoint*conditions)
        cond_flat_PSTH2=cond_flat_PSTH-np.mean(cond_flat_PSTH,axis=0)#zero-mean
        #calculate the total variance
        tot_var=np.sum(np.var(cond_flat_PSTH2,axis=1))
        #find the relevant PC
        cur_PC=PCs_array[cond_inx_PC,PC_rank,:]
        #project on the PC
        data_proj_PC=np.matmul(np.transpose(cond_flat_PSTH2),cur_PC)
        #calculate the variance of the projected data
        proj_var=np.var(data_proj_PC)
        exp_var_array_conds[cond_inx_data,cond_inx_PC]=proj_var/tot_var
    
# save_path="C:/Users/Owner/Documents/Matan/Mati's Lab/population analysis paper/figure3_PCA/"
# mdic = {"explained_variance_PCs":exp_var_array_conds}
# savemat(save_path+"explained_variance_PCs_shuffled"+ ".mat", mdic)


#Plot alignement index
alignement_index=np.sum(exp_var_array_conds,axis=1)
normalization_matrix=np.transpose(np.tile(np.sum(exp_var_array,axis=1),(len(task_array),1))) # a 4*4 matrix where all columns are identical. Each element is the sum of variance explained by the n first PC in the condition itself
alignement_index2=alignement_index/normalization_matrix


#%% bootstrap ALIGNEMENT INDEX
cond_1_inx_data=2 #put larger value here
cond_2_inx_data=1

cond_inx_PC=3 #index of the condition of reference. The PCs we algin the activity to
exp_var_array_conds_cond1=np.empty([n_PCs])#the explaine variance explained by PCs of other conditions
exp_var_array_conds_cond1[:]=np.nan

exp_var_array_conds_cond2=np.empty([n_PCs])#the explaine variance explained by PCs of other conditions
exp_var_array_conds_cond2[:]=np.nan

population_flat_PSTH_PCA2=[np.delete(x,nan_inxs,axis=0) for x in population_flat_PSTH_PCA]

#prepare exp_var_array_conds array
#num of cells 
alignement_index_array_cond1=[]
alignement_index_array_cond2=[]
n_iterations=1000

#create the distribution of differences in the alignement index
for iteration in np.arange(n_iterations):   
    n_half_cells=int(np.floor(np.size(population_flat_PSTH_PCA2[cond_inx_data],0)/2))
    #sample half of the cells that switch between the two conditions
    switch_inxs=np.sort(random.sample(range(1,np.size(population_flat_PSTH_PCA2[cond_inx_data],0) ), n_half_cells))
    not_switch_inxs=[x for x in np.arange(np.size(population_flat_PSTH_PCA2[cond_inx_data],0)) if x not in switch_inxs]
    cond1_activity_temp=np.vstack((population_flat_PSTH_PCA2[cond_2_inx_data][switch_inxs],population_flat_PSTH_PCA2[cond_1_inx_data][not_switch_inxs]))
    cond2_activity_temp=np.vstack((population_flat_PSTH_PCA2[cond_1_inx_data][switch_inxs],population_flat_PSTH_PCA2[cond_2_inx_data][not_switch_inxs]))
    
    inxs_stacked=np.hstack((switch_inxs,not_switch_inxs))
    sorted_cond1_activity_temp = cond1_activity_temp[np.argsort(inxs_stacked),:]
    sorted_cond2_activity_temp = cond2_activity_temp[np.argsort(inxs_stacked),:]
    
    #subtract the mean of each variable (each timepoint*conditions)
    sorted_cond1_activity_temp=sorted_cond1_activity_temp-np.mean(sorted_cond1_activity_temp,axis=0)#zero-mean
    sorted_scond2_activity_temp=sorted_cond2_activity_temp-np.mean(sorted_cond2_activity_temp,axis=0)#zero-mean

    #calculate the total variance
    tot_var_cond1=np.sum(np.var(sorted_cond1_activity_temp,axis=1))
    tot_var_cond2=np.sum(np.var(sorted_cond2_activity_temp,axis=1))
     
    for PC_rank in np.arange(n_PCs):
        #find the relevant PC
        cur_PC=PCs_array[cond_inx_PC,PC_rank,:]
        #project on the PC
        data_proj_PC_cond1=np.matmul(np.transpose(sorted_cond1_activity_temp),cur_PC)
        data_proj_PC_cond2=np.matmul(np.transpose(sorted_cond2_activity_temp),cur_PC)

        #calculate the variance of the projected data
        proj_var_cond1=np.var(data_proj_PC_cond1)
        proj_var_cond2=np.var(data_proj_PC_cond2)

        exp_var_array_conds_cond1[PC_rank]=proj_var_cond1/tot_var_cond1
        exp_var_array_conds_cond2[PC_rank]=proj_var_cond2/tot_var_cond2
            
    #Plot alignement index
    alignement_index_array_cond1.append(np.sum(exp_var_array_conds_cond1,axis=0))
    alignement_index_array_cond2.append(np.sum(exp_var_array_conds_cond2,axis=0))

#check real difference of alignement index to cue between cond1 and cond2    
    if iteration==n_iterations-1:
        cond1_activity=np.vstack((population_flat_PSTH_PCA2[cond_1_inx_data][switch_inxs],population_flat_PSTH_PCA2[cond_1_inx_data][not_switch_inxs]))
        cond2_activity=np.vstack((population_flat_PSTH_PCA2[cond_2_inx_data][switch_inxs],population_flat_PSTH_PCA2[cond_2_inx_data][not_switch_inxs]))
        
        inxs_stacked=np.hstack((switch_inxs,not_switch_inxs))
        sorted_cond1_activity = cond1_activity[np.argsort(inxs_stacked),:]
        sorted_cond2_activity= cond2_activity[np.argsort(inxs_stacked),:]
        
        #subtract the mean of each variable (each timepoint*conditions)
        sorted_cond1_activity_norm=sorted_cond1_activity-np.mean(sorted_cond1_activity,axis=0)#zero-mean
        sorted_scond2_activity_norm=sorted_cond2_activity-np.mean(sorted_cond2_activity,axis=0)#zero-mean

        #calculate the total variance
        tot_var_cond1=np.sum(np.var(sorted_cond1_activity_norm,axis=1))
        tot_var_cond2=np.sum(np.var(sorted_scond2_activity_norm,axis=1))
         
        for PC_rank in np.arange(n_PCs):
            #find the relevant PC
            cur_PC=PCs_array[cond_inx_PC,PC_rank,:]
            #project on the PC
            data_proj_PC_cond1=np.matmul(np.transpose(sorted_cond1_activity_norm),cur_PC)
            data_proj_PC_cond2=np.matmul(np.transpose(sorted_scond2_activity_norm),cur_PC)

            #calculate the variance of the projected data
            proj_var_cond1=np.var(data_proj_PC_cond1)
            proj_var_cond2=np.var(data_proj_PC_cond2)

            exp_var_array_conds_cond1[PC_rank]=proj_var_cond1/tot_var_cond1
            exp_var_array_conds_cond2[PC_rank]=proj_var_cond2/tot_var_cond2
                
        #Plot alignement index
        alignement_index_cond1=np.sum(exp_var_array_conds_cond1,axis=0)
        alignement_index_cond2=np.sum(exp_var_array_conds_cond2,axis=0)

alignement_index_diffs_distribution=np.sort(np.abs((np.array(alignement_index_array_cond1)-np.array(alignement_index_array_cond2))))

#find critical value:
crit_inx=int(0.95*n_iterations)
crit_value=alignement_index_diffs_distribution[crit_inx]
alignement_index_diff=alignement_index_cond1-alignement_index_cond2
if alignement_index_diff>crit_value:
    print('Bootstrap is significant')
else:
    print('Bootstrap is not significant')
    
p_value=np.sum(alignement_index_diff<alignement_index_diffs_distribution)/n_iterations

#%%
trials_df=cur_cell_task.getTrials_df()