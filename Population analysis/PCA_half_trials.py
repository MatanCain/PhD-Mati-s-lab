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

# task_array=['8dir_saccade_100_25','8dir_active_passive_interleaved_100_25','8dir_active_passive_interleaved_100_25','8dir_active_passive_interleaved_100_25','8dir_active_passive_interleaved_100_25']
# trial_type_array=['filterOff','v20a','v20p','v20a','v20p']
# event_array=['motion_onset','motion_onset','motion_onset','cue_onset','cue_onset']
# condition_array=['saccade','saccade2','active','active2','passive','passive2','cue active','cue active2','cue passive','cue passive2']

task_array=['8dir_saccade_100_25','8dir_active_passive_interleaved_100_25','8dir_active_passive_interleaved_100_25']
trial_type_array=['filterOff','v20a','v20p']
event_array=['motion_onset','motion_onset','motion_onset',]
condition_array=['saccade','saccade2','active','active2','passive','passive2']

task_array=['8dir_saccade_100_25']
trial_type_array=['filterOff']
event_array=['motion_onset']
condition_array=['saccade','saccade2']
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

grade_crit=8
sig_pop_array=np.empty([len(condition_array),len(cell_list)])
sig_pop_array[:]=np.nan

population_flat_PSTH=[np.zeros([len(cell_list),PSTH_length*len(directions)]) for ii in condition_array]#each element of the list is for a condition. Each element is a numpy array (n_cells*PSTH_length)
for cur_np_array in population_flat_PSTH:
    cur_np_array[:]=np.nan

population_flat_PSTH_PCA=[np.zeros([len(cell_list),PSTH_PCA_length*len(directions)]) for ii in condition_array]#each element of the list is for a condition. Each element is a numpy array (n_cells*PSTH_length)
for cur_np_array in population_flat_PSTH_PCA:
    cur_np_array[:]=np.nan

average_PSTH_array=[np.zeros([len(cell_list),PSTH_length]) for ii in condition_array]#each element of the list is for a condition. Each element is a numpy array (n_cells*PSTH_length)
for cur_np_array in average_PSTH_array:
    cur_np_array[:]=np.nan
    
for cell_inx,cell_ID in enumerate(cell_list):
    array_inx=-1    
    for condition_inx in np.arange(len(task_array)):
        cur_task=task_array[condition_inx]
        cur_trial_type=trial_type_array[condition_inx]
        cur_event=event_array[condition_inx]

        cur_cell_task=load_cell_task(cell_task_py_folder,cur_task,cell_ID)
        
        if int(cur_cell_task.getGrade())>grade_crit:
            continue

        # Check whether cell is significant for the event relative to baseline
        dictFilterTrials_test = {'dir':'filterOff', 'trial_name':cur_trial_type, 'fail':0}
        try:
            check_sig_cell_cond=cur_cell_task.check_main_effect_motion_vs_baseline(cur_event,dictFilterTrials_test,crit_value=0.01)
        except:
            check_sig_cell_cond=False
        sig_pop_array[array_inx,cell_inx]=check_sig_cell_cond
        sig_pop_array[array_inx+1,cell_inx]=check_sig_cell_cond

        dictFilterTrials_PSTH = {'dir':'filterOff', 'trial_name':cur_trial_type, 'fail':0}
        
        #split indexes of trials_df into 2 random groups of same size
        trials_df=cur_cell_task.filtTrials(dictFilterTrials_PSTH)
        n_trials=len(trials_df)
        n_trials_group=int(np.floor(n_trials/2))
        group1_inxs= random.sample(range(0,n_trials),n_trials_group )
        group1_inxs.sort()
        group2_inxs=[x for x in np.arange(n_trials) if x not in group1_inxs]
        group2_inxs=group2_inxs[0:len(group1_inxs)]
        group2_inxs.sort()
        inxs=trials_df.index.to_list()
        
        window_PSTH={"timePoint":cur_event,'timeBefore':win_begin_PSTH-SMOOTH_PSTH_EDGE,'timeAfter':win_end_PSTH+SMOOTH_PSTH_EDGE}
        window_PCA={"timePoint":cur_event,'timeBefore':win_begin_PCA-SMOOTH_PSTH_EDGE,'timeAfter':win_end_PCA+SMOOTH_PSTH_EDGE}
        
        for group_inxs in [group1_inxs,group2_inxs]:
            dictFilterTrials_PSTH = {'dir':'filterOff', 'trial_name':cur_trial_type, 'fail':0}
            array_inx=array_inx+1
            # Create a flatPSTH (משורשר)
            dictFilterTrials_PSTH['trial_inxs']=[inxs[x] for x in group_inxs]
            #calculate average PSTH
            try:
                average_PSTH=cur_cell_task.PSTH(window_PSTH,dictFilterTrials_PSTH) #PSTH of cell for given condition and direction
                average_PSTH=average_PSTH[SMOOTH_PSTH_EDGE:-SMOOTH_PSTH_EDGE]
                
                average_PSTH_PCA=cur_cell_task.PSTH(window_PCA,dictFilterTrials_PSTH) #PSTH of cell for given condition and direction
                average_PSTH_PCA=average_PSTH_PCA[SMOOTH_PSTH_EDGE:-SMOOTH_PSTH_EDGE]
            
            except:
                print(cell_ID)
                continue
            average_PSTH_array[array_inx][cell_inx,:]= average_PSTH   
            cell_cond_flat_PSTH=np.array([])
            cell_cond_flat_PSTH_PCA=np.array([])                            
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
                    continue
                    
                cell_cond_flat_PSTH=np.hstack([cell_cond_flat_PSTH,cur_PSTH])
                cell_cond_flat_PSTH_PCA=np.hstack([cell_cond_flat_PSTH_PCA,cur_PSTH_PCA])

            population_flat_PSTH[array_inx][cell_inx,:]=cell_cond_flat_PSTH
            population_flat_PSTH_PCA[array_inx][cell_inx,:]=cell_cond_flat_PSTH_PCA

# if a cell is missing a direction in a given condition then the PSTH is an array of nan. ex: dir 0 in saccade for cell 7391

plt.plot(np.nanmean(average_PSTH_array[0],axis=0))
plt.plot(np.nanmean(average_PSTH_array[1],axis=0))
plt.legend(['g1','g2'])
plt.show()

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

n_cells=len(cell_list)-len(nan_inxs)
n_PCs=5#number of PCs

PCs_array=np.empty([len(condition_array),n_PCs,n_cells])
exp_var_array=np.empty([len(condition_array),n_PCs])
eigen_values_array=np.empty([len(condition_array),n_PCs])

#Finding PCS of each conditions
for cond_inx in np.arange(len(condition_array)):
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
    
# plot the explained variance and the cumulative explained variance for each condition
fig, ax = plt.subplots(1,len(condition_array))  
for cond_inx in np.arange(len(condition_array)):
    cum_var=np.cumsum(exp_var_array[cond_inx,:])
    ax[cond_inx].bar(np.arange(n_PCs), exp_var_array[cond_inx,:],alpha=0.5, align='center', label='Individual explained variance')
    ax[cond_inx].step(range(0,len(cum_var)), cum_var,where='mid',label='Cumulative explained variance')
    ax[cond_inx].set_title(condition_array[cond_inx],fontsize=6)
   # ax[cond_inx].set_ylabel('Explained variance ratio')
    #ax[cond_inx].set_xlabel('Principal component index')
    ax[0].legend(loc='best',prop = { "size": 5},handlelength=0.5)
    ax[cond_inx].set_ylim([0,1])
    ax[cond_inx].axhline(y=0.7,color='black',linestyle='--')
    ax[cond_inx].tick_params(axis='both', labelsize=6, pad=0)
fig.tight_layout()
plt.show()    


#%% Calculate the percentage of variance explained in one condition by the PCs  of other condition and ALIGNEMENT INDEX
cond_inx_data=0
cond_inx_PC=0
exp_var_array_conds=np.empty([len(condition_array),len(condition_array),n_PCs])#the explaine variance explained by PCs of other conditions

#prepare exp_var_array_conds array
for PC_rank in np.arange(n_PCs):
    for cond_inx_data in np.arange(len(condition_array)):
        for cond_inx_PC in np.arange(len(condition_array)):    
            #rextract relevant neural data
            cond_flat_PSTH=population_flat_PSTH_PCA[cond_inx_data]
            cond_flat_PSTH=np.delete(cond_flat_PSTH,nan_inxs,axis=0)
            #subtract the mean of each variable (each timepoint*conditions)
            cond_flat_PSTH2=cond_flat_PSTH-np.mean(cond_flat_PSTH,axis=0)#zero-mean
            
            if shuffle_option:
                #shuffle the 8 directions randomly for each cell
                cur_PSTH_length=np.size(cond_flat_PSTH,axis=1)/len(directions)
                #draw a number between 0 and 8 times the PSTH length for each neuron
                shift_factor=np.random.randint(len(directions), size=int(np.size(cond_flat_PSTH,axis=0)))*int(cur_PSTH_length)
                for  cur_row_inx in np.arange(np.size(cond_flat_PSTH,axis=0)):
                    cond_flat_PSTH2[cur_row_inx,:]=np.roll(cond_flat_PSTH2[cur_row_inx,:],shift_factor[cur_row_inx])
                
                
            #calculate the total variance
            tot_var=np.sum(np.var(cond_flat_PSTH2,axis=1))
            #find the relevant PC
            cur_PC=PCs_array[cond_inx_PC,PC_rank,:]
            #project on the PC
            data_proj_PC=np.matmul(np.transpose(cond_flat_PSTH2),cur_PC)
            #calculate the variance of the projected data
            proj_var=np.var(data_proj_PC)
            exp_var_array_conds[cond_inx_data,cond_inx_PC,PC_rank]=proj_var/tot_var

#prepare the figure
width=0.1
fig, ax = plt.subplots(len(condition_array)) 
for cond_inx_data in np.arange(len(condition_array)):
    multiplier=0
    for cond_inx_PC in np.arange(len(condition_array)):
        offset = width * multiplier
        ax[cond_inx_data].bar(np.arange(n_PCs)+offset, exp_var_array_conds[cond_inx_data,cond_inx_PC,:],width=width, label=condition_array[cond_inx_PC])
        ax[cond_inx_data].set_ylim([0,0.45])
        ax[0].legend(prop = { "size": 5},handlelength=0.5)
        multiplier=multiplier+1
    ax[cond_inx_data].set_title('PCs of ' +condition_array[cond_inx_data],fontsize=6)
    ax[cond_inx_data].set_ylabel('% variance',fontsize=6)
    #ax[cond_inx_data].set_xlabel('PC rank',fontsize=6)
    ax[cond_inx_data].set_xticks(np.arange(n_PCs) + 1.5*width, np.arange(n_PCs)+1)
    ax[cond_inx_data].tick_params(axis='both', labelsize=6, pad=0)
fig.tight_layout()

plt.show()        
# save_path="C:/Users/Owner/Documents/Matan/Mati's Lab/population analysis paper/figure3_PCA/"
# mdic = {"explained_variance_PCs":exp_var_array_conds}
# savemat(save_path+"explained_variance_PCs_shuffled"+ ".mat", mdic)


#Plot alignement index
alignement_index=np.sum(exp_var_array_conds,axis=2)
normalization_matrix=np.transpose(np.tile(np.sum(exp_var_array,axis=1),(len(condition_array),1))) # a 4*4 matrix where all columns are identical. Each element is the sum of variance explained by the n first PC in the condition itself
alignement_index2=alignement_index/normalization_matrix
if shuffle_option==0:
    np.fill_diagonal(alignement_index2, np.nan)
shw=plt.imshow(alignement_index2, cmap = 'OrRd' , interpolation = 'nearest')
cb=plt.colorbar(shw)
ax = plt.gca()
ax.set_xticks([])
ax.set_yticks([])
plt.title('Alignement index with '+str(n_PCs)+' PCs')
plt.show()
# save_path="C:/Users/Owner/Documents/Matan/Mati's Lab/population analysis paper/figure6_cue/"
# mdic = {"alignement_index":alignement_index2}
# savemat(save_path+"alignement_index"+ ".mat", mdic)

# projection on PC1 as a function of projection on PC2
fig, ax = plt.subplots(len(condition_array),len(condition_array)) 
FIG_EDGE=120 
PC1_list=[]
PC2_list=[]
title_list=[]
for cond_inx in np.arange(len(condition_array)):
    for cond_inx2 in np.arange(len(condition_array)):    
        #choose the PC from cond_inx2
        PC1_cond_inx2=PCs_array[cond_inx2,0,:]
        PC2_cond_inx2=PCs_array[cond_inx2,1,:]

        #choose a PSTH from cond_inx
        cond_flat_PSTH=population_flat_PSTH[cond_inx]
        cond_flat_PSTH=np.delete(cond_flat_PSTH,nan_inxs,axis=0)
                
        if shuffle_option:
            #shuffle the 8 directions randomly for each cell
            cur_PSTH_length=np.size(cond_flat_PSTH,axis=1)/len(directions)
            #draw a number between 0 and 8 times the PSTH length for each neuron
            shift_factor=np.random.randint(len(directions), size=int(np.size(cond_flat_PSTH,axis=0)))*int(cur_PSTH_length)
            for  cur_row_inx in np.arange(np.size(cond_flat_PSTH,axis=0)):
                cond_flat_PSTH[cur_row_inx,:]=np.roll(cond_flat_PSTH[cur_row_inx,:],shift_factor[cur_row_inx])
        #project the neural activity on the PC
        data_proj_PC1=np.matmul(np.transpose(cond_flat_PSTH),PC1_cond_inx2)
        data_proj_PC1_8dir=np.reshape(data_proj_PC1,(len(directions),-1))
        data_proj_PC2=np.matmul(np.transpose(cond_flat_PSTH),PC2_cond_inx2)
        data_proj_PC2_8dir=np.reshape(data_proj_PC2,(len(directions),-1))
        #plot the figure        
        ax[cond_inx,cond_inx2].plot(np.transpose(data_proj_PC1_8dir),np.transpose(data_proj_PC2_8dir),linewidth=0.75)
        ax[cond_inx,cond_inx2].set_title(condition_array[cond_inx]+' by '+condition_array[cond_inx2],fontsize=6)
        ax[cond_inx,cond_inx2].tick_params(axis='both', labelsize=6, pad=0)
        ax[cond_inx,cond_inx2].set_xlim([-FIG_EDGE,FIG_EDGE])
        ax[cond_inx,cond_inx2].set_ylim([-FIG_EDGE,FIG_EDGE])

        ax[0,0].legend(directions,loc='upper right', prop = { "size": 4 },handlelength=0.5)
        PC1_list.append(data_proj_PC1_8dir)
        PC2_list.append(data_proj_PC2_8dir)
        title_list.append(condition_array[cond_inx]+' by '+condition_array[cond_inx2])
fig.suptitle('PC1 vs PC2')
fig.tight_layout()
plt.show()        
                
        
#%% Average the alignment index for the different goupes (when dividing the trials we get two or four measurements for each couple of conditions)

sacccade_inx=0
pursuit_inx=1
suppression_inx=2
cue_pursuit_inx=3
cue_suppression_inx=4

#Alignment index of a task with itself
cur_task_inx=sacccade_inx
AI_task=np.nanmean(alignement_index2[cur_task_inx*2:cur_task_inx*2+2,cur_task_inx*2:cur_task_inx*2+2].flatten()) #extract relevant square in the alignment index matrix
print(AI_task)

#Alignment index between tasks
task_data_inx=cue_pursuit_inx
task_PC_inx=cue_pursuit_inx
AI_task=np.nanmean(alignement_index2[task_data_inx*2:task_data_inx*2+2,task_PC_inx*2:task_PC_inx*2+2].flatten()) #extract relevant square in the alignment index matrix
print(AI_task)
