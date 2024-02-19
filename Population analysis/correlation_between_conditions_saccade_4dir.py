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
from scipy.stats import kruskal


# General parameters
path="C:/Users/Owner/Documents/Matan/Mati's Lab/FEF learning project/Data_Analyses"
os.chdir(path)
cell_db_file='fiona_cells_db.xlsx'
#get list of cells
cell_task_py_folder="units_task_python_two_monkeys/"


#%% Choose list of cells

cell_list=os.listdir(cell_task_py_folder+'4dir_saccade_cue') #list of strings
cell_list=[int(item) for item in cell_list] #list of ints
cutoff_cell=8229 #cutoff between yasmin and fiona
yasmin_cell_list=[x for x in cell_list if x>cutoff_cell]
fiona_cell_list=[x for x in cell_list if x<=cutoff_cell]
cell_list=cell_list

#%% Heatmap of avergae correlations between tuning curves of bins (100 ms) of cells
#Each element in the heatmap is the average of the correlation between tuning in those bins correlation across cells significant for tuning in those bins 
directions=[0,90,180,270]

task_array=['4dir_saccade_cue','4dir_saccade_cue']
trial_type_array=['filterOff','filterOff']
event_array=['motion_onset','cue_onset']
condition_array=['saccade','cue early']
bin_lenth=50 #ms
window_begin=0
window_end=500
bin_begins=np.arange(window_begin,window_end,bin_lenth)


pop_sig_bin_conditions=np.empty([len(cell_list),np.size(bin_begins),len(task_array)])
pop_sig_bin_conditions[:]=np.nan

pop_tuning_bin_conditions=np.empty([len(cell_list),np.size(bin_begins),len(directions),len(task_array)])
pop_tuning_bin_conditions[:]=np.nan

#create a list with the indexes couples of condition
cond_inxs_list = list(range(0,len(task_array)))
#list with all the possible couple of condition including with themselves
cond_couples_list1 = [(a, b) for idx, a in enumerate(cond_inxs_list) for b in cond_inxs_list[idx:]]
cond_couples_list2= [(b, a) for idx, a in enumerate(cond_inxs_list) for b in cond_inxs_list[idx:]]
cond_couples_list=cond_couples_list1+cond_couples_list2
cond_couples_list = list(dict.fromkeys(cond_couples_list))

population_correlation_matrix=np.empty([len(cond_couples_list),len(cell_list),np.size(bin_begins),np.size(bin_begins)])
population_correlation_matrix[:]=np.nan

for cell_inx,cell_ID in enumerate(cell_list):
    print(cell_ID)
    cell_correlation_matrix = [[[] for x in range(len(task_array))] for y in range(len(task_array))] 
    cell_sig_bin_conditions=np.empty([np.size(bin_begins),len(task_array)])
    cell_sig_bin_conditions[:]=np.nan
    cell_tuning_bin_conditions=np.empty([np.size(bin_begins),len(directions),len(task_array)])
    cell_tuning_bin_conditions[:]=np.nan
    
    for condition_inx in np.arange(len(task_array)):
        cur_task=task_array[condition_inx]
        cur_cell_task=load_cell_task(cell_task_py_folder,cur_task,cell_ID)

        cur_trial_type=trial_type_array[condition_inx]
        cur_event=event_array[condition_inx]

        
        for bin_inx,cur_bin_begin in enumerate(bin_begins):
            cur_bin_end=cur_bin_begin+bin_lenth

            #tuning curve of the cell within the bin
            dictFilterTrials_TC = {'dir':'filterOff', 'trial_name':cur_trial_type, 'fail':0}
            tuning_curve_list=[]
            for cur_dir_inx,cur_direction in enumerate(directions):
                dictFilterTrials_TC['dir']=cur_direction
            
                try:
                    FR_cell_bin_cond=cur_cell_task.get_mean_FR_event(dictFilterTrials_TC,cur_event,window_pre=cur_bin_begin,window_post=cur_bin_end)
                    tuning_curve_list.append(FR_cell_bin_cond)
                except:
                    continue
                cell_tuning_bin_conditions[bin_inx,cur_dir_inx,condition_inx]=  np.mean(FR_cell_bin_cond) 
            
            #check if tuning is significant in current bin:
            try:
                test_result=kruskal(*tuning_curve_list)
                sig_tuning_bin=test_result[1]<0.05
            except:
                sig_tuning_bin=0
            cell_sig_bin_conditions[bin_inx,condition_inx]=sig_tuning_bin
            
            
    #For each cell we create a matrix of n_conditions * n_conditions. In each element we have a matrix n_bins * n_bins where each element is the correlation between the tuning of the 2 conditions.
    #If the tuning of the cell is not significant in both of the condition we will keep a nan
    for cur_couple_inx,cur_couple in enumerate(cond_couples_list):
        cond_inx1=cur_couple[0]
        cond_inx2=cur_couple[1]
        cell_correlation=np.corrcoef(cell_tuning_bin_conditions[:,:,cond_inx1],cell_tuning_bin_conditions[:,:,cond_inx2])
        #corrcoeff keeps also correlations within the row of the matrix. We remove them:
        cell_correlation2=cell_correlation[0:np.size(bin_begins),np.size(bin_begins):2*np.size(bin_begins)]
        cell_correlation_matrix[cond_inx1][cond_inx2]=cell_correlation2
        
        #Put nan at non significant bins in one of the two relevant bins
        #cond1:
        mask_matrix=np.empty([np.size(bin_begins),np.size(bin_begins)])  
        mask_matrix[:]=np.nan
        for bin_inx1,cur_bin_begin in enumerate(bin_begins):
            for bin_inx2,cur_bin_begin in enumerate(bin_begins):
                mask_matrix[bin_inx1,bin_inx2]=np.logical_and(cell_sig_bin_conditions[bin_inx1,cond_inx1],cell_sig_bin_conditions[bin_inx2,cond_inx2])
        cell_correlation_matrix[cond_inx1][cond_inx2][np.logical_not(mask_matrix)]=np.nan
        population_correlation_matrix[cur_couple_inx,cell_inx,:,:]=cell_correlation_matrix[cond_inx1][cond_inx2]


#%% subplot heatmaps of correlations
population_correlation_matrix_masked=population_correlation_matrix

fig, ax = plt.subplots(len(task_array),len(task_array))
COLOR_BAR_RANGE=1
for cur_couple_inx,cur_couple in enumerate(cond_couples_list):
    cur_correlation_matrix=np.nanmean(population_correlation_matrix_masked[cur_couple_inx,:,:,:],axis=0)
    cond1_inx=cond_couples_list[cur_couple_inx][0]
    cond2_inx=cond_couples_list[cur_couple_inx][1]
    shw=ax[cond1_inx][cond2_inx].imshow(cur_correlation_matrix , cmap = 'Spectral' , interpolation = 'nearest', vmin=-COLOR_BAR_RANGE, vmax=COLOR_BAR_RANGE)
    ax[cond1_inx,cond2_inx].tick_params(axis='both', labelsize=4, pad=0)
    ax[cond1_inx][cond2_inx].set_title(condition_array[cond1_inx]+'-'+condition_array[cond2_inx],fontsize=8)
    cb=plt.colorbar(shw, ax=ax[cond1_inx, cond2_inx])
    cb.ax.tick_params(labelsize=6)
fig.tight_layout()
plt.show()
   
#subplot of number of significant pair of cells in each bins
fig, ax = plt.subplots(len(task_array),len(task_array))
COLOR_BAR_RANGE=len(cell_list)
for cur_couple_inx,cur_couple in enumerate(cond_couples_list):
    cur_N_sig_matrix=np.count_nonzero(~np.isnan(population_correlation_matrix_masked[cur_couple_inx,:,:,:]),axis=0)
    cond1_inx=cond_couples_list[cur_couple_inx][0]
    cond2_inx=cond_couples_list[cur_couple_inx][1]
    shw=ax[cond1_inx][cond2_inx].imshow(cur_N_sig_matrix , cmap = 'YlOrRd' , interpolation = 'nearest')
    ax[cond1_inx,cond2_inx].tick_params(axis='both', labelsize=4, pad=0)
    ax[cond1_inx][cond2_inx].set_title(condition_array[cond1_inx]+'-'+condition_array[cond2_inx],fontsize=8)
    cb=plt.colorbar(shw, ax=ax[cond1_inx, cond2_inx])
    cb.ax.tick_params(labelsize=6)
fig.tight_layout()
plt.show()
    

#cue-saccade
x=np.arange(window_begin,window_end,bin_lenth)+bin_lenth/2
cond_inx1=0
cond_inx2=1
cur_couple_inx = cond_couples_list.index((cond_inx1,cond_inx2))
cur_correlation_matrix=np.nanmean(population_correlation_matrix_masked[cur_couple_inx,:,:,:],axis=0)
cur_correlation_matrix_sem=stats.sem(population_correlation_matrix_masked[cur_couple_inx,:,:,:], axis=0, ddof=1, nan_policy='omit')
plt.plot(x,np.diag(cur_correlation_matrix))
plt.errorbar(x,np.diag(cur_correlation_matrix),np.diag(cur_correlation_matrix_sem))
plt.axhline(0,color='black',linestyle='dashed')
plt.xlabel('time from MO (ms)')
plt.ylabel('Correaltion')
plt.title('Average tuning correlation '+condition_array[cond_inx1]+'-'+condition_array[cond_inx2])
plt.show()

