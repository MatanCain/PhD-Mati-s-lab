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
cell_pursuit_list=os.listdir(cell_task_py_folder+'8dir_active_passive_interleaved_100_25') #list of strings
cell_saccade_list=os.listdir(cell_task_py_folder+'8dir_saccade_100_25') #list of strings
cell_list=[x for x in cell_saccade_list if x in cell_pursuit_list]
cell_list=[int(item) for item in cell_list] #list of ints
cutoff_cell=8229 #cutoff between yasmin and fiona
yasmin_cell_list=[x for x in cell_list if x>cutoff_cell]
fiona_cell_list=[x for x in cell_list if x<=cutoff_cell]
cell_list=cell_list


#check if cell is significant for cue before motion onset and if so remove it because correlation can be caused by cue
cue_cells=[]
directions=[0,45,90,135,180,225,270,315]
cur_task='8dir_active_passive_interleaved_100_25'
dictFilterTrials_TC = {'dir':'filterOff', 'trial_name':'v20a|v20p', 'fail':0}

for cell_ID in cell_pursuit_list:
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

cue_cells=[x for x in cue_cells if x in cell_saccade_list]
#%% Heatmap of avergae correlations between tuning curves of bins (100 ms) of cells
#Each element in the heatmap is the average of the correlation between tuning in those bins correlation across cells significant for tuning in those bins 

task_array=['8dir_saccade_100_25','8dir_active_passive_interleaved_100_25','8dir_active_passive_interleaved_100_25','8dir_active_passive_interleaved_100_25']
trial_type_array=['filterOff','v20a','v20p','v20a|v20p']
event_array=['motion_onset','motion_onset','motion_onset','cue_onset']
condition_array=['saccade',' active','passive','cue']

bin_lenth=50 #ms
window_begin=-200
window_end=700
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


#%%
#t test between 2 conditions in comparison to a reference condition
#for example: Is sacade more correlated to cue than pursuit is correlated to cue
cond_ref=3
cond1=0
cond2=2
couple_inx1=cond_couples_list.index((cond1,cond_ref))
couple_inx2=cond_couples_list.index((cond2,cond_ref))

#keep relevant matrices in the population correlation matrix
population_correlation_matrix_couple_inx1=population_correlation_matrix[couple_inx1,:,:,:]
population_correlation_matrix_couple_inx2=population_correlation_matrix[couple_inx2,:,:,:]

#extract diagonal (in time)
population_correlation_matrix_couple_inx1=np.diagonal(population_correlation_matrix_couple_inx1,axis1=1,axis2=2)
population_correlation_matrix_couple_inx2=np.diagonal(population_correlation_matrix_couple_inx2,axis1=1,axis2=2)

 #select the relevant time range
win_begin2=0
win_end2=350

bin_begin=int((win_begin2-window_begin)/bin_lenth)
bin_end=int((win_end2-window_end)/bin_lenth)

population_correlation_matrix_couple_inx1=population_correlation_matrix_couple_inx1[:,bin_begin:bin_end]
population_correlation_matrix_couple_inx2=population_correlation_matrix_couple_inx2[:,bin_begin:bin_end]

population_correlation_matrix_couple_inx1=np.nanmean(population_correlation_matrix_couple_inx1,axis=1)
population_correlation_matrix_couple_inx2=np.nanmean(population_correlation_matrix_couple_inx2,axis=1)

stats.ttest_rel(population_correlation_matrix_couple_inx1,population_correlation_matrix_couple_inx2,nan_policy='omit')
#%% Filter population_correlation_matrix (remove cells significant (or not) for passive)


not_late_cue_cells_inx=[x_inx for x_inx,x in enumerate(cell_list) if str(x) not in cue_cells]
population_correlation_matrix_masked=population_correlation_matrix
#population_correlation_matrix_masked=population_correlation_matrix[:,not_late_cue_cells_inx,:,:]

#%% subplot heatmaps of correlations
fig, ax = plt.subplots(4,4)
COLOR_BAR_RANGE=0.75
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

#Plot the diagonals of the correlation matrix for active vs passive and active vs saccade
x=np.arange(window_begin,window_end,bin_lenth)+bin_lenth/2
#active-passive 
cur_couple_inx = cond_couples_list.index((1,2))
cur_correlation_matrix=np.nanmean(population_correlation_matrix_masked[cur_couple_inx,:,:,:],axis=0)
cur_correlation_matrix_sem=stats.sem(population_correlation_matrix_masked[cur_couple_inx,:,:,:], axis=0, ddof=1, nan_policy='omit')
plt.plot(x,np.diag(cur_correlation_matrix))
plt.errorbar(x,np.diag(cur_correlation_matrix),np.diag(cur_correlation_matrix_sem))

#save_path="C:/Users/Owner/Documents/Matan/Mati's Lab/population analysis paper/figure4_active_passive/"
# mdic = {'timecourse':x,"tuning_correlation_pursuit_suppression":np.diag(cur_correlation_matrix),"tuning_correlation_pursuit_suppression_SEM":np.diag(cur_correlation_matrix_sem)}
# savemat(save_path+"tuning_correlation_pursuit_suppression"+ ".mat", mdic)

# mdic = {'timecourse':x,"tuning_correlation_pursuit_suppression_noCue":np.diag(cur_correlation_matrix),"tuning_correlation_pursuit_suppression_noCue_SEM":np.diag(cur_correlation_matrix_sem)}
# savemat(save_path+"tuning_correlation_pursuit_suppression_noCue"+ ".mat", mdic)


#active-saccade
cur_couple_inx = cond_couples_list.index((1,0))
cur_correlation_matrix=np.nanmean(population_correlation_matrix_masked[cur_couple_inx,:,:,:],axis=0)
cur_correlation_matrix_sem=stats.sem(population_correlation_matrix_masked[cur_couple_inx,:,:,:], axis=0, ddof=1, nan_policy='omit')
plt.plot(x,np.diag(cur_correlation_matrix))
plt.errorbar(x,np.diag(cur_correlation_matrix),np.diag(cur_correlation_matrix_sem))
plt.axhline(0,color='black',linestyle='dashed')
plt.xlabel('time from MO (ms)')
plt.ylabel('Correaltion')
plt.title('Average tuning correlation across neurons')
plt.show()

# mdic = {'timecourse':x,"tuning_correlation_pursuit_saccade":np.diag(cur_correlation_matrix),"tuning_correlation_pursuit_saccade_SEM":np.diag(cur_correlation_matrix_sem)}
# savemat(save_path+"tuning_correlation_pursuit_saccade"+ ".mat", mdic)

# mdic = {'timecourse':x,"tuning_correlation_pursuit_saccade_noCue":np.diag(cur_correlation_matrix),"tuning_correlation_pursuit_saccade_noCue_SEM":np.diag(cur_correlation_matrix_sem)}
# savemat(save_path+"tuning_correlation_pursuit_saccade_noCue"+ ".mat", mdic)

#active-cue
cur_couple_inx = cond_couples_list.index((1,3))
cur_correlation_matrix=np.nanmean(population_correlation_matrix_masked[cur_couple_inx,:,:,:],axis=0)
cur_correlation_matrix_sem=stats.sem(population_correlation_matrix_masked[cur_couple_inx,:,:,:], axis=0, ddof=1, nan_policy='omit')
plt.plot(x,np.diag(cur_correlation_matrix))
plt.errorbar(x,np.diag(cur_correlation_matrix),np.diag(cur_correlation_matrix_sem))
plt.axhline(0,color='black',linestyle='dashed')
plt.xlabel('time from MO (ms)')
plt.ylabel('Correaltion')
plt.title('Average tuning correlation - pursuit and cue')
plt.show()
# save_path="C:/Users/Owner/Documents/Matan/Mati's Lab/population analysis paper/figure7_cue_saccade/"
# mdic = {'timecourse':x,"tuning_correlation_pursuit_cue":np.diag(cur_correlation_matrix),"tuning_correlation_pursuit_cue_SEM":np.diag(cur_correlation_matrix_sem)}
# savemat(save_path+"tuning_correlation_pursuit_cue"+ ".mat", mdic)


#supression-cue
cur_couple_inx = cond_couples_list.index((2,3))
cur_correlation_matrix=np.nanmean(population_correlation_matrix_masked[cur_couple_inx,:,:,:],axis=0)
cur_correlation_matrix_sem=stats.sem(population_correlation_matrix_masked[cur_couple_inx,:,:,:], axis=0, ddof=1, nan_policy='omit')
plt.plot(x,np.diag(cur_correlation_matrix))
plt.errorbar(x,np.diag(cur_correlation_matrix),np.diag(cur_correlation_matrix_sem))
plt.axhline(0,color='black',linestyle='dashed')
plt.xlabel('time from MO (ms)')
plt.ylabel('Correaltion')
plt.title('Average tuning correlation - suppression and cue')
plt.show()

# save_path="C:/Users/Owner/Documents/Matan/Mati's Lab/population analysis paper/figure7_cue_saccade/"
# mdic = {'timecourse':x,"tuning_correlation_suppression_cue":np.diag(cur_correlation_matrix),"tuning_correlation_suppression_cue_SEM":np.diag(cur_correlation_matrix_sem)}
# savemat(save_path+"tuning_correlation_suppression_cue"+ ".mat", mdic)

#saccade-cue
cur_couple_inx = cond_couples_list.index((3,0))
cur_correlation_matrix=np.nanmean(population_correlation_matrix_masked[cur_couple_inx,:,:,:],axis=0)
cur_correlation_matrix_sem=stats.sem(population_correlation_matrix_masked[cur_couple_inx,:,:,:], axis=0, ddof=1, nan_policy='omit')
plt.plot(x,np.diag(cur_correlation_matrix))
plt.errorbar(x,np.diag(cur_correlation_matrix),np.diag(cur_correlation_matrix_sem))
plt.axhline(0,color='black',linestyle='dashed')
plt.xlabel('time from MO (ms)')
plt.ylabel('Correaltion')
plt.title('Average tuning correlation - saccade and cue')
plt.show()
# save_path="C:/Users/Owner/Documents/Matan/Mati's Lab/population analysis paper/figure7_cue_saccade/"
# mdic = {'timecourse':x,"tuning_correlation_saccade_cue":np.diag(cur_correlation_matrix),"tuning_correlation_saccade_cue_SEM":np.diag(cur_correlation_matrix_sem)}
# savemat(save_path+"tuning_correlation_saccade_cue"+ ".mat", mdic)

# legend_array=[]
# cur_correlation_matrix=np.nanmean(population_correlation_matrix_masked[cur_couple_inx,:,:,:],axis=0)
# plt.bar(np.diag(cur_correlation_matrix))
# for cur_couple_inx,cur_couple in enumerate(cond_couples_list):
    
#     cur_correlation_matrix=np.nanmean(population_correlation_matrix_masked[cur_couple_inx,:,:,:],axis=0)
#     cond1_inx=cond_couples_list[cur_couple_inx][0]
#     cond2_inx=cond_couples_list[cur_couple_inx][1]
#     if cond1_inx<=cond2_inx:
#         continue
#     plt.plot(np.diag(cur_correlation_matrix))
#     legend_array.append(condition_array[cond1_inx]+'-'+condition_array[cond2_inx])
# plt.xlabel('bin inx')
# plt.ylabel('correlation')
# plt.title('correlation of tuning curve')
# plt.legend(legend_array,loc='upper right')    
# plt.show()
    

#subplot of number of significant pair of cells in each bins
fig, ax = plt.subplots(4,4)
COLOR_BAR_RANGE=len(cell_list)
for cur_couple_inx,cur_couple in enumerate(cond_couples_list):

    cur_N_sig_matrix=np.count_nonzero(~np.isnan(population_correlation_matrix_masked[cur_couple_inx,:,:,:]),axis=0)
    cond1_inx=cond_couples_list[cur_couple_inx][0]
    cond2_inx=cond_couples_list[cur_couple_inx][1]
    shw=ax[cond1_inx][cond2_inx].imshow(cur_N_sig_matrix , cmap = 'YlOrRd' , interpolation = 'nearest', vmin=0, vmax=200)
    ax[cond1_inx,cond2_inx].tick_params(axis='both', labelsize=4, pad=0)
    ax[cond1_inx][cond2_inx].set_title(condition_array[cond1_inx]+'-'+condition_array[cond2_inx],fontsize=8)
    cb=plt.colorbar(shw, ax=ax[cond1_inx, cond2_inx])
    cb.ax.tick_params(labelsize=6)
    
#    if cur_couple_inx==5:
        # save_path="C:/Users/Owner/Documents/Matan/Mati's Lab/population analysis paper/figure4_active_passive/"
        # mdic = {"N_sig_cells_pursuit_suppression":np.diag(cur_N_sig_matrix)}
        # savemat(save_path+"N_sig_cells_pursuit_suppression"+ ".mat", mdic)
        
        # save_path="C:/Users/Owner/Documents/Matan/Mati's Lab/population analysis paper/figure4_active_passive/"
        # mdic = {"N_sig_cells_pursuit_suppression_noCue":np.diag(cur_N_sig_matrix)}
        # savemat(save_path+"N_sig_cells_pursuit_suppression_noCue"+ ".mat", mdic)
#    if cur_couple_inx==10:
        # save_path="C:/Users/Owner/Documents/Matan/Mati's Lab/population analysis paper/figure5_active_saccade/"
        # mdic = {"N_sig_cells_pursuit_saccade":np.diag(cur_N_sig_matrix)}
        # savemat(save_path+"N_sig_cells_pursuit_saccade"+ ".mat", mdic)

        # save_path="C:/Users/Owner/Documents/Matan/Mati's Lab/population analysis paper/figure5_active_saccade/"
        # mdic = {"N_sig_cells_pursuit_saccade_noCue":np.diag(cur_N_sig_matrix)}
        # savemat(save_path+"N_sig_cells_pursuit_saccade_noCue"+ ".mat", mdic)
            
fig.tight_layout()
plt.show()
    

#%% distribution of correlation across cell for a given couple of conditions
couples_indexes_list=[(3,0),(3,1),(3,2)]
title_array=['saccade_cue','pursuit_cue','suppresion_cue']

#time we want to look at during the trial
win_begin2=0
win_end2=350


bin_begin=int((win_begin2-window_begin)/bin_lenth)
bin_end=int((win_end2-window_end)/bin_lenth)

fig, ax = plt.subplots(len(title_array),1)
for cur_couple_inx,cur_couple in enumerate(couples_indexes_list):
    cell_correlations=[]
    cur_correlation_matrix=population_correlation_matrix_masked[cond_couples_list.index(cur_couple),:,:,:]
    for cur_cell_inx in np.arange(np.size(cur_correlation_matrix,axis=0)):
            cell_mean_correlation=np.nanmean(np.diag(cur_correlation_matrix[cur_cell_inx,bin_begin:bin_end,bin_begin:bin_end]))
            if 0 in cur_couple:
                cell_mean_correlation=-cell_mean_correlation
            if not(np.isnan(cell_mean_correlation)):
                cell_correlations.append(cell_mean_correlation)   
    ax[cur_couple_inx].hist(cell_correlations)
    ax[cur_couple_inx].set_title(title_array[cur_couple_inx],fontsize=8)
    ax[cur_couple_inx].set_xlim((-1,1))
    #ax[cur_couple_inx].set_ylim((0,1.1))
fig.tight_layout()
plt.show()
            
    
