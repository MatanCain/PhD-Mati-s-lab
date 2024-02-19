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
from scipy.stats import kruskal
from random import sample
from scipy.stats import norm
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
task_array=['8dir_saccade_100_25','8dir_active_passive_interleaved_100_25','8dir_active_passive_interleaved_100_25']
trial_type_array=['filterOff','v20a','v20p']
event_array=['motion_onset','motion_onset','motion_onset']
condition_array=['saccade','pursuit active','pursuit passive']

#separates cue to before active and before passive
# task_array=['8dir_saccade_100_25','8dir_active_passive_interleaved_100_25','8dir_active_passive_interleaved_100_25','8dir_active_passive_interleaved_100_25','8dir_active_passive_interleaved_100_25']
# trial_type_array=['filterOff','v20a','v20p','v20a','v20p']
# event_array=['motion_onset','motion_onset','motion_onset','cue_onset','cue_onset']
# condition_array=['saccade','pursuit active','pursuit passive','cue active','cue passive']

win_begin_array=[-100,-100,-100]#for PD
win_end_array=[350,350,350]#for PD
win_begin_PSTH=-100
win_end_PSTH=350

SMOOTH_PSTH_EDGE=200


win_begin_baseline=-300 #for the baseline calculation of the cell
win_end_baseline=-100

PSTH_length=win_end_PSTH-win_begin_PSTH
directions=[0,45,90,135,180,225,270,315]

population_flat_PSTH=[np.zeros([len(cell_list),PSTH_length]) for cur_task in task_array]#each element of the list is for a condition. Each element is a numpy array (n_cells*PSTH_length)
for cur_np_array in population_flat_PSTH:
    cur_np_array[:]=np.nan


PD_pop_array=np.empty([len(task_array),len(cell_list)])
PD_pop_array[:]=np.nan
sig_pop_array=np.empty([len(task_array),len(cell_list)])
sig_pop_array[:]=np.nan

for cell_inx,cell_ID in enumerate(cell_list):
    cell_tuning_curve=np.empty([len(task_array),len(directions)])
    cell_tuning_curve[:]=np.nan
    
    for condition_inx in np.arange(len(task_array)):
        cur_task=task_array[condition_inx]
        cur_trial_type=trial_type_array[condition_inx]
        cur_event=event_array[condition_inx]
        cur_win_begin=win_begin_array[condition_inx]
        cur_win_end=win_end_array[condition_inx]
        
        cur_cell_task=load_cell_task(cell_task_py_folder,cur_task,cell_ID)
        
        
        #Find PD of the cells
        dictFilterTrials_PD = {'dir':'filterOff', 'trial_name':cur_trial_type, 'fail':0}
        
        window_PD={"timePoint":cur_event,'timeBefore':cur_win_begin,'timeAfter':cur_win_end}
        try:
            PD_cell_condition=cur_cell_task.get_PD(window_PD,dictFilterTrials_PD) 
        except:
            continue
        PD_pop_array[condition_inx,cell_inx]=PD_cell_condition
                      
       # Check whether cell is significant for the event relative to baseline       
        #option 1
        try:
            check_sig_cell_cond=cur_cell_task.check_main_effect_motion_vs_baseline(cur_event,dictFilterTrials_PD,crit_value=0.01)
        except:
            check_sig_cell_cond=False
        sig_pop_array[condition_inx,cell_inx]=check_sig_cell_cond
        
        # Create a flatPSTH (משורשר)
        dictFilterTrials_PSTH = {'dir':'filterOff', 'trial_name':cur_trial_type, 'fail':0}

        window_PSTH={"timePoint":cur_event,'timeBefore':win_begin_PSTH-SMOOTH_PSTH_EDGE,'timeAfter':win_end_PSTH+SMOOTH_PSTH_EDGE}
        cell_cond_flat_PSTH=np.array([])
        
        #calculate average PSTH
        try:
            average_PSTH=cur_cell_task.PSTH(window_PSTH,dictFilterTrials_PSTH) #PSTH of cell for given condition and direction
            average_PSTH=average_PSTH[SMOOTH_PSTH_EDGE:-SMOOTH_PSTH_EDGE]
        except:
            print(cell_ID)
                        
            
        population_flat_PSTH[condition_inx][cell_inx,:]=average_PSTH
            
# if a cell is missing a direction in a given condition then the PSTH is an array of nan. ex: dir 0 in saccade for cell 7391 

           


#%%  Pairwise correlation correlation of flat PSTHs for a given condition
#current_measurement='correlation' #or covariance (which is the correlation before normalization by std)
current_measurement='covariance'

#create a list with the indexes of condition
cond_inxs_list = list(range(0,len(task_array)))
#list with all the possible couple of condition including with themselves
cond_couples_list1 = [(a, b) for idx, a in enumerate(cond_inxs_list) for b in cond_inxs_list[idx:]]
cond_couples_list2= [(b, a) for idx, a in enumerate(cond_inxs_list) for b in cond_inxs_list[idx:]]
cond_couples_list=cond_couples_list1+cond_couples_list2
cond_couples_list = list(dict.fromkeys(cond_couples_list))

fig, ax = plt.subplots(len(task_array),len(task_array))
correlation_array=np.empty([len(cond_inxs_list),len(cond_inxs_list)])
correlation_array[:]=np.nan


for cur_couple_cond_inx in np.arange(len(cond_couples_list)):
    cond_inx1=cond_couples_list[cur_couple_cond_inx][0]
    cond_inx2=cond_couples_list[cur_couple_cond_inx][1]
    ax[cond_inx1,cond_inx2].tick_params(axis='both', labelsize=6, pad=0)

    if cond_inx1==cond_inx2:
        continue
    
    #Select only cells that respond to saccade and passive pursuit
    #find indexes of cell sig for saccade and pursuit
    sig_pop_array2=np.array(sig_pop_array,dtype=bool)
    #cell_inxs=np.where((sig_pop_array2[cond_inx1,:] & sig_pop_array2[cond_inx2,:])==True)[0]
    cell_inxs=np.where((sig_pop_array2[cond_inx1,:] | sig_pop_array2[cond_inx2,:])==True)[0]
    
    #condition 1
    saccade_PSTH=population_flat_PSTH[cond_inx1][cell_inxs,:]
    saccade_pairwise_correlation=np.corrcoef(saccade_PSTH) #correlation between PSTHs
    saccade_pairwise_covariance=np.cov(saccade_PSTH) #covariance between PSTHs
   
    #condition 2
    active_PSTH=population_flat_PSTH[cond_inx2][cell_inxs,:]
    active_pairwise_correlation=np.corrcoef(active_PSTH)
    active_pairwise_covariance=np.cov(active_PSTH)

    cond1_array=[]
    cond2_array=[]
    for n1_inx in np.arange(np.size(saccade_pairwise_correlation,0)):
        for n2_inx in np.arange(np.size(active_pairwise_correlation,0)):
            if n1_inx>n2_inx:
                if current_measurement=='correlation':
                    cond1_array.append(saccade_pairwise_correlation[n1_inx,n2_inx])
                    cond2_array.append(active_pairwise_correlation[n1_inx,n2_inx])
                if current_measurement=='covariance':
                    cond1_array.append(saccade_pairwise_covariance[n1_inx,n2_inx])
                    cond2_array.append(active_pairwise_covariance[n1_inx,n2_inx])    
    #remove nan cells
    cond1_array=np.array(cond1_array)
    cond2_array=np.array(cond2_array)
    
    cond1_array_noNan=cond1_array[~np.isnan(cond1_array) & ~np.isnan(cond2_array)]
    cond2_array_noNan=cond2_array[~np.isnan(cond1_array) & ~np.isnan(cond2_array)]

    cond1_cond2_correlation=np.corrcoef(cond1_array_noNan,cond2_array_noNan)[0,1]    
    ax[cond_inx1,cond_inx2].plot(cond1_array_noNan,cond2_array_noNan,'.', markersize=1.5)
    ax[cond_inx1,cond_inx2].plot([-1,1],[-1,1],color='k',linewidth=0.5)
    #ax[cond_inx1,cond_inx2].set_xlabel('cond1 pairwise correltion')
    #ax[cond_inx1,cond_inx2].set_ylabel('cond 2 pairwise correltion')
    if current_measurement=='correlation':
        ax[cond_inx1,cond_inx2].set_xlim([-1,1])
        ax[cond_inx1,cond_inx2].set_ylim([-1,1])
    ax[cond_inx1,cond_inx2].set_title('corr:'+str(np.round(cond1_cond2_correlation,3)),fontsize=6, pad=0)
    ax[cond_inx1,cond_inx2].set_xlabel(condition_array[cond_inx1])
    ax[cond_inx1,cond_inx2].set_ylabel(condition_array[cond_inx2])
    
    correlation_array[cond_inx1,cond_inx2]=np.round(cond1_cond2_correlation,3)

    if cur_couple_cond_inx==6:
        save_path="C:/Users/Owner/Documents/Matan/Mati's Lab/population analysis paper/figureR5_omega_time/"
        mdic = {"pursuit_pairwise_correlation1":cond1_array_noNan,"saccade_pairwise_correlation1":cond2_array_noNan}
        savemat(save_path+"pursuit_saccade_pairwise_correlation"+ ".mat", mdic)

    elif cur_couple_cond_inx==4:
        save_path="C:/Users/Owner/Documents/Matan/Mati's Lab/population analysis paper/figureR5_omega_time/"

        mdic = {"pursuit_pairwise_correlation3":cond1_array_noNan,"suppression_pairwise_correlation3":cond2_array_noNan}
        savemat(save_path+"pursuit_suppression_pairwise_correlation"+ ".mat", mdic)    

fig.suptitle('Pairwise correlation between conditions')
fig.tight_layout()
    
if current_measurement=='correlation': 
    fig.suptitle('Pairwise correlation between conditions')
if current_measurement=='covariance':     
    fig.suptitle('Pairwise covariance between conditions')
fig.tight_layout()
plt.show()

#Replaces the scatters by a heatmap showing the correlation of correlation
shw=plt.imshow(correlation_array , cmap = 'YlOrRd' , interpolation = 'nearest')
cb=plt.colorbar(shw)
ax = plt.gca()
ax.set_xticks([])
ax.set_yticks([])
plt.show()


#%% sanity check


# saccade_PSTH=population_flat_PSTH[0]
# saccade_pairwise_correlation=np.corrcoef(saccade_PSTH) #correlation between PSTHs

    
# cell_1_inx=20
# cell_2_inx=1


# cell_1_PSTH=population_flat_PSTH[0][cell_1_inx,:]-np.nanmean(population_flat_PSTH[0][cell_1_inx,:])
# cell_2_PSTH=population_flat_PSTH[0][cell_2_inx,:]-np.nanmean(population_flat_PSTH[0][cell_2_inx,:])

# plt.plot(cell_1_PSTH,label='cell1')
# plt.plot(cell_2_PSTH,label='cell2')
# plt.show()


