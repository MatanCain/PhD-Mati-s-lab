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

cell_list=os.listdir(cell_task_py_folder+'4dir_saccade_cue') #list of strings

task_array=['4dir_saccade_cue','4dir_saccade_cue']
trial_type_array=['filterOff','filterOff']
event_array=['motion_onset','cue_onset']
condition_array=['saccade','cue']

#separates cue to before active and before passive
# task_array=['8dir_saccade_100_25','8dir_active_passive_interleaved_100_25','8dir_active_passive_interleaved_100_25','8dir_active_passive_interleaved_100_25','8dir_active_passive_interleaved_100_25']
# trial_type_array=['filterOff','v20a','v20p','v20a','v20p']
# event_array=['motion_onset','motion_onset','motion_onset','cue_onset','cue_onset']
# condition_array=['saccade','pursuit active','pursuit passive','cue active','cue passive']

win_begin_array=[0,0]#for PD
win_end_array=[350,350]#for PD
win_begin_PSTH=0
win_end_PSTH=350

SMOOTH_PSTH_EDGE=200


win_begin_baseline=-300 #for the baseline calculation of the cell
win_end_baseline=-100

PSTH_length=win_end_PSTH-win_begin_PSTH
directions=[0,45,90,180,270]

population_flat_PSTH=[np.zeros([len(cell_list),PSTH_length*len(directions)]) for cur_task in task_array]#each element of the list is for a condition. Each element is a numpy array (n_cells*PSTH_length)
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
                
        #ADD OPTION 2: cell with reaction larger than 5 stds from baseline
        
        
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
                        
        for cur_direction in directions:
            dictFilterTrials_PSTH['dir']=cur_direction
            cur_PSTH=np.empty([PSTH_length])
            cur_PSTH[:]=np.nan
            try:
                cur_PSTH=cur_cell_task.PSTH(window_PSTH,dictFilterTrials_PSTH) #PSTH of cell for given condition and direction
                cur_PSTH=cur_PSTH[SMOOTH_PSTH_EDGE:-SMOOTH_PSTH_EDGE]-average_PSTH
                
            except:
                print(cell_ID)
                
            cell_cond_flat_PSTH=np.hstack([cell_cond_flat_PSTH,cur_PSTH])
            
        population_flat_PSTH[condition_inx][cell_inx,:]=cell_cond_flat_PSTH
            
# if a cell is missing a direction in a given condition then the PSTH is an array of nan. ex: dir 0 in saccade for cell 7391            

       
#%% Plot the average PSTH of each condition according to PD in another condition

#create a list with the indexes of condition
cond_inxs_list = list(range(0,len(task_array)))
#list with all the possible couple of condition including with themselves
cond_couples_list1 = [(a, b) for idx, a in enumerate(cond_inxs_list) for b in cond_inxs_list[idx:]]
cond_couples_list2= [(b, a) for idx, a in enumerate(cond_inxs_list) for b in cond_inxs_list[idx:]]
cond_couples_list=cond_couples_list1+cond_couples_list2
cond_couples_list = list(dict.fromkeys(cond_couples_list))

fig, ax = plt.subplots(len(task_array),len(task_array))
for cur_couple_cond_inx in np.arange(len(cond_couples_list)):
    cond_inx1=cond_couples_list[cur_couple_cond_inx][0]
    cond_inx2=cond_couples_list[cur_couple_cond_inx][1]
    shifted_PSTH_population=np.empty([len(cell_list),len(directions)*PSTH_length])
    shifted_PSTH_population[:]=np.nan
    for cell_inx,cell_ID in enumerate(cell_list):

        cell_sig_both_cond=bool(sig_pop_array[cond_inx1,cell_inx] and sig_pop_array[cond_inx2,cell_inx] ) #keep only cells significant for both condition

       # if cell_sig_both_cond and str(cell_ID) not in cue_cells: #remove cells responsive for late cue
        if cell_sig_both_cond :
            #find PD in condition 1
            PD_cond1=PD_pop_array[cond_inx1,cell_inx]
            #find the index od PD in the direction
            PD1_index = directions.index(PD_cond1)
            PD_INX=3 #we always want the PD to be in the 3 place in the flat PSTH
            #shift the PSTH of cond2 such that the PD dir of cond 1 will be in the middle
            shift_amplitude=PD_INX-PD1_index
            
            cell_shifted_PSTH=np.roll(population_flat_PSTH[cond_inx2][cell_inx,:], shift_amplitude*PSTH_length)
            shifted_PSTH_population[cell_inx,:]=cell_shifted_PSTH
        else:
            continue
    
    shifted_PSTH_average=np.nanmean(shifted_PSTH_population,axis=0) #average across cells
    shifted_PSTH_average_directions=np.reshape(shifted_PSTH_average,(len(directions),-1) ) #reshape the flat psth to 8 psth according to directions
    
    timecourse=np.arange(win_begin_PSTH,win_end_PSTH)   
    #couple symetric directions:
    PD_0=shifted_PSTH_average_directions[PD_INX,:]
    PD_90=np.mean(np.vstack((shifted_PSTH_average_directions[1,:],shifted_PSTH_average_directions[3,:])),axis=0)
    PD_180= shifted_PSTH_average_directions[2,:]
    
    if cur_couple_cond_inx==0:
        save_path="C:/Users/Owner/Documents/Matan/Mati's Lab/population analysis paper/figure7_cue/saccade_4_dir/"
        mdic = {"PSTH_saccade_PD_saccade":np.vstack((PD_0,PD_90,PD_180))}
        savemat(save_path+"PSTH_saccade_PD_saccade"+ ".mat", mdic)
    if cur_couple_cond_inx==1:
        save_path="C:/Users/Owner/Documents/Matan/Mati's Lab/population analysis paper/figure7_cue/saccade_4_dir/"
        mdic = {"PSTH_cue_PD_saccade":np.vstack((PD_0,PD_90,PD_180))}
        savemat(save_path+"PSTH_cue_PD_saccade"+ ".mat", mdic)
    if cur_couple_cond_inx==2:
        save_path="C:/Users/Owner/Documents/Matan/Mati's Lab/population analysis paper/figure7_cue/saccade_4_dir/"
        mdic = {"PSTH_cue_PD_cue":np.vstack((PD_0,PD_90,PD_180))}
        savemat(save_path+"PSTH_cue_PD_cue"+ ".mat", mdic)
    if cur_couple_cond_inx==3:
        save_path="C:/Users/Owner/Documents/Matan/Mati's Lab/population analysis paper/figure7_cue/saccade_4_dir/"
        mdic = {"PSTH_saccade_PD_cue":np.vstack((PD_0,PD_90,PD_180))}
        savemat(save_path+"PSTH_saccade_PD_cue"+ ".mat", mdic)
        
    legend_array=['PD','PD_90','PD_180']
    ax[cond_inx1,cond_inx2].plot(timecourse,PD_0,linewidth=1)
    ax[cond_inx1,cond_inx2].plot(timecourse,PD_90,linewidth=0.3)
    ax[cond_inx1,cond_inx2].plot(timecourse,PD_180,linewidth=1)
    ax[cond_inx1,cond_inx2].tick_params(axis='both', labelsize=6, pad=0)
    ax[0,0].legend(legend_array,loc='upper right', prop = { "size": 3.5 },handlelength=0.5)
    ax[cond_inx1,cond_inx2].axvline(x=0, color='k',linestyle='--',linewidth=0.2)
    #plt.xlabel('time from event (ms)')
    #plt.ylabel('Delta FR (Hz)')
    ax[cond_inx1,cond_inx2].set_title(  condition_array[cond_inx2]+ ' by ' +condition_array[cond_inx1],y=1.0, pad=-5, loc='left',fontsize = 6)
    
fig.tight_layout()
plt.show()


