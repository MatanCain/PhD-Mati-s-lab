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


task_array=['8dir_saccade_100_25','8dir_active_passive_interleaved_100_25','8dir_active_passive_interleaved_100_25']
trial_type_array=['filterOff','v20a','v20p']
event_array=['motion_onset','motion_onset','motion_onset']
condition_array=['saccade','pursuit active','pursuit passive']

# task_array=['8dir_saccade_100_25','8dir_active_passive_interleaved_100_25','8dir_active_passive_interleaved_100_25','8dir_active_passive_interleaved_100_25']
# trial_type_array=['filterOff','v20a','v20p','v20a|v20p']
# event_array=['motion_onset','motion_onset','motion_onset','cue_onset']
# condition_array=['saccade','pursuit active','pursuit passive','cue']

#separates cue to before active and before passive
# task_array=['8dir_saccade_100_25','8dir_active_passive_interleaved_100_25','8dir_active_passive_interleaved_100_25','8dir_active_passive_interleaved_100_25','8dir_active_passive_interleaved_100_25']
# trial_type_array=['filterOff','v20a','v20p','v20a','v20p']
# event_array=['motion_onset','motion_onset','motion_onset','cue_onset','cue_onset']
# condition_array=['saccade','pursuit active','pursuit passive','cue active','cue passive']

win_begin_array=[0,0,0,0,0]#for PD
win_end_array=[350,350,350,350,350]#for PD
win_begin_PSTH=-100
win_end_PSTH=350

SMOOTH_PSTH_EDGE=200


win_begin_baseline=-300 #for the baseline calculation of the cell
win_end_baseline=-100

PSTH_length=win_end_PSTH-win_begin_PSTH
directions=[0,45,90,135,180,225,270,315]

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

           
#%% check if cell is significant for cue before motion onset and if so remove it because correlation can be caused by cue
#
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
        #cell_sig_both_cond=bool(sig_pop_array[0,cell_inx] and sig_pop_array[3,cell_inx] ) 
        #cell_sig_both_cond=bool(sig_pop_array[0,cell_inx] and sig_pop_array[0,cell_inx] and sig_pop_array[1,cell_inx] and np.logical_not(sig_pop_array[2,cell_inx]) ) #keep only cells significant for both saccade, active pursuit and NOT passive pursuit (motor cells)
            
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
    PD_45=np.mean(np.vstack((shifted_PSTH_average_directions[2,:],shifted_PSTH_average_directions[4,:])),axis=0)
    PD_90=np.mean(np.vstack((shifted_PSTH_average_directions[1,:],shifted_PSTH_average_directions[5,:])),axis=0)
    PD_135=np.mean(np.vstack((shifted_PSTH_average_directions[0,:],shifted_PSTH_average_directions[6,:])),axis=0)
    PD_180= shifted_PSTH_average_directions[7,:]
    
    # if cur_couple_cond_inx==0:
    #     save_path="C:/Users/Owner/Documents/Matan/Mati's Lab/population analysis paper/figure6_active_saccade/"
    #     mdic = {"PSTH_saccade_PD_saccade":np.vstack((PD_0,PD_45,PD_90,PD_135,PD_180))}
    #     savemat(save_path+"PSTH_saccade_PD_saccade"+ ".mat", mdic)
    # if cur_couple_cond_inx==10:
    #     save_path="C:/Users/Owner/Documents/Matan/Mati's Lab/population analysis paper/figure6_active_saccade/"
    #     mdic = {"PSTH_saccade_PD_pursuit":np.vstack((PD_0,PD_45,PD_90,PD_135,PD_180))}
    #     savemat(save_path+"PSTH_saccade_PD_pursuit"+ ".mat", mdic)
    # if cur_couple_cond_inx==1:
    #     save_path="C:/Users/Owner/Documents/Matan/Mati's Lab/population analysis paper/figure6_active_saccade/"
    #     mdic = {"PSTH_pursuit_PD_saccade":np.vstack((PD_0,PD_45,PD_90,PD_135,PD_180))}
    #     savemat(save_path+"PSTH_pursuit_PD_saccade"+ ".mat", mdic)
    # if cur_couple_cond_inx==4:
    #     save_path="C:/Users/Owner/Documents/Matan/Mati's Lab/population analysis paper/figure6_active_saccade/"
    #     mdic = {"PSTH_pursuit_PD_pursuit":np.vstack((PD_0,PD_45,PD_90,PD_135,PD_180))}
    #     savemat(save_path+"PSTH_pursuit_PD_pursuit"+ ".mat", mdic)
        
    # if cur_couple_cond_inx==5:
    #     save_path="C:/Users/Owner/Documents/Matan/Mati's Lab/population analysis paper/figure5_active_passive/"
    #     mdic = {"PSTH_suppression_PD_pursuit":np.vstack((PD_0,PD_45,PD_90,PD_135,PD_180))}
    #     savemat(save_path+"PSTH_suppression_PD_pursuit"+ ".mat", mdic)
    # if cur_couple_cond_inx==13:
    #     save_path="C:/Users/Owner/Documents/Matan/Mati's Lab/population analysis paper/figure5_active_passive/"
    #     mdic = {"PSTH_pursuit_PD_suppression":np.vstack((PD_0,PD_45,PD_90,PD_135,PD_180))}
    #     savemat(save_path+"PSTH_pursuit_PD_suppression"+ ".mat", mdic)
    # if cur_couple_cond_inx==7:
    #     save_path="C:/Users/Owner/Documents/Matan/Mati's Lab/population analysis paper/figure5_active_passive/"
    #     mdic = {"PSTH_suppression_PD_suppression":np.vstack((PD_0,PD_45,PD_90,PD_135,PD_180))}
    #     savemat(save_path+"PSTH_suppression_PD_suppression"+ ".mat", mdic)
    # if cur_couple_cond_inx==4:
    #     save_path="C:/Users/Owner/Documents/Matan/Mati's Lab/population analysis paper/figure5_active_passive/"
    #     mdic = {"PSTH_pursuit_PD_pursuit":np.vstack((PD_0,PD_45,PD_90,PD_135,PD_180))}
    #     savemat(save_path+"PSTH_pursuit_PD_pursuit"+ ".mat", mdic)
    
    legend_array=['PD','PD_45','PD_90','PD_135','PD_180']
    ax[cond_inx1,cond_inx2].plot(timecourse,PD_0,linewidth=1)
    ax[cond_inx1,cond_inx2].plot(timecourse,PD_45,linewidth=0.3)
    ax[cond_inx1,cond_inx2].plot(timecourse,PD_90,linewidth=0.3)
    ax[cond_inx1,cond_inx2].plot(timecourse,PD_135,linewidth=0.3)
    ax[cond_inx1,cond_inx2].plot(timecourse,PD_180,linewidth=1)
    ax[cond_inx1,cond_inx2].tick_params(axis='both', labelsize=6, pad=0)
    ax[0,0].legend(legend_array,loc='upper right', prop = { "size": 3.5 },handlelength=0.5)
    ax[cond_inx1,cond_inx2].axvline(x=0, color='k',linestyle='--',linewidth=0.2)
    #plt.xlabel('time from event (ms)')
    #plt.ylabel('Delta FR (Hz)')
    ax[cond_inx1,cond_inx2].set_title(  condition_array[cond_inx2]+ ' by ' +condition_array[cond_inx1],y=1.0, pad=-5, loc='left',fontsize = 6)
    
fig.tight_layout()
plt.show()



#%%  Pairwise correlation correlation of flat PSTHs for a given condition
save_path="C:/Users/Owner/Documents/Matan/Mati's Lab/population analysis paper/figure4_PCA/"
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

#not_late_cue_cells_inxs=[True if str(cell_ID) not in cue_cells else False for cell_inx,cell_ID in enumerate(cell_list) ]

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

    #cell_inxs=np.where((sig_pop_array2[cond_inx1,:] & sig_pop_array2[cond_inx2,:] & not_late_cue_cells_inxs)==True  )[0] #remove cells responsive for late cue

    
    #condition 1
    saccade_PSTH=population_flat_PSTH[cond_inx1][cell_inxs,:]
    saccade_pairwise_correlation=np.corrcoef(saccade_PSTH)
    saccade_pairwise_covariance=np.cov(saccade_PSTH)
   
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
    correlation_array[cond_inx1,cond_inx2]=np.round(cond1_cond2_correlation,3)
    print(cur_couple_cond_inx)
    if cur_couple_cond_inx==6:
        mdic = {"pursuit_pairwise_correlation":cond1_array_noNan,"saccade_pairwise_correlation":cond2_array_noNan}
        savemat(save_path+"pursuit_saccade_pairwise_correlation"+ ".mat", mdic)
    elif cur_couple_cond_inx==4:
        mdic = {"pursuit_pairwise_correlation1":cond1_array_noNan,"suppression_pairwise_correlation1":cond2_array_noNan}
        savemat(save_path+"pursuit_suppression_pairwise_correlation"+ ".mat", mdic)    
    elif cur_couple_cond_inx==2:
        mdic = {"saccade_pairwise_correlation2":cond1_array_noNan,"suppression_pairwise_correlation2":cond2_array_noNan}
        savemat(save_path+"saccade_suppression_pairwise_correlation"+ ".mat", mdic) 


fig.suptitle('Pairwise correlation between conditions')
fig.tight_layout()
    
if current_measurement=='correlation': 
    fig.suptitle('Pairwise correlation between conditions')
if current_measurement=='covariance':     
    fig.suptitle('Pairwise covariance between conditions')
fig.tight_layout()
plt.show()

#Replaces the scatters by a heatmap showing the correlation of correlation
# shw=plt.imshow(correlation_array , cmap = 'YlOrRd' , interpolation = 'nearest')
# cb=plt.colorbar(shw)
# ax = plt.gca()
# ax.set_xticks([])
# ax.set_yticks([])
# plt.show()




#%% Fisher's r-to-z transformation - Determine if the difference in correlation between cue vs pursuit and cue vs saccade are significant with cells significant in all relvant conditions
#check cue vs saccade and cue vs pursuit
cond_inx0=0 
cond_inx1=1 
cond_inx2=2 
cond_inx3=3

 
#find indexes of cell sig for saccade and pursuit
sig_pop_array2=np.array(sig_pop_array,dtype=bool)
cell_inxs=np.where((sig_pop_array2[0,:] & sig_pop_array2[1,:] & sig_pop_array2[2,:])==True  )[0] 
 
#pre processing: calculate pairwise tuning correlation for all couple of cells within task   
#condition 0 - saccade
cond0_PSTH=population_flat_PSTH[cond_inx0][cell_inxs,:]
cond0_pairwise_correlation=np.corrcoef(cond0_PSTH)
#condition 1 - pursuit
cond1_PSTH=population_flat_PSTH[cond_inx1][cell_inxs,:]
cond1_pairwise_correlation=np.corrcoef(cond1_PSTH)
#condition 2 - suppression
cond2_PSTH=population_flat_PSTH[cond_inx2][cell_inxs,:]
cond2_pairwise_correlation=np.corrcoef(cond2_PSTH)
#condition 3 - cue
cond3_PSTH=population_flat_PSTH[cond_inx3][cell_inxs,:]
cond3_pairwise_correlation=np.corrcoef(cond3_PSTH)

cond0_array=[]
cond1_array=[]
cond2_array=[]
cond3_array=[]
for n1_inx in np.arange(np.size(cond1_pairwise_correlation,0)):
    for n2_inx in np.arange(np.size(cond1_pairwise_correlation,0)):
        if n1_inx>n2_inx:
            cond0_array.append(cond0_pairwise_correlation[n1_inx,n2_inx])
            cond1_array.append(cond1_pairwise_correlation[n1_inx,n2_inx])
            cond2_array.append(cond2_pairwise_correlation[n1_inx,n2_inx])
            cond3_array.append(cond3_pairwise_correlation[n1_inx,n2_inx])

#remove nan cells
cond0_array=np.array(cond0_array)
cond1_array=np.array(cond1_array)
cond2_array=np.array(cond2_array)
cond3_array=np.array(cond3_array)

cond0_array_noNan=cond0_array[~np.isnan(cond0_array) & ~np.isnan(cond1_array) & ~np.isnan(cond2_array) & ~np.isnan(cond3_array)]  
cond1_array_noNan=cond1_array[~np.isnan(cond0_array) & ~np.isnan(cond1_array) & ~np.isnan(cond2_array) & ~np.isnan(cond3_array)]
cond2_array_noNan=cond2_array[~np.isnan(cond0_array) & ~np.isnan(cond1_array) & ~np.isnan(cond2_array) & ~np.isnan(cond3_array)]
cond3_array_noNan=cond3_array[~np.isnan(cond0_array) & ~np.isnan(cond1_array) & ~np.isnan(cond2_array) & ~np.isnan(cond3_array)]

condA_noNan=cond1_array_noNan
condB_noNan=cond0_array_noNan
condC_noNan=cond2_array_noNan
#step 1: calculate the correlation in tuning correlation between (cue and saccade) and (cue and pursuit)
condB_condA_correlation=np.corrcoef(condB_noNan,condA_noNan)[0,1]
condC_condA_correlation=np.corrcoef(condC_noNan,condA_noNan)[0,1]   
diff_correlation=condB_condA_correlation-condC_condA_correlation #calculate the difference between those correlations  

# Correlation coefficients
r_AB = condB_condA_correlation  # Replace with your value
r_AC = condC_condA_correlation  # Replace with your value

# Sample sizes
n_AB = np.size(cond0_array_noNan)  # Replace with your sample size
n_AC = np.size(cond3_array_noNan)  # Replace with your sample size

# Fisher's r-to-z transformation
z_AB = 0.5 * np.log((1 + r_AB) / (1 - r_AB))
z_AC = 0.5 * np.log((1 + r_AC) / (1 - r_AC))

# Standard error of the difference
SE_diff = np.sqrt((1 / (n_AB - 3)) + (1 / (n_AC - 3)))

# Z-score for the difference
z_diff = (z_AB - z_AC) / SE_diff

# Two-tailed critical value for a significance level of 0.05
critical_value = norm.ppf(1 - 0.025)

# Calculate the p-value
p_value = 2 * (1 - norm.cdf(np.abs(z_diff)))

# Compare z_diff to critical value
if np.abs(z_diff) > critical_value:
    print("The difference between correlations is statistically significant.")
else:
    print("The difference between correlations is not statistically significant.")

# Back-transform the result
r_diff = (np.exp(2 * z_diff) - 1) / (np.exp(2 * z_diff) + 1)
print(f"Difference in correlations: {r_diff}")



#%% Correlation between PSTH of single cells in active pursuit and saccades

# sig_pop_array=np.array(sig_pop_array,dtype=bool)
# #cell_inxs=np.where((sig_pop_array[1,:] | sig_pop_array[0,:])==True  )[0] #active OR saccade cells
# #cell_inxs=np.where((sig_pop_array[1,:] & sig_pop_array[0,:])==True  )[0] #active and saccade cells
# cell_inxs_visuomotor=np.where((sig_pop_array[1,:] & sig_pop_array[0,:] & sig_pop_array[2,:]) ==True )[0] #active and saccade cells and passive - Visuomotor cells
# cell_inxs_motor=np.where((sig_pop_array[1,:] & sig_pop_array[0,:]& np.logical_not(sig_pop_array[2,:]))==True )[0] #active and saccade cells and passive - motor cells

# #motor
# saccade_PSTH_motor=population_flat_PSTH[0][cell_inxs_motor,:]
# active_PSTH_motor=population_flat_PSTH[1][cell_inxs_motor,:]
# #visuomotor
# saccade_PSTH_VM=population_flat_PSTH[0][cell_inxs_visuomotor,:]
# active_PSTH_VM=population_flat_PSTH[1][cell_inxs_visuomotor,:]

# cell_correlation_motor = [pearsonr(saccade_PSTH_motor[i, :], active_PSTH_motor[i, :])[0] for i in range(np.size(active_PSTH_motor,0))]
# cell_correlation_VM = [pearsonr(saccade_PSTH_VM[i, :], active_PSTH_VM[i, :])[0] for i in range(np.size(active_PSTH_VM,0))]

# fig, ax = plt.subplots(2)
# ax[0].hist(np.array(cell_correlation_motor),bins=np.arange(-1,1.1,0.1),density=True)
# ax[0].set_title('motor cells')
# ax[0].set_xlabel('saccade-active correlation')
# ax[1].hist(np.array(cell_correlation_VM),bins=np.arange(-1,1.1,0.1),density=True)
# ax[1].set_xlabel('saccade-active correlation')
# ax[1].set_title('VM cells')
# fig.tight_layout()
# plt.show()

#%% Check PSTHS of pair of cells with high covariance in saccade and pursuit
#Looks for pairs of neuron with high correlation in both saccades and active pursuit
#set the diagonals as nan instead of 1
# np.fill_diagonal(saccade_pairwise_covariance, np.nan)
# np.fill_diagonal(active_pairwise_covariance, np.nan)
# HIGH_CORR_THRESHOLD_COND1=30
# HIGH_CORR_THRESHOLD_COND2=5

# temp1=np.where(saccade_pairwise_covariance>HIGH_CORR_THRESHOLD_COND1)
# temp2=np.where(active_pairwise_covariance>HIGH_CORR_THRESHOLD_COND2)
# neuron_inx_list=[cell_inxs[x] for x in list(temp1[0]) if x in list(temp2[0])]
# neuron_inx_list=np.unique(np.array(neuron_inx_list))
# for neuron_inx in neuron_inx_list:

#     #PSTH in saccades
#     flat_PTSH_cell1_saccade=population_flat_PSTH[0][neuron_inx,:]
#     PTSH_cell1_saccade=np.reshape(flat_PTSH_cell1_saccade,(len(directions),-1))

#     flat_PTSH_cell1_active=population_flat_PSTH[1][neuron_inx,:]
#     PTSH_cell1_active=np.reshape(flat_PTSH_cell1_active,(len(directions),-1))

#     fig, ax = plt.subplots(2,1)
#     ax[0].plot(np.transpose(PTSH_cell1_saccade))
#     ax[1].plot(np.transpose(PTSH_cell1_active))
#     ax[0].set_title('saccade')
#     ax[1].set_title('cue')
#     ax[1].legend(directions)
# fig.tight_layout()
# plt.show()
    

#%% Check PSTHS of pair of cells with high covariance in saccade and pursuit
#Looks for pairs of neuron with high correlation in both saccades and active pursuit
#set the diagonals as nan instead of 1
# np.fill_diagonal(saccade_pairwise_covariance, np.nan)
# np.fill_diagonal(active_pairwise_covariance, np.nan)
# HIGH_CORR_THRESHOLD_COND1=30
# HIGH_CORR_THRESHOLD_COND2=5

# temp1=np.where(saccade_pairwise_covariance>HIGH_CORR_THRESHOLD_COND1)
# temp2=np.where(active_pairwise_covariance>HIGH_CORR_THRESHOLD_COND2)
# neuron1=[cell_inxs[x] for x in list(temp1[0]) if x in list(temp2[0])]
# neuron2=[cell_inxs[x] for x in list(temp1[1]) if x in list(temp2[1])]

# neuron1b=[x for x in list(temp1[0]) if x in list(temp2[0])]
# neuron2b=[x for x in list(temp1[1]) if x in list(temp2[1])]

# pair_high_corr=np.array(np.vstack([neuron1,neuron2])) 

# #Indexes are taken from cell_list
# pair_high_corr=pair_high_corr[:,pair_high_corr[0,:]<pair_high_corr[1,:]]  #Each row is a neuron, each column is apair of neuron

# for pair_inx in np.arange(np.size(pair_high_corr,1)):
#     cell_1_PSTH_inx=pair_high_corr[0][pair_inx]
#     cell_2_PSTH_inx=pair_high_corr[1][pair_inx]
    
#     #PSTH in saccades
#     flat_PTSH_cell1_saccade=population_flat_PSTH[0][cell_1_PSTH_inx,:]
#     PTSH_cell1_saccade=np.reshape(flat_PTSH_cell1_saccade,(len(directions),-1))

#     flat_PTSH_cell2_saccade=population_flat_PSTH[0][cell_2_PSTH_inx,:]
#     PTSH_cell2_saccade=np.reshape(flat_PTSH_cell2_saccade,(len(directions),-1))

#     #PSTH in pursuit
#     flat_PTSH_cell1_saccade=population_flat_PSTH[0][cell_1_PSTH_inx,:]
#     PTSH_cell1_saccade=np.reshape(flat_PTSH_cell1_saccade,(len(directions),-1))
    
#     flat_PTSH_cell2_saccade=population_flat_PSTH[0][cell_2_PSTH_inx,:]
#     PTSH_cell2_saccade=np.reshape(flat_PTSH_cell2_saccade,(len(directions),-1))
    
#     flat_PTSH_cell1_active=population_flat_PSTH[1][cell_1_PSTH_inx,:]
#     PTSH_cell1_active=np.reshape(flat_PTSH_cell1_active,(len(directions),-1))
    
#     flat_PTSH_cell2_active=population_flat_PSTH[1][cell_2_PSTH_inx,:]
#     PTSH_cell2_active=np.reshape(flat_PTSH_cell2_active,(len(directions),-1))
    
#     fig, ax = plt.subplots(2,2)
#     ax[0,0].plot(np.transpose(PTSH_cell1_saccade))
#     ax[1,0].plot(np.transpose(PTSH_cell2_saccade))
#     ax[0,1].plot(np.transpose(PTSH_cell1_active))
#     ax[1,1].plot(np.transpose(PTSH_cell2_active))
#     ax[0,0].set_title('cell 1 saccade')
#     ax[0,1].set_title('cell 1 cue')
#     ax[1,0].set_title('cell 2 saccade')
#     ax[1,1].set_title('cell 2 cue')
#     ax[1,1].legend(directions)
# fig.tight_layout()
# plt.show()
    
#%% Looks at PSTH of single cells - cells that react in active passive or cue

# sig_pop_array=np.array(sig_pop_array,dtype=bool)
# cell_inxs=np.where((sig_pop_array[1,:] & sig_pop_array[2,:] )==True )[0] #cells that are active and  passive

# task_array=['8dir_saccade_100_25','8dir_active_passive_interleaved_100_25','8dir_active_passive_interleaved_100_25']
# trial_type_array=['filterOff','v20a','v20p']
# event_array=['motion_onset','motion_onset','motion_onset']
# condition_array=['saccade','active','passive']
# win_begin=-400
# win_end=1000
# SMOOTH_EDGE=200
# PSTH_length=win_end-win_begin-2*SMOOTH_EDGE
# timecourse=np.arange(win_begin+SMOOTH_EDGE,win_end-SMOOTH_EDGE)
# directions=[0,90,180,270]

# cell_list2 = [cell_list[i] for i in cell_inxs]

# for cell_inx,cell_ID in enumerate(cell_list2):
#     cell_PSTH=np.empty([len(task_array),len(directions),PSTH_length])
#     cell_PSTH[:]=np.nan
#     for condition_inx in np.arange(len(task_array)):
#         cur_task=task_array[condition_inx]
#         cur_trial_type=trial_type_array[condition_inx]
#         cur_event=event_array[condition_inx]


        
#         for cur_dir_inx,cur_dir in enumerate(directions):
            
#             dictFilterTrials_PSTH = {'dir':cur_dir, 'trial_name':cur_trial_type, 'fail':0}
#             window_PSTH={"timePoint":cur_event,'timeBefore':win_begin,'timeAfter':win_end}
#             cur_cell_task=load_cell_task(cell_task_py_folder,cur_task,cell_ID)
#             try:
#                 cur_PSTH=cur_cell_task.PSTH(window_PSTH,dictFilterTrials_PSTH,smooth_option=1)
#                 cur_PSTH=cur_PSTH[SMOOTH_EDGE:-SMOOTH_EDGE]
#             except:
#                 continue
#             cell_PSTH[condition_inx,cur_dir_inx,:]=cur_PSTH
    
#     fig, ax = plt.subplots(len(task_array))
    
#     for condition_inx in np.arange(len(task_array)):
#         ax[condition_inx].plot(timecourse,np.transpose(cell_PSTH[condition_inx,:,:]))
#         ax[condition_inx].set_title(condition_array[condition_inx],fontsize=4)
#         ax[condition_inx].set_ylabel('FR')
#         ax[condition_inx].axvline(x=0,color='k')
#         ax[0].legend(directions,loc='upper right', prop = { "size": 5 },handlelength=0.5)
#     plt.suptitle(str(cell_ID))
#     fig.tight_layout()
#     plt.show()
    

