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
from sklearn import linear_model
import pandas as pd
from scipy import signal


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
#%% Check if cell is significant for cue before motion onset and if so remove it because correlation can be caused by cue
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
        

task_array=['8dir_saccade_100_25','8dir_active_passive_interleaved_100_25','8dir_active_passive_interleaved_100_25']
trial_type_array=['filterOff','v20a','v20p']
event_array=['motion_onset','motion_onset','motion_onset']
condition_array=['saccade','active','passive']

#PCA_conditions='PCA_condition_apart' #Calculate PCs for each condition apart
PCA_conditions='PCA_condition_together' #calculate PCs based on all conditions togethher

#Window for PSTH calculation
win_begin_PSTH=0 
win_end_PSTH=350
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
                if PCA_conditions=='PCA_condition_apart':
                    cur_PSTH_PCA=cur_PSTH_PCA[SMOOTH_PSTH_EDGE:-SMOOTH_PSTH_EDGE]-average_PSTH_PCA
                else:
                    cur_PSTH_PCA=cur_PSTH_PCA[SMOOTH_PSTH_EDGE:-SMOOTH_PSTH_EDGE]
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

#remove cells signififant for late cue cells
# late_cue_cells_inxs=[cell_inx for cell_inx,cell_ID in enumerate(cell_list) if (cell_ID) in cue_cells ]
# nan_inxs=np.hstack((nan_inxs,non_reactive_cells_inx,late_cue_cells_inxs)) 
# nan_inxs=np.unique(nan_inxs)
# nan_inxs=[int(x) for x  in nan_inxs]

n_cells=len(cell_list)-len(nan_inxs)
#n_PCs=3#number of PCs

min_PCs=2
max_PCs=7
N_PCs=max_PCs-min_PCs
r_square_array=np.empty([N_PCs,max_PCs-1])
r_square_array[:]=np.nan #r square for linear fit. Each row is for a number of PCs, each column is the r square for a given PC
for PCInx,n_PCs in enumerate(np.arange(min_PCs,max_PCs)): #loop across a number of total PCs
    PCs_array=np.empty([len(task_array),n_PCs,n_cells])
    exp_var_array=np.empty([len(task_array),n_PCs])
    eigen_values_array=np.empty([len(task_array),n_PCs])
    
    #Finding PCS of each conditions
    for cond_inx in np.arange(len(task_array)):
        if PCA_conditions=='PCA_condition_apart':
            cond_flat_PSTH=population_flat_PSTH_PCA[cond_inx]
            cond_flat_PSTH=np.delete(cond_flat_PSTH,nan_inxs,axis=0)
    
        elif PCA_conditions=='PCA_condition_together':
            cond_flat_PSTH=np.hstack(population_flat_PSTH_PCA)
            cond_flat_PSTH=np.delete(cond_flat_PSTH,nan_inxs,axis=0)
            cell_means=np.mean(cond_flat_PSTH,axis=1)
            cell_means2=np.tile(cell_means, (np.size(cond_flat_PSTH,1),1))
            cond_flat_PSTH=cond_flat_PSTH-np.transpose(cell_means2) #remove average across cells    
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
        
    #Create PC_array a numpy array where each row is the projection of a PSTH on a PC
    #The PSTH is number of conditions* number of directions * trial length
    timecourse=np.arange(win_begin_PSTH,win_end_PSTH)
    PC_array=[]
    for cond_inx in np.arange(len(task_array)):
        cond_PCs_array=np.empty([n_PCs,len(directions)*PSTH_length])
        cond_PCs_array[:]=np.nan
        for PC_inx in np.arange(n_PCs): #we project the activity of the current condition on the current PC 
            #choose the PC from cond_inx
            cur_PC=PCs_array[cond_inx,PC_inx,:]
            #choose a PSTH fron cond_inx
            cond_flat_PSTH2=population_flat_PSTH_PCA[cond_inx]
            cond_flat_PSTH2=np.delete(cond_flat_PSTH2,nan_inxs,axis=0)
            #project the neural activity on the PC
            data_proj_PC=np.matmul(np.transpose(cond_flat_PSTH2),cur_PC)
            cond_PCs_array[PC_inx,:]=data_proj_PC
        PC_array.append(cond_PCs_array)
    PC_array = np.hstack(PC_array)    
    
    #Prepare data for linear regression
    X=PC_array
    X_dot=[]
    X_dot_bins=np.arange(0,np.size(X,1),PSTH_length)
    for cur_bin in X_dot_bins:
        cur_X_dot=np.diff(X[:,cur_bin:cur_bin+PSTH_length],1,1)
        cur_X_dot = signal.savgol_filter(cur_X_dot, window_length=21, polyorder=3, mode="nearest")
        cur_X_dot=np.column_stack((cur_X_dot,cur_X_dot[:,-1])) #duplicate last column to keep dimensions equals
        X_dot.append(cur_X_dot)
    X_dot=np.hstack(X_dot)
        
    # Linear regression without external inputs
    y_pred_array=[]
    #fig, ax = plt.subplots(len(np.arange(n_PCs)))
    for regression_index in np.arange(n_PCs):
        #calculate the fit for the given PC
        cur_X= pd.DataFrame(np.transpose(X))
        y=pd.DataFrame(X_dot[regression_index,:],columns=['y'])
        regr = linear_model.LinearRegression()
        regr.fit_intercept=False
        #update current r square in the array
        r_sq = regr.fit(cur_X, y)
        r_sq = regr.score(cur_X, y)
        r_square_array[PCInx,regression_index]=r_sq
        
        y_pred=regr.predict(cur_X)
        y_pred_array.append(y_pred)
        #Plot the predicted  x_dot by the linear regression as a function of x_dot
        # ax[regression_index].scatter(X_dot[regression_index,:],y_pred_array[regression_index])
        # ax[regression_index].set_xlabel('x_dot')
        # ax[regression_index].set_ylabel('M*x')
    #fig.tight_layout()
    #plt.show()

    # Plot X_dot and the predictied x_dot
        if n_PCs==5:
            X_dot_bins=np.arange(0,np.size(X,1),PSTH_length*len(directions))
            fig, ax = plt.subplots(len(task_array),2)
            for bin_inx,cur_bin in enumerate(X_dot_bins):
                ax[bin_inx,0].plot(np.transpose(X_dot[regression_index,cur_bin:cur_bin+PSTH_length*len(directions)]))
                ax[bin_inx,0].plot(y_pred[cur_bin:cur_bin+PSTH_length*len(directions)])
                ax[bin_inx,1].plot(PC_array[regression_index,cur_bin:cur_bin+PSTH_length*len(directions)])
                ax[bin_inx,0].legend(['real','predicted'],fontsize=6)
                ax[bin_inx,0].set_title(condition_array[bin_inx],fontsize=6)
                ax[bin_inx,0].tick_params(axis='both', which='major', labelsize=6)
                ax[bin_inx,0].tick_params(axis='both', which='minor', labelsize=6)
                ax[bin_inx,1].tick_params(axis='both', which='major', labelsize=6)
                ax[bin_inx,1].tick_params(axis='both', which='minor', labelsize=6)
                for ii in np.arange(0,PSTH_length*len(directions),PSTH_length):
                  ax[bin_inx,0].axvline(ii,linestyle='dashed',color='black') 
                  ax[bin_inx,1].axvline(ii,linestyle='dashed',color='black') 

            fig.suptitle('PC'+str(regression_index+1))  
            plt.tight_layout()
            plt.show()
            
            #PC weights
            # fig, ax = plt.subplots(3)
            # ax[0].plot((PCs_cond[0,:]),color='r')
            # ax[1].plot((PCs_cond[1,:]),color='g')
            # ax[2].plot((PCs_cond[2,:]),color='b')
            # ax[0].set_ylim((-0.5,0.5))
            # ax[1].set_ylim((-0.5,0.5))
            # ax[2].set_ylim((-0.5,0.5))           
            # fig.suptitle('PC weights')
            # plt.show()
            

      
  
# save_path="C:/Users/Owner/Documents/Matan/Mati's Lab/population analysis paper/figure8_dynamics/"
# mdic = {"cond_flat_PSTH":cond_flat_PSTH}
# savemat(save_path+"cond_flat_PSTH"+ ".mat", mdic)
#%% Plot r square
 
# set width of bar
barWidth = 0.15
fig = plt.subplots(figsize =(12, 8))
 
# set height of bar
PCs1 = list(r_square_array[:,0])
PCs2 = list(r_square_array[:,1])
PCs3 = list(r_square_array[:,2])
PCs4 = list(r_square_array[:,3])
 
# Set position of bar on X axis
br1 = np.arange(len(PCs2))
br2 = [x + barWidth for x in br1]
br3 = [x + barWidth for x in br2]
br4 = [x + barWidth for x in br3]
 
# Make the plot
plt.bar(br1, PCs1, color ='r', width = barWidth, label ='PC1')
plt.bar(br2, PCs2, color ='g', width = barWidth, label ='PC2')
plt.bar(br3, PCs3, color ='b', width = barWidth,label ='PC3')
plt.bar(br4, PCs4, color ='c', width = barWidth, label ='PC4')

# Adding Xticks
plt.xlabel('number of PCs', fontweight ='bold', fontsize = 15)
plt.ylabel('r^2', fontweight ='bold', fontsize = 15)
plt.xticks([r + barWidth for r in range(len(PCs2))],['2PCs', '3PCs', '4PCs'])
 
plt.legend()
plt.show()
#%% Build the external inputs

# #The inputs are target velocity (H and V) abnd position and also the presence of fixation cue


# #target velocity
# #pursuit
# pursuitTargetVel=[]
# RT_delay=50#ms
# ramp_delay=30 #ms
# pursuit_vel=20
# PSTH_length_left=PSTH_length-np.size(np.zeros(RT_delay))-np.size(np.linspace(0,pursuit_vel,ramp_delay))
# pursuitTargetHvel=np.hstack([np.zeros(RT_delay),np.linspace(0,pursuit_vel,ramp_delay),np.ones(PSTH_length_left)*pursuit_vel])  #we simulate a ramp for the taget velocity. 50 ms reaction time and then the velocity increase linearly during 30 ms up to 20deg/s
# pursuitTargetVvel=np.zeros([PSTH_length])
# #suppression
# suppressionTargetVel=[]
# suppressionTargetHvel=np.hstack([np.zeros(RT_delay),np.linspace(0,pursuit_vel,ramp_delay),np.ones(PSTH_length_left)*pursuit_vel])  #we simulate a ramp for the taget velocity. 50 ms reaction time and then the velocity increase linearly during 30 ms up to 20deg/s
# suppressionTargetVvel=np.zeros([PSTH_length])
# #saccade
# saccadeTargetHvel=np.zeros([len(directions)*PSTH_length])
# saccadeTargetVvel=np.zeros([len(directions)*PSTH_length])

# #target position
# #pursuit 
# pursuitTargetPos=[]
# PSTH_length_left=PSTH_length-np.size(np.zeros(RT_delay))
# pursuit_edge=pursuit_vel*PSTH_length_left/1000
# pursuitTargetHpos=np.hstack([np.zeros(RT_delay),np.linspace(0,pursuit_edge,PSTH_length_left)])
# pursuitTargetVpos=np.zeros([PSTH_length])
# #suppression
# suppressionTargetPos=[]
# suppressionTargetHpos=np.hstack([np.zeros(RT_delay),np.linspace(0,pursuit_edge,PSTH_length_left)])
# suppressionTargetVpos=np.zeros([PSTH_length])
# #saccade
# saccadeTargetPos=[]
# saccadeTargetHpos=np.zeros([PSTH_length])+10
# saccadeTargetVpos=np.zeros([PSTH_length])

# #fixation cue
# saccadeFixationCue=np.zeros([len(directions)*PSTH_length])
# pursuitFixationCue=np.zeros([len(directions)*PSTH_length])
# supressionFixationCue=np.ones([len(directions)*PSTH_length])


# #Rotation
# #Find relevant direction
# for cur_dir in directions:
#     cur_dir_rad=math.radians(cur_dir)
#     rotation_matrix=np.array([[math.cos(cur_dir_rad),-math.sin(cur_dir_rad)],[math.sin(cur_dir_rad),math.cos(cur_dir_rad)]])
    
#     cur_saccadeTargetPos=np.matmul(rotation_matrix,np.vstack([saccadeTargetHpos,saccadeTargetVpos]))
#     saccadeTargetPos.append(cur_saccadeTargetPos)
    
#     curPursuitTargetVel=np.matmul(rotation_matrix,np.vstack([pursuitTargetHvel,pursuitTargetVvel]))
#     pursuitTargetVel.append(curPursuitTargetVel)
    
#     curPursuitTargetPos=np.matmul(rotation_matrix,np.vstack([pursuitTargetHpos,pursuitTargetVpos]))
#     pursuitTargetPos.append(curPursuitTargetPos)
    
#     curSuppressionTargetVel=np.matmul(rotation_matrix,np.vstack([suppressionTargetHvel,suppressionTargetVvel]))
#     suppressionTargetVel.append(curSuppressionTargetVel)
    
#     curSupressionTargetPos=np.matmul(rotation_matrix,np.vstack([suppressionTargetHpos,suppressionTargetVpos]))
#     suppressionTargetPos.append(curSupressionTargetPos)
    
# #Target position vector
# targetPosConds=np.hstack([np.hstack(saccadeTargetPos),np.hstack(pursuitTargetPos),np.hstack(suppressionTargetPos)])
# targetHpos=targetPosConds[0,:]
# targetVpos=targetPosConds[1,:]

# #Target velocity vector
# targetVelConds=np.hstack([np.vstack([saccadeTargetHvel,saccadeTargetVvel]),np.hstack(pursuitTargetVel),np.hstack(suppressionTargetVel)])
# targetHvel=targetPosConds[0,:]
# targetVvel=targetPosConds[1,:]

# #fixation cue vector
# targetPosConds=np.hstack([saccadeFixationCue,pursuitFixationCue,supressionFixationCue])

# #%% PCA for each condition
# shuffle_option=0
# #indexes of cell non significant for all the conditions
# non_reactive_cells_inx=np.where(np.sum(sig_pop_array,axis=0)==0)[0]

# #remove rows with nan
# nan_inxs=np.array([])
# nan_inxs=np.where(np.isnan(population_flat_PSTH_PCA).any(axis=0))[0]    
# nan_inxs=np.hstack((nan_inxs,non_reactive_cells_inx))
# nan_inxs=np.unique(nan_inxs)
# nan_inxs=[int(x) for x  in nan_inxs]

# #remove cells signififant for late cue cells
# # late_cue_cells_inxs=[cell_inx for cell_inx,cell_ID in enumerate(cell_list) if (cell_ID) in cue_cells ]
# # nan_inxs=np.hstack((nan_inxs,non_reactive_cells_inx,late_cue_cells_inxs)) 
# # nan_inxs=np.unique(nan_inxs)
# # nan_inxs=[int(x) for x  in nan_inxs]

# n_cells=len(cell_list)-len(nan_inxs)
# #n_PCs=3#number of PCs


# min_PCs=2
# max_PCs=5
# N_PCs=max_PCs-min_PCs
# scaling_array=[0,0.01,0.1,1]
# for a3_inx,a3 in enumerate(scaling_array):
#     fig, ax = plt.subplots(len(scaling_array),len(scaling_array))
#     for a1_inx,a1 in enumerate(scaling_array):
#         for a2_inx,a2 in enumerate(scaling_array):
#             r_square_array=np.empty([N_PCs,max_PCs-1])
#             r_square_array[:]=np.nan #r square for linear fit. Each row is for a number of PCs, each column is the r square for a given PC
#             for PCInx,n_PCs in enumerate(np.arange(min_PCs,max_PCs)): #loop across a number of total PCs
#                 PCs_array=np.empty([len(task_array),n_PCs,n_cells])
#                 exp_var_array=np.empty([len(task_array),n_PCs])
#                 eigen_values_array=np.empty([len(task_array),n_PCs])
                
#                 #Finding PCS of each conditions
#                 for cond_inx in np.arange(len(task_array)):
#                     if PCA_conditions=='PCA_condition_apart':
#                         cond_flat_PSTH=population_flat_PSTH_PCA[cond_inx]
#                         cond_flat_PSTH=np.delete(cond_flat_PSTH,nan_inxs,axis=0)
                
#                     elif PCA_conditions=='PCA_condition_together':
#                         cond_flat_PSTH=np.hstack(population_flat_PSTH_PCA)
#                         cond_flat_PSTH=np.delete(cond_flat_PSTH,nan_inxs,axis=0)
#                         cell_means=np.mean(cond_flat_PSTH,axis=1)
#                         cell_means2=np.tile(cell_means, (np.size(cond_flat_PSTH,1),1))
#                         cond_flat_PSTH=cond_flat_PSTH-np.transpose(cell_means2) #remove average across cells    
#                     #remove the mean of each time point
#                     cond_flat_PSTH=cond_flat_PSTH-np.mean(cond_flat_PSTH,axis=0)#zero-mean
#                     pca = PCA(n_components=n_PCs)
#                     pca.fit(np.transpose(cond_flat_PSTH))
#                     exp_var_cond=pca.explained_variance_ratio_
#                     eigen_values=pca.explained_variance_
#                     PCs_cond=pca.components_
#                     PCs_array[cond_inx,:,:]=PCs_cond
#                     exp_var_array[cond_inx,:]=exp_var_cond
#                     eigen_values_array[cond_inx,:]=eigen_values
                    
#                 #Create PC_array a numpy array where each row is the projection of a PSTH on a PC
#                 #The PSTH is number of conditions* number of directions * trial length
#                 timecourse=np.arange(win_begin_PSTH,win_end_PSTH)
#                 PC_array=[]
#                 for cond_inx in np.arange(len(task_array)):
#                     cond_PCs_array=np.empty([n_PCs,len(directions)*PSTH_length])
#                     cond_PCs_array[:]=np.nan
#                     for PC_inx in np.arange(n_PCs): #we project the activity of the current condition on the current PC 
#                         #choose the PC from cond_inx
#                         cur_PC=PCs_array[cond_inx,PC_inx,:]
#                         #choose a PSTH fron cond_inx
#                         cond_flat_PSTH=population_flat_PSTH[cond_inx]
#                         cond_flat_PSTH=np.delete(cond_flat_PSTH,nan_inxs,axis=0)
#                         #project the neural activity on the PC
#                         data_proj_PC=np.matmul(np.transpose(cond_flat_PSTH),cur_PC)
#                         cond_PCs_array[PC_inx,:]=data_proj_PC
#                     PC_array.append(cond_PCs_array)
#                 PC_array = np.hstack(PC_array)    
                
#                 #Prepare data for linear regression
#                 X=PC_array
#                 X_dot=[]
#                 X_dot_bins=np.arange(0,np.size(X,1),PSTH_length)
#                 for cur_bin in X_dot_bins:
#                     cur_X_dot=np.diff(X[:,cur_bin:cur_bin+PSTH_length],1,1)
#                     cur_X_dot = signal.savgol_filter(cur_X_dot, window_length=21, polyorder=3, mode="nearest")
#                     cur_X_dot=np.column_stack((cur_X_dot,cur_X_dot[:,-1])) #duplicate last column to keep dimensions equals
#                     X_dot.append(cur_X_dot)
#                 X_dot=np.hstack(X_dot)
#                 X_dot=X_dot-a2*np.tile(targetHvel,(np.size(X_dot,0),1))-a2*np.tile(targetVvel,(np.size(X_dot,0),1))-a1*np.tile(targetHpos,(np.size(X_dot,0),1))-a1*np.tile(targetVpos,(np.size(X_dot,0),1))-a3*np.tile(targetPosConds,(np.size(X_dot,0),1))
                    
#                 # Linear regression without external inputs
#                 y_pred_array=[]
#                 #fig, ax = plt.subplots(len(np.arange(n_PCs)))
#                 for regression_index in np.arange(n_PCs):
#                     #calculate the fit for the given PC
#                     cur_X= pd.DataFrame(np.transpose(X))
#                     y=pd.DataFrame(X_dot[regression_index,:],columns=['y'])
#                     regr = linear_model.LinearRegression()
#                     regr.fit_intercept=False
#                     #update current r square in the array
#                     r_sq = regr.fit(cur_X, y)
#                     r_sq = regr.score(cur_X, y)
#                     r_square_array[PCInx,regression_index]=r_sq
                    
#                     y_pred=regr.predict(cur_X)
#                     y_pred_array.append(y_pred)
#                     #Plot the predicted  x_dot by the linear regression as a function of x_dot
#                     # ax[regression_index].scatter(X_dot[regression_index,:],y_pred_array[regression_index])
#                     # ax[regression_index].set_xlabel('x_dot')
#                     # ax[regression_index].set_ylabel('M*x')
#                 #fig.tight_layout()
#                 #plt.show()
                
#                 # Plot r square
         
#             # set width of bar
#             barWidth = 0.15
             
#             # set height of bar
#             PCs1 = list(r_square_array[:,0])
#             PCs2 = list(r_square_array[:,1])
#             PCs3 = list(r_square_array[:,2])
#             PCs4 = list(r_square_array[:,3])
    
             
#             # Set position of bar on X axis
#             br1 = np.arange(len(PCs2))
#             br2 = [x + barWidth for x in br1]
#             br3 = [x + barWidth for x in br2]
#             br4 = [x + barWidth for x in br3]
             
#             # Make the plot
#             ax[a1_inx,a2_inx].bar(br1, PCs1, color ='r', width = barWidth,label ='PC1')
#             ax[a1_inx,a2_inx].bar(br2, PCs2, color ='g', width = barWidth,label ='PC2')
#             ax[a1_inx,a2_inx].bar(br3, PCs3, color ='b', width = barWidth, label ='PC3')
#             ax[a1_inx,a2_inx].bar(br4, PCs4, color ='m', width = barWidth, label ='PC4')
#             # Adding Xticks
    
#             ax[a1_inx,a2_inx].set_xticks([r + barWidth for r in range(len(PCs2))],['2PCs', '3PCs', '4PCs'], fontsize = 6)
#             # We change the fontsize of minor ticks label 
#             ax[a1_inx,a2_inx].tick_params(axis='both', which='major', labelsize=6)
#             ax[a1_inx,a2_inx].tick_params(axis='both', which='minor', labelsize=6)
#             ax[a1_inx,a2_inx].set_title('a2='+str(a2)+' a1='+str(a1), fontsize = 6)
#             ax[a1_inx,a2_inx].set_ylim((0,0.5))
#             ax[a1_inx,a2_inx].axhline(0.25,color='k',linestyle='--',linewidth=0.5)
#     fig.suptitle('a3='+str(a3))        
#     plt.tight_layout()
#     plt.show()