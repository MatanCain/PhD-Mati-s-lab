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
cell_task_py_folder="units_task_two_monkeys_python_kinematics/"   

############################
def calculate_PD_2d(PD):
    if PD==0:
        PD_coor=(1,0)
    elif PD==45:
        PD_coor=((2**0.5)/2,(2**0.5)/2)
    elif PD==90:
        PD_coor=(0,1)
    elif PD==135:
        PD_coor=(-(2**0.5)/2,(2**0.5)/2)
    elif PD==180:
        PD_coor=(-1,0)
    elif PD==225:
        PD_coor=(-(2**0.5)/2,-(2**0.5)/2)
    elif PD==270:
        PD_coor=(0,-1)
    elif PD==315:
        PD_coor=(+(2**0.5)/2,-(2**0.5)/2)        
    else:
        PD_coor=(np.nan,np.nan)
    return PD_coor
############################

#%% In this part of the script, we create population_tuning_curve_noNan a 3D array (n_conds*n_dirs*n_cells) with the tuning curve of all the cells recorded during mapping

cell_pursuit_list=os.listdir(cell_task_py_folder+'8dir_active_passive_interleaved_100_25') #list of strings
cell_pursuit_list=[int(item) for item in cell_pursuit_list] #list of ints
cutoff_cell=8229 #cutoff between yasmin and fiona
yasmin_cell_list=[x for x in cell_pursuit_list if x>cutoff_cell]
fiona_cell_list=[x for x in cell_pursuit_list if x<=cutoff_cell]
cell_list=cell_pursuit_list

#For each cell get the PD and the average PSTH after average subtraction
cur_task='8dir_active_passive_interleaved_100_25'

#PSTH parameters
directions=[0,45,90,135,180,225,270,315]
window_PD={"timePoint":'motion_onset','timeBefore':0,'timeAfter':350}
win_before_PSTH=-200
win_after_PSTH=550
window_PSTH={"timePoint":'motion_onset','timeBefore':win_before_PSTH,'timeAfter':win_after_PSTH}
SMOOTH_PSTH_EDGE=200

#parameters for behaviour
saccade_motion_parameter=0
window_pre_behaviour=100
window_post_behaviour=350


cells_dict_array=[]

active_behaviour=np.empty([len(directions),len(cell_list)])
active_behaviour[:]=np.nan
passive_behaviour=np.empty([len(directions),len(cell_list)])
passive_behaviour[:]=np.nan
active_behaviour_x=np.empty([len(directions),len(cell_list)])
active_behaviour_x[:]=np.nan
passive_behaviour_x=np.empty([len(directions),len(cell_list)])
passive_behaviour_x[:]=np.nan
active_behaviour_y=np.empty([len(directions),len(cell_list)])
active_behaviour_y[:]=np.nan
passive_behaviour_y=np.empty([len(directions),len(cell_list)])
passive_behaviour_y[:]=np.nan


for cell_inx,cell_ID in enumerate(cell_list):
    cur_cell_task=load_cell_task(cell_task_py_folder,cur_task,cell_ID)
    cur_cell_dict={}
    
    #divide trials to PD trials and PSTH trials to prevent double dip (bias in the population vector)
    #active trials
    dictFilterTrials_PD_active = {'dir':'filterOff', 'trial_name':'v20a', 'fail':0}
    trials_df_active=cur_cell_task.filtTrials(dictFilterTrials_PD_active)
    inx_list=np.array(trials_df_active.index)
    np.random.shuffle(inx_list)
    group_size=int(np.floor(np.size(inx_list)/2))
    PD_active_inxs=inx_list[0:group_size]
    PSTH_active_inxs=inx_list[-group_size:]
    #passive trials
    dictFilterTrials_PD_passive = {'dir':'filterOff', 'trial_name':'v20p', 'fail':0}    
    trials_df_passive=cur_cell_task.filtTrials(dictFilterTrials_PD_passive)
    inx_list=np.array(trials_df_passive.index)
    np.random.shuffle(inx_list)
    group_size=int(np.floor(np.size(inx_list)/2))
    PD_passive_inxs=inx_list[0:group_size]
    PSTH_passive_inxs=inx_list[-group_size:]
    
    #PD
    dictFilterTrials_PD_active = {'dir':'filterOff', 'trial_name':'v20a', 'fail':0,'trial_inxs':PD_active_inxs}
    PD_active=cur_cell_task.get_PD(window_PD,dictFilterTrials_PD_active)
    
    dictFilterTrials_PD_passive = {'dir':'filterOff', 'trial_name':'v20p', 'fail':0,'trial_inxs':PD_passive_inxs}    
    PD_passive=cur_cell_task.get_PD(window_PD,dictFilterTrials_PD_passive)

    dictFilterTrials_PSTH_active = {'dir':'filterOff', 'trial_name':'v20a', 'fail':0,'trial_inxs':PSTH_active_inxs}
    dictFilterTrials_PSTH_passive = {'dir':'filterOff', 'trial_name':'v20p', 'fail':0,'trial_inxs':PSTH_passive_inxs}    
   
    #Average PSTH
    temp=cur_cell_task.filtTrials(dictFilterTrials_PSTH_active)
    average_PSTH_active=cur_cell_task.PSTH(window_PSTH,dictFilterTrials_PSTH_active) #PSTH of cell for given condition and direction
    average_PSTH_active=average_PSTH_active[SMOOTH_PSTH_EDGE:-SMOOTH_PSTH_EDGE]

    average_PSTH_passive=cur_cell_task.PSTH(window_PSTH,dictFilterTrials_PSTH_passive) #PSTH of cell for given condition and direction
    average_PSTH_passive=average_PSTH_passive[SMOOTH_PSTH_EDGE:-SMOOTH_PSTH_EDGE]        
   
    #PSTH
    cell_PSTH_array_active=np.empty([len(directions),win_after_PSTH-win_before_PSTH-2*SMOOTH_PSTH_EDGE])
    cell_PSTH_array_passive=np.empty([len(directions),win_after_PSTH-win_before_PSTH-2*SMOOTH_PSTH_EDGE])

    #for behaviour
    dictFilterTrials_Pos_passive={'trial_name':'v20p', 'fail':0,'saccade_motion':saccade_motion_parameter}
    dictFilterTrials_Pos_active={ 'trial_name':'v20a', 'fail':0 ,'saccade_motion':saccade_motion_parameter}
    for dir_inx,cur_dir in enumerate(directions):
        dictFilterTrials_PSTH_active['dir']=cur_dir
        dictFilterTrials_PSTH_passive['dir']=cur_dir
        average_PSTH_dir_active=cur_cell_task.PSTH(window_PSTH,dictFilterTrials_PSTH_active) #PSTH of cell for given condition and direction
        average_PSTH_dir_active=average_PSTH_dir_active[SMOOTH_PSTH_EDGE:-SMOOTH_PSTH_EDGE]
        average_PSTH_dir_active=average_PSTH_dir_active-average_PSTH_active
        cell_PSTH_array_active[dir_inx,:]=average_PSTH_dir_active

        average_PSTH_dir_passive=cur_cell_task.PSTH(window_PSTH,dictFilterTrials_PSTH_passive) #PSTH of cell for given condition and direction
        average_PSTH_dir_passive=average_PSTH_dir_passive[SMOOTH_PSTH_EDGE:-SMOOTH_PSTH_EDGE]
        average_PSTH_dir_passive=average_PSTH_dir_passive-average_PSTH_passive
        cell_PSTH_array_passive[dir_inx,:]=average_PSTH_dir_passive
        
        dictFilterTrials_Pos_passive['dir']=cur_dir
        dictFilterTrials_Pos_active['dir']=cur_dir
        try:
            trials_df_active=cur_cell_task.filtTrials(dictFilterTrials_Pos_active)  
            trials_df_passive=cur_cell_task.filtTrials(dictFilterTrials_Pos_passive)
            #change in behaviour in widow
            hPosChangeActive= trials_df_active.apply(lambda row: [(row.hPos[window_post_behaviour+row.motion_onset])-(row.hPos[window_pre_behaviour+row.motion_onset])],axis=1).to_list()
            hPosChangeActive = np.array([item for sublist in hPosChangeActive for item in sublist])
            vPosChangeActive = trials_df_active.apply(lambda row: [(row.vPos[window_post_behaviour+row.motion_onset])-(row.vPos[window_pre_behaviour+row.motion_onset])],axis=1).to_list()
            vPosChangeActive = np.array([item for sublist in vPosChangeActive for item in sublist])
            TotPosChangeActive=(hPosChangeActive**2+vPosChangeActive**2)**0.5
            active_behaviour[dir_inx,cell_inx]=(np.nanmean(TotPosChangeActive))#x and y together
            active_behaviour_x[dir_inx,cell_inx]=(np.nanmean(hPosChangeActive))#x and y apart
            active_behaviour_y[dir_inx,cell_inx]=(np.nanmean(vPosChangeActive))

    
            hPosChangePassive= trials_df_passive.apply(lambda row: [(row.hPos[window_post_behaviour+row.motion_onset])-(row.hPos[window_pre_behaviour+row.motion_onset])],axis=1).to_list()
            hPosChangePassive = np.array([item for sublist in hPosChangePassive for item in sublist])
            vPosChangePassive = trials_df_passive.apply(lambda row: [(row.vPos[window_post_behaviour+row.motion_onset])-(row.vPos[window_pre_behaviour+row.motion_onset])],axis=1).to_list()
            vPosChangePassive = np.array([item for sublist in vPosChangePassive for item in sublist])
            TotPosChangePassive=(hPosChangePassive**2+vPosChangePassive**2)**0.5
            passive_behaviour[dir_inx,cell_inx]=(np.nanmean(TotPosChangePassive))
            passive_behaviour_x[dir_inx,cell_inx]=(np.nanmean(hPosChangePassive))
            passive_behaviour_y[dir_inx,cell_inx]=(np.nanmean(vPosChangePassive))
        except:
            continue
    
    cur_cell_dict['PD_active']=calculate_PD_2d(PD_active)  
    cur_cell_dict['PD_passive']=calculate_PD_2d(PD_passive)  
    cur_cell_dict['PSTH_active']=cell_PSTH_array_active  
    cur_cell_dict['PSTH_passive']=cell_PSTH_array_passive  
    cells_dict_array.append(cur_cell_dict)

        
        
#%% Population vector calculation
cur_monkey='both' #'both','ya','fi'
if cur_monkey=='fi':
    cur_monkey_inxs=[cell_inx for cell_inx,cell_ID in enumerate(cell_list) if cell_ID in fiona_cell_list]
elif cur_monkey=='ya':
    cur_monkey_inxs=[cell_inx for cell_inx,cell_ID in enumerate(cell_list) if cell_ID in yasmin_cell_list]
else:
    cur_monkey_inxs=[cell_inx for cell_inx,cell_ID in enumerate(cell_list)]

fig, ax = plt.subplots(2,4)
# PD-active PSTH-active
PVx_active_active=np.zeros([len(directions),win_after_PSTH-win_before_PSTH-2*SMOOTH_PSTH_EDGE])
PVy_active_active=np.zeros([len(directions),win_after_PSTH-win_before_PSTH-2*SMOOTH_PSTH_EDGE])
n_cells=0
timecourse=np.arange(win_before_PSTH+SMOOTH_PSTH_EDGE,win_after_PSTH-SMOOTH_PSTH_EDGE)
EDGE=1.2
for cell_inx,cell_ID in enumerate(cell_list):
    
    if cur_monkey=='ya':
        if cell_ID<8229:
            continue
    elif cur_monkey=='fi':
        if cell_ID>8229:
            continue
        
    
    cur_cell_dict=cells_dict_array[cell_inx]
    if np.isnan(cur_cell_dict['PD_active'][0]) or any(any(row) for row in np.isnan(cur_cell_dict['PSTH_active'])) :
        continue
    else:
        n_cells=n_cells+1
        PVx_active_active=PVx_active_active+cur_cell_dict['PD_active'][0]*cur_cell_dict['PSTH_active']       
        PVy_active_active=PVy_active_active+cur_cell_dict['PD_active'][1]*cur_cell_dict['PSTH_active']

PVx_active_active=PVx_active_active/n_cells
PVy_active_active=PVy_active_active/n_cells
ax[0,0].plot(timecourse,np.transpose(PVx_active_active))
ax[1,0].plot(timecourse,np.transpose(PVy_active_active))
ax[0,0].legend(directions,prop = { "size": 5},handlelength=0.5)
ax[0,0].set_title('PD-active PSTH-active'+'-X',fontsize=5)
ax[1,0].set_title('PD-active PSTH-active'+'-Y',fontsize=5)
ax[0,0].tick_params(axis='both', labelsize=6, pad=0)
ax[1,0].tick_params(axis='both', labelsize=6, pad=0)
ax[0,0].axhline(0, color='black')
ax[1,0].axhline(0, color='black')
ax[0,0].set_ylim(-EDGE,EDGE)
ax[1,0].set_ylim(-EDGE,EDGE)

# PD-passive PSTH-passive
PVx_passive_passive=np.zeros([len(directions),win_after_PSTH-win_before_PSTH-2*SMOOTH_PSTH_EDGE])
PVy_passive_passive=np.zeros([len(directions),win_after_PSTH-win_before_PSTH-2*SMOOTH_PSTH_EDGE])
n_cells=0
for cell_inx,cell_ID in enumerate(cell_list):

    cur_cell_dict=cells_dict_array[cell_inx]
    if np.isnan(cur_cell_dict['PD_passive'][0]) or any(any(row) for row in np.isnan(cur_cell_dict['PSTH_passive'])) :
        continue
    else:
        n_cells=n_cells+1
        PVx_passive_passive=PVx_passive_passive+cur_cell_dict['PD_passive'][0]*cur_cell_dict['PSTH_passive']       
        PVy_passive_passive=PVy_passive_passive+cur_cell_dict['PD_passive'][1]*cur_cell_dict['PSTH_passive']
PVx_passive_passive=PVx_passive_passive/n_cells
PVy_passive_passive=PVy_passive_passive/n_cells
ax[0,1].plot(timecourse,np.transpose(PVx_passive_passive))
ax[1,1].plot(timecourse,np.transpose(PVy_passive_passive))
ax[0,1].set_title('PD-passive PSTH-passive'+'-X',fontsize=5)
ax[1,1].set_title('PD-passive PSTH-passive'+'-Y',fontsize=5)
ax[0,1].tick_params(axis='both', labelsize=6, pad=0)
ax[1,1].tick_params(axis='both', labelsize=6, pad=0)
ax[0,1].axhline(0, color='black')
ax[1,1].axhline(0, color='black')
ax[0,1].set_ylim(-EDGE,EDGE)
ax[1,1].set_ylim(-EDGE,EDGE)

# PD-active PSTH-passive
PVx_active_passive=np.zeros([len(directions),win_after_PSTH-win_before_PSTH-2*SMOOTH_PSTH_EDGE])
PVy_active_passive=np.zeros([len(directions),win_after_PSTH-win_before_PSTH-2*SMOOTH_PSTH_EDGE])
n_cells=0
for cell_inx,cell_ID in enumerate(cell_list):
    cur_cell_dict=cells_dict_array[cell_inx]
    if np.isnan(cur_cell_dict['PD_active'][0]) or any(any(row) for row in np.isnan(cur_cell_dict['PSTH_passive'])) :
        continue
    else:
        n_cells=n_cells+1
        PVx_active_passive=PVx_active_passive+cur_cell_dict['PD_active'][0]*cur_cell_dict['PSTH_passive']       
        PVy_active_passive=PVy_active_passive+cur_cell_dict['PD_active'][1]*cur_cell_dict['PSTH_passive']
PVx_active_passive=PVx_active_passive/n_cells
PVy_active_passive=PVy_active_passive/n_cells
ax[0,2].plot(timecourse,np.transpose(PVx_active_passive))
ax[1,2].plot(timecourse,np.transpose(PVy_active_passive))
ax[0,2].set_title('PD-active PSTH-passive'+'-X',fontsize=5)
ax[1,2].set_title('PD-active PSTH-passive'+'-Y',fontsize=5)
ax[0,2].tick_params(axis='both', labelsize=6, pad=0)
ax[1,2].tick_params(axis='both', labelsize=6, pad=0)
ax[0,2].axhline(0, color='black')
ax[1,2].axhline(0, color='black')
ax[0,2].set_ylim(-EDGE,EDGE)
ax[1,2].set_ylim(-EDGE,EDGE)

# PD-passive PSTH-active
PVx_passive_active=np.zeros([len(directions),win_after_PSTH-win_before_PSTH-2*SMOOTH_PSTH_EDGE])
PVy_passive_active=np.zeros([len(directions),win_after_PSTH-win_before_PSTH-2*SMOOTH_PSTH_EDGE])
n_cells=0
for cell_inx,cell_ID in enumerate(cell_list):
    cur_cell_dict=cells_dict_array[cell_inx]
    if np.isnan(cur_cell_dict['PD_passive'][0]) or any(any(row) for row in np.isnan(cur_cell_dict['PSTH_active'])) :
        continue
    else:
        n_cells=n_cells+1
        PVx_passive_active=PVx_passive_active+cur_cell_dict['PD_passive'][0]*cur_cell_dict['PSTH_active']       
        PVy_passive_active=PVy_passive_active+cur_cell_dict['PD_passive'][1]*cur_cell_dict['PSTH_active']

PVx_passive_active=PVx_passive_active/n_cells
PVy_passive_active=PVy_passive_active/n_cells

ax[0,3].plot(timecourse,np.transpose(PVx_passive_active))
ax[1,3].plot(timecourse,np.transpose(PVy_passive_active))
ax[0,3].set_title('PD-passive PSTH-active'+'-X',fontsize=5)
ax[1,3].set_title('PD-passive PSTH-active'+'-Y',fontsize=5)
ax[0,3].tick_params(axis='both', labelsize=6, pad=0)
ax[1,3].tick_params(axis='both', labelsize=6, pad=0)
ax[0,3].axhline(0, color='black')
ax[1,3].axhline(0, color='black')
ax[0,3].set_ylim(-EDGE,EDGE)
ax[1,3].set_ylim(-EDGE,EDGE)

fig.suptitle('Population vectors')
fig.tight_layout()
plt.show()

# save_path="C:/Users/Owner/Documents/Matan/Mati's Lab/population analysis paper/figure4_active_passive/"
# mdic = {"PVx_active_active":PVx_active_active,"PVx_active_passive":PVx_active_passive,"PVx_passive_active":PVx_passive_active,"PVx_passive_passive":PVx_active_active}
# savemat(save_path+"PVx"+cur_monkey+".mat", mdic)
# mdic = {"PVy_active_active":PVy_active_active,"PVy_active_passive":PVy_active_passive,"PVy_passive_active":PVy_passive_active,"PVy_passive_passive":PVy_active_active}
# savemat(save_path+"PVy"+cur_mokey+".mat", mdic)



# QUantification of the difference between Population vectors

plt.scatter(np.mean(PVx_active_active,axis=1),np.mean(PVx_active_passive,axis=1))
plt.scatter(np.mean(PVy_active_active,axis=1),np.mean(PVy_active_passive,axis=1))
plt.xlabel('PD active PSTH active')
plt.ylabel('PD active PSTH passive')
plt.title('Correlation of population vectors')
plt.axline((0, 0), slope=1., color='black')
plt.show()

save_path="C:/Users/Owner/Documents/Matan/Mati's Lab/population analysis paper/figure4_active_passive/"
mdic = {"PVx_active_active_mean_"+cur_monkey:np.mean(PVx_active_active,axis=1),"PVy_active_active_mean_"+cur_monkey:np.mean(PVy_active_active,axis=1),"PVx_active_passive_mean_"+cur_monkey:np.mean(PVx_active_passive,axis=1),"PVy_active_passive_mean_"+cur_monkey:np.mean(PVy_active_passive,axis=1)}
savemat(save_path+"PV_scatter_"+cur_monkey+".mat", mdic)
#%% Behaviour

active_behaviour_mean=np.nanmean(active_behaviour[:,cur_monkey_inxs],axis=1)
passive_behaviour_mean=np.nanmean(passive_behaviour[:,cur_monkey_inxs],axis=1)
active_behaviour_x_mean=np.nanmean(active_behaviour_x[:,cur_monkey_inxs],axis=1)
passive_behaviour_x_mean=np.nanmean(passive_behaviour_x[:,cur_monkey_inxs],axis=1)
active_behaviour_y_mean=np.nanmean(active_behaviour_y[:,cur_monkey_inxs],axis=1)
passive_behaviour_y_mean=np.nanmean(passive_behaviour_y[:,cur_monkey_inxs],axis=1)
#scatter behaviour active vs passive
plt.scatter(np.abs(active_behaviour_x_mean),np.abs(passive_behaviour_x_mean))
plt.scatter(np.abs(active_behaviour_y_mean),np.abs(passive_behaviour_y_mean))
plt.axline((0, 0), slope=1., color='black')
plt.xlabel('active')
plt.ylabel('passive')
plt.title('active vs passive behavior')
plt.xlim(0,3)
plt.ylim(0,3)
plt.show()

save_path="C:/Users/Owner/Documents/Matan/Mati's Lab/population analysis paper/figure4_active_passive/"
mdic = {"behaviour_active_x"+cur_monkey:active_behaviour_x_mean,"behaviour_active_y"+cur_monkey:active_behaviour_y_mean,"behaviour_passive_x"+cur_monkey:passive_behaviour_x_mean,"behaviour_passive_y"+cur_monkey:passive_behaviour_y_mean}
savemat(save_path+"scatter_behavior_"+cur_monkey+".mat", mdic)


#ratio behaviour
ratio_behaviour=(passive_behaviour_mean/active_behaviour_mean)

PV_active=(PVx_active_active**2+PVy_active_active**2)**0.5
PV_passive=(PVx_passive_passive**2+PVy_passive_passive**2)**0.5
PV_active=np.nanmean(PV_active[:,win_before_PSTH+SMOOTH_PSTH_EDGE+window_pre_behaviour:win_before_PSTH+SMOOTH_PSTH_EDGE+window_post_behaviour],axis=1)
PV_passive=np.nanmean(PV_passive[:,win_before_PSTH+SMOOTH_PSTH_EDGE+window_pre_behaviour:win_before_PSTH+SMOOTH_PSTH_EDGE+window_post_behaviour],axis=1)
ratio_neural=PV_passive/PV_active

plt.scatter(ratio_behaviour,ratio_neural)
plt.xlabel('ratio behaviour')
plt.ylabel('ratio PV neural')
plt.title('ratio behaviour vs ratio neural')
plt.axline((0, 0), slope=1., color='black')
plt.xlim(0,2)
plt.ylim(0,2)
plt.show()


