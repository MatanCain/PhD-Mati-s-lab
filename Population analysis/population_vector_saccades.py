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
cell_saccade_list=os.listdir(cell_task_py_folder+'8dir_saccade_100_25') #list of strings
cell_list=[x for x in cell_saccade_list if x in cell_pursuit_list]
cell_list=[int(item) for item in cell_list] #list of ints
cutoff_cell=8229 #cutoff between yasmin and fiona
yasmin_cell_list=[x for x in cell_list if x>cutoff_cell]
fiona_cell_list=[x for x in cell_list if x<=cutoff_cell]
cell_list=cell_list



directions=[0,45,90,135,180,225,270,315]
window_PD={"timePoint":'motion_onset','timeBefore':0,'timeAfter':350}
SMOOTH_EDGE=200
win_before_PSTH=-200
win_after_PSTH=550
window_PSTH={"timePoint":'motion_onset','timeBefore':win_before_PSTH,'timeAfter':win_after_PSTH}
SMOOTH_PSTH_EDGE=200
cells_dict_array=[]
for cell_inx,cell_ID in enumerate(cell_list):
    cur_task='8dir_active_passive_interleaved_100_25'
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
    #saccades trials
    dictFilterTrials_PD_saccades = {'dir':'filterOff', 'trial_name':'v20p', 'fail':0}    
    trials_df_saccades=cur_cell_task.filtTrials(dictFilterTrials_PD_saccades)
    inx_list=np.array(trials_df_saccades.index)
    np.random.shuffle(inx_list)
    group_size=int(np.floor(np.size(inx_list)/2))
    PD_saccades_inxs=inx_list[0:group_size]
    PSTH_saccades_inxs=inx_list[-group_size:]
    
    #PD
    dictFilterTrials_PD_active = {'dir':'filterOff', 'trial_name':'v20a', 'fail':0,'trial_inxs':PD_active_inxs}
    PD_active=cur_cell_task.get_PD(window_PD,dictFilterTrials_PD_active)
    
    dictFilterTrials_PD_saccades = {'dir':'filterOff', 'trial_name':'v20p', 'fail':0,'trial_inxs':PD_saccades_inxs}    
    PD_saccades=cur_cell_task.get_PD(window_PD,dictFilterTrials_PD_saccades)

    dictFilterTrials_PSTH_active = {'dir':'filterOff', 'trial_name':'v20a', 'fail':0,'trial_inxs':PSTH_active_inxs}
    dictFilterTrials_PSTH_saccades = {'dir':'filterOff', 'trial_name':'v20p', 'fail':0,'trial_inxs':PSTH_saccades_inxs}    
   
    #Average PSTH
    temp=cur_cell_task.filtTrials(dictFilterTrials_PSTH_active)
    average_PSTH_active=cur_cell_task.PSTH(window_PSTH,dictFilterTrials_PSTH_active) #PSTH of cell for given condition and direction
    average_PSTH_active=average_PSTH_active[SMOOTH_PSTH_EDGE:-SMOOTH_PSTH_EDGE]

    average_PSTH_saccades=cur_cell_task.PSTH(window_PSTH,dictFilterTrials_PSTH_saccades) #PSTH of cell for given condition and direction
    average_PSTH_saccades=average_PSTH_saccades[SMOOTH_PSTH_EDGE:-SMOOTH_PSTH_EDGE]        
   
    #PSTH
    cell_PSTH_array_active=np.empty([len(directions),win_after_PSTH-win_before_PSTH-2*SMOOTH_PSTH_EDGE])
    cell_PSTH_array_saccades=np.empty([len(directions),win_after_PSTH-win_before_PSTH-2*SMOOTH_PSTH_EDGE])

    for dir_inx,cur_dir in enumerate(directions):
        dictFilterTrials_PSTH_active['dir']=cur_dir
        dictFilterTrials_PSTH_saccades['dir']=cur_dir
        average_PSTH_dir_active=cur_cell_task.PSTH(window_PSTH,dictFilterTrials_PSTH_active) #PSTH of cell for given condition and direction
        average_PSTH_dir_active=average_PSTH_dir_active[SMOOTH_PSTH_EDGE:-SMOOTH_PSTH_EDGE]
        average_PSTH_dir_active=average_PSTH_dir_active-average_PSTH_active
        cell_PSTH_array_active[dir_inx,:]=average_PSTH_dir_active

        average_PSTH_dir_saccades=cur_cell_task.PSTH(window_PSTH,dictFilterTrials_PSTH_saccades) #PSTH of cell for given condition and direction
        average_PSTH_dir_saccades=average_PSTH_dir_saccades[SMOOTH_PSTH_EDGE:-SMOOTH_PSTH_EDGE]
        average_PSTH_dir_saccades=average_PSTH_dir_saccades-average_PSTH_saccades
        cell_PSTH_array_saccades[dir_inx,:]=average_PSTH_dir_saccades
    
    cur_cell_dict['PD_active']=calculate_PD_2d(PD_active)  
    cur_cell_dict['PD_saccades']=calculate_PD_2d(PD_saccades)  
    cur_cell_dict['PSTH_active']=cell_PSTH_array_active  
    cur_cell_dict['PSTH_saccades']=cell_PSTH_array_saccades
    
    
    #saccades
    cur_task='8dir_saccade_100_25'
    cur_cell_task=load_cell_task(cell_task_py_folder,cur_task,cell_ID)
    #divide trials to PD trials and PSTH trials to prevent double dip (bias in the population vector)
    #active trials
    dictFilterTrials_PD_saccades = {'dir':'filterOff', 'trial_name':'filterOff', 'fail':0}
    trials_df_saccades=cur_cell_task.filtTrials(dictFilterTrials_PD_saccades)
    inx_list=np.array(trials_df_saccades.index)
    np.random.shuffle(inx_list)
    group_size=int(np.floor(np.size(inx_list)/2))
    PD_saccades_inxs=inx_list[0:group_size]
    PSTH_saccades_inxs=inx_list[-group_size:]

    
    #PD
    dictFilterTrials_PD_saccades = {'dir':'filterOff', 'trial_name':'filterOff', 'fail':0,'trial_inxs':PD_saccades_inxs}
    PD_saccades=cur_cell_task.get_PD(window_PD,dictFilterTrials_PD_saccades)
    
    dictFilterTrials_PSTH_saccades = {'dir':'filterOff', 'trial_name':'filterOff', 'fail':0,'trial_inxs':PSTH_saccades_inxs}
   
    #Average PSTH
    temp=cur_cell_task.filtTrials(dictFilterTrials_PSTH_saccades)
    try:
        average_PSTH_saccades=cur_cell_task.PSTH(window_PSTH,dictFilterTrials_PSTH_saccades) #PSTH of cell for given condition and direction
        average_PSTH_saccades=average_PSTH_saccades[SMOOTH_PSTH_EDGE:-SMOOTH_PSTH_EDGE]

    except:
        average_PSTH_saccades=np.empty([win_after_PSTH-win_before_PSTH-2*SMOOTH_PSTH_EDGE])
        average_PSTH_saccades[:]=np.nan
     
    #PSTH
    cell_PSTH_array_saccades=np.empty([len(directions),win_after_PSTH-win_before_PSTH-2*SMOOTH_PSTH_EDGE])

    for dir_inx,cur_dir in enumerate(directions):
        dictFilterTrials_PSTH_saccades['dir']=cur_dir
        try:
            average_PSTH_dir_saccades=cur_cell_task.PSTH(window_PSTH,dictFilterTrials_PSTH_saccades) #PSTH of cell for given condition and direction
            average_PSTH_dir_saccades=average_PSTH_dir_saccades[SMOOTH_PSTH_EDGE:-SMOOTH_PSTH_EDGE]
            average_PSTH_dir_saccades=average_PSTH_dir_saccades-average_PSTH_saccades
        except:
            average_PSTH_dir_saccades=np.empty([win_after_PSTH-win_before_PSTH-2*SMOOTH_PSTH_EDGE])
            average_PSTH_dir_saccades[:]=np.nan
        cell_PSTH_array_saccades[dir_inx,:]=average_PSTH_dir_saccades
              

    cur_cell_dict['PD_saccades']=calculate_PD_2d(PD_saccades)  
    cur_cell_dict['PSTH_saccades']=cell_PSTH_array_saccades 
    cells_dict_array.append(cur_cell_dict)      
        
#%% Population vector calculation

cur_monkey='fi' #'both','ya','fi'
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

# PD-saccades PSTH-saccades
PVx_saccades_saccades=np.zeros([len(directions),win_after_PSTH-win_before_PSTH-2*SMOOTH_PSTH_EDGE])
PVy_saccades_saccades=np.zeros([len(directions),win_after_PSTH-win_before_PSTH-2*SMOOTH_PSTH_EDGE])
n_cells=0
for cell_inx,cell_ID in enumerate(cell_list):

    cur_cell_dict=cells_dict_array[cell_inx]
    if np.isnan(cur_cell_dict['PD_saccades'][0]) or any(any(row) for row in np.isnan(cur_cell_dict['PSTH_saccades'])) :
        continue
    else:
        n_cells=n_cells+1
        PVx_saccades_saccades=PVx_saccades_saccades+cur_cell_dict['PD_saccades'][0]*cur_cell_dict['PSTH_saccades']       
        PVy_saccades_saccades=PVy_saccades_saccades+cur_cell_dict['PD_saccades'][1]*cur_cell_dict['PSTH_saccades']
PVx_saccades_saccades=PVx_saccades_saccades/n_cells
PVy_saccades_saccades=PVy_saccades_saccades/n_cells
ax[0,1].plot(timecourse,np.transpose(PVx_saccades_saccades))
ax[1,1].plot(timecourse,np.transpose(PVy_saccades_saccades))
ax[0,1].set_title('PD-saccades PSTH-saccades'+'-X',fontsize=5)
ax[1,1].set_title('PD-saccades PSTH-saccades'+'-Y',fontsize=5)
ax[0,1].tick_params(axis='both', labelsize=6, pad=0)
ax[1,1].tick_params(axis='both', labelsize=6, pad=0)
ax[0,1].axhline(0, color='black')
ax[1,1].axhline(0, color='black')
ax[0,1].set_ylim(-EDGE,EDGE)
ax[1,1].set_ylim(-EDGE,EDGE)

# PD-active PSTH-saccades
PVx_active_saccades=np.zeros([len(directions),win_after_PSTH-win_before_PSTH-2*SMOOTH_PSTH_EDGE])
PVy_active_saccades=np.zeros([len(directions),win_after_PSTH-win_before_PSTH-2*SMOOTH_PSTH_EDGE])
n_cells=0
for cell_inx,cell_ID in enumerate(cell_list):
    cur_cell_dict=cells_dict_array[cell_inx]
    if np.isnan(cur_cell_dict['PD_active'][0]) or any(any(row) for row in np.isnan(cur_cell_dict['PSTH_saccades'])) :
        continue
    else:
        n_cells=n_cells+1
        PVx_active_saccades=PVx_active_saccades+cur_cell_dict['PD_active'][0]*cur_cell_dict['PSTH_saccades']       
        PVy_active_saccades=PVy_active_saccades+cur_cell_dict['PD_active'][1]*cur_cell_dict['PSTH_saccades']
PVx_active_saccades=PVx_active_saccades/n_cells
PVy_active_saccades=PVy_active_saccades/n_cells
ax[0,2].plot(timecourse,np.transpose(PVx_active_saccades))
ax[1,2].plot(timecourse,np.transpose(PVy_active_saccades))
ax[0,2].set_title('PD-active PSTH-saccades'+'-X',fontsize=5)
ax[1,2].set_title('PD-active PSTH-saccades'+'-Y',fontsize=5)
ax[0,2].tick_params(axis='both', labelsize=6, pad=0)
ax[1,2].tick_params(axis='both', labelsize=6, pad=0)
ax[0,2].axhline(0, color='black')
ax[1,2].axhline(0, color='black')
ax[0,2].set_ylim(-EDGE,EDGE)
ax[1,2].set_ylim(-EDGE,EDGE)

# PD-saccades PSTH-active
PVx_saccades_active=np.zeros([len(directions),win_after_PSTH-win_before_PSTH-2*SMOOTH_PSTH_EDGE])
PVy_saccades_active=np.zeros([len(directions),win_after_PSTH-win_before_PSTH-2*SMOOTH_PSTH_EDGE])
n_cells=0
for cell_inx,cell_ID in enumerate(cell_list):
    cur_cell_dict=cells_dict_array[cell_inx]
    if np.isnan(cur_cell_dict['PD_saccades'][0]) or any(any(row) for row in np.isnan(cur_cell_dict['PSTH_active'])) :
        continue
    else:
        n_cells=n_cells+1
        PVx_saccades_active=PVx_saccades_active+cur_cell_dict['PD_saccades'][0]*cur_cell_dict['PSTH_active']       
        PVy_saccades_active=PVy_saccades_active+cur_cell_dict['PD_saccades'][1]*cur_cell_dict['PSTH_active']

PVx_saccades_active=PVx_saccades_active/n_cells
PVy_saccades_active=PVy_saccades_active/n_cells

ax[0,3].plot(timecourse,np.transpose(PVx_saccades_active))
ax[1,3].plot(timecourse,np.transpose(PVy_saccades_active))
ax[0,3].set_title('PD-saccades PSTH-active'+'-X',fontsize=5)
ax[1,3].set_title('PD-saccades PSTH-active'+'-Y',fontsize=5)
ax[0,3].tick_params(axis='both', labelsize=6, pad=0)
ax[1,3].tick_params(axis='both', labelsize=6, pad=0)
ax[0,3].axhline(0, color='black')
ax[1,3].axhline(0, color='black')
ax[0,3].set_ylim(-EDGE,EDGE)
ax[1,3].set_ylim(-EDGE,EDGE)


fig.suptitle('Population vectors')
fig.tight_layout()
plt.show()

save_path="C:/Users/Owner/Documents/Matan/Mati's Lab/population analysis paper/figure5_active_saccade/"
mdic = {"PVx_active_active":PVx_active_active,"PVx_active_saccades":PVx_active_saccades,"PVx_saccades_active":PVx_saccades_active,"PVx_saccades_saccades":PVx_active_active}
savemat(save_path+"PVx_"+cur_monkey+".mat", mdic)
mdic = {"PVy_active_active":PVy_active_active,"PVy_active_saccades":PVy_active_saccades,"PVy_saccades_active":PVy_saccades_active,"PVy_saccades_saccades":PVy_active_active}
savemat(save_path+"PVy_"+cur_monkey+".mat", mdic)



# QUantification of the difference between Population vectors

plt.scatter(np.mean(PVx_active_active,axis=1),np.mean(PVx_active_saccades,axis=1))
plt.scatter(np.mean(PVy_active_active,axis=1),np.mean(PVy_active_saccades,axis=1))
plt.xlabel('PD active PSTH active')
plt.ylabel('PD active PSTH saccade')
plt.title('Correlation of population vectors')
plt.axline((0, 0), slope=1., color='black')
plt.axline((0, 0), slope=-1., color='black',linestyle='--')
plt.xlim([-1,1])
plt.ylim([-1,1])
plt.show()

save_path="C:/Users/Owner/Documents/Matan/Mati's Lab/population analysis paper/figure5_active_saccade/"
mdic = {"PVx_active_active_mean_"+cur_monkey:np.mean(PVx_active_active,axis=1),"PVy_active_active_mean_"+cur_monkey:np.mean(PVy_active_active,axis=1),"PVx_active_saccade_mean_"+cur_monkey:np.mean(PVx_active_saccades,axis=1),"PVy_active_saccade_mean_"+cur_monkey:np.mean(PVy_active_saccades,axis=1)}
savemat(save_path+"PV_scatter_"+cur_monkey+".mat", mdic)