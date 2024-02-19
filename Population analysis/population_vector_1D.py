# -*- coding: utf-8 -*-
#in this script PD for populaiton vector is calculated based on the fit and is not one of the eight directions
#We calculated the population vector angle and not the projection on horizontal and verttical axis


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

sig_pop_array=[]
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
    
    # Check whether cell is significant for the event relative to baseline
    dictFilterTrials_test_active = {'dir':'filterOff', 'trial_name':'v20a', 'fail':0}
    try:
        check_sig_cell_active=cur_cell_task.check_main_effect_motion_vs_baseline('motion_onset',dictFilterTrials_test_active,crit_value=0.01)
    except:
        check_sig_cell_active=False

    dictFilterTrials_test_passive = {'dir':'filterOff', 'trial_name':'v20p', 'fail':0}
    try:
        check_sig_cell_passive=cur_cell_task.check_main_effect_motion_vs_baseline('motion_onset',dictFilterTrials_test_passive,crit_value=0.01)
    except:
        check_sig_cell_passive=False

    sig_pop_array.append(check_sig_cell_active and check_sig_cell_passive)
    #sig_pop_array.append(check_sig_cell_active )
    
   
    #PD
    dictFilterTrials_PD_active = {'dir':'filterOff', 'trial_name':'v20a', 'fail':0,'trial_inxs':PD_active_inxs}
    PD_active=cur_cell_task.get_exact_PD(window_PD,dictFilterTrials_PD_active)
    
    dictFilterTrials_PD_passive = {'dir':'filterOff', 'trial_name':'v20p', 'fail':0,'trial_inxs':PD_passive_inxs}    
    PD_passive=cur_cell_task.get_exact_PD(window_PD,dictFilterTrials_PD_passive)

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
    
    cur_cell_dict['PD_active']=PD_active
    cur_cell_dict['PD_passive']=PD_passive
    cur_cell_dict['PSTH_active']=cell_PSTH_array_active  
    cur_cell_dict['PSTH_passive']=cell_PSTH_array_passive  
    cells_dict_array.append(cur_cell_dict)

        
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
#%% Population vector calculation
cur_monkey='ya'

# PSTH active
n_cells=-1
# List to store firing rate vectors
firing_rate_vectors = []
active_behavior_x_vector=[]
active_behavior_y_vector=[]
for cell_inx,neuron_data in enumerate(cells_dict_array):
    cell_ID=cell_list[cell_inx]
    if cur_monkey=='ya':
        if cell_ID<8229:
            continue
    elif cur_monkey=='fi':
        if cell_ID>8229:
            continue
    cell_pd = neuron_data['PD_active']
    cell_rates = np.nanmean(neuron_data['PSTH_active'],1)
    if sig_pop_array[cell_inx]==False or any(np.isnan(cell_rates)) or np.isnan(cell_pd):
        continue
    n_cells=n_cells+1
    cell_vector = np.array([rate * np.exp(1j * np.radians(cell_pd)) for rate in cell_rates])
    firing_rate_vectors.append(cell_vector)
    
    active_behavior_x_vector.append(active_behaviour_x[:,cell_inx])
    active_behavior_y_vector.append(active_behaviour_y[:,cell_inx])
population_vector = np.sum(firing_rate_vectors, axis=0)
population_vector_degrees_PSTH_active = np.mod(np.degrees(np.angle(population_vector)),360)
population_vector_magnitude_PSTH_active = np.abs(population_vector)

np.sum(population_vector_magnitude_PSTH_active**2)
# PSTH passive
n_cells=-1
# List to store firing rate vectors
firing_rate_vectors = []
passive_behavior_x_vector=[]
passive_behavior_y_vector=[]
for cell_inx,neuron_data in enumerate(cells_dict_array):
    cell_ID=cell_list[cell_inx]
    if cur_monkey=='ya':
        if cell_ID<8229:
            continue
    elif cur_monkey=='fi':
        if cell_ID>8229:
            continue
    cell_pd = neuron_data['PD_active']
    cell_rates = np.nanmean(neuron_data['PSTH_passive'],1)
    if sig_pop_array[cell_inx]==False or any(np.isnan(cell_rates)) or np.isnan(cell_pd):
        continue
    n_cells=n_cells+1
    cell_vector = np.array([rate * np.exp(1j * np.radians(cell_pd)) for rate in cell_rates])
    firing_rate_vectors.append(cell_vector)
    passive_behavior_x_vector.append(passive_behaviour_x[:,cell_inx])
    passive_behavior_y_vector.append(passive_behaviour_y[:,cell_inx])
# Calculate the population vector
population_vector = np.sum(firing_rate_vectors, axis=0)
population_vector_degrees_PSTH_passive = np.mod(np.degrees(np.angle(population_vector)),360)
population_vector_magnitude_PSTH_passive = np.abs(population_vector)


#Plot population vector for active and passive data
fig = plt.figure()
ax = fig.add_subplot(111, projection='polar')
ax.set_rticks([])  # Less radial ticks
angles_rad = np.radians(population_vector_degrees_PSTH_active)
ax.plot(angles_rad, population_vector_magnitude_PSTH_active, marker='o')
angles_rad = np.radians(population_vector_degrees_PSTH_passive)
ax.plot(angles_rad, population_vector_magnitude_PSTH_passive, marker='o')
ax.legend(['active','suppression'])
ax.set_title("Population vector")
plt.show()


#calculate attenuation - how many percentage of active psth magnitude vector is the magnitude vector of passive psth in average 
pop_vector_attenuation=(np.sum(population_vector_magnitude_PSTH_passive**2))**0.5/(np.sum(population_vector_magnitude_PSTH_active**2))**0.5


#Behaviour
active_behavior_x_mean=np.nanmean(np.array(active_behavior_x_vector),axis=0)
active_behavior_y_mean=np.nanmean(np.array(active_behavior_y_vector),axis=0)
passive_behavior_x_mean=np.nanmean(np.array(passive_behavior_x_vector),axis=0)
passive_behavior_y_mean=np.nanmean(np.array(passive_behavior_y_vector),axis=0)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(active_behavior_x_mean, active_behavior_y_mean, marker='o')
ax.plot(passive_behavior_x_mean, passive_behavior_y_mean, marker='o')
ax.legend(['active','suppression'])
ax.set_title("Change in behaviour")
plt.show()

#behaviour attenuation
def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

(rho_active,phi_active)=cart2pol(active_behavior_x_mean,active_behavior_y_mean)
(rho_passive,phi_passive)=cart2pol(passive_behavior_x_mean,passive_behavior_y_mean)
behaviour_attenuation=(np.sum(rho_passive**2))**0.5/(np.sum(rho_active**2))**0.5


save_path="C:/Users/Owner/Documents/Matan/Mati's Lab/population analysis paper/figure5_active_passive/"
mdic = {"behaviour_active_x_"+cur_monkey:active_behavior_x_mean,"behaviour_active_y_"+cur_monkey:active_behavior_y_mean,"behaviour_passive_x_"+cur_monkey:passive_behavior_x_mean,"behaviour_passive_y_"+cur_monkey:passive_behavior_y_mean}
savemat(save_path+"behavior_active_supression_"+cur_monkey+".mat", mdic)

save_path="C:/Users/Owner/Documents/Matan/Mati's Lab/population analysis paper/figure5_active_passive/"
mdic = {"population_vector_degrees_PSTH_active_"+cur_monkey:population_vector_degrees_PSTH_active,"population_vector_magnitude_PSTH_active_"+cur_monkey:population_vector_magnitude_PSTH_active,
"population_vector_degrees_PSTH_passive_"+cur_monkey:population_vector_degrees_PSTH_passive,"population_vector_magnitude_PSTH_passive_"+cur_monkey:population_vector_magnitude_PSTH_passive}
savemat(save_path+"population_vector_"+cur_monkey+".mat", mdic)