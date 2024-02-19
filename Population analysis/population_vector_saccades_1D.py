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
sig_pop_array=[]
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
    PD_active=cur_cell_task.get_exact_PD(window_PD,dictFilterTrials_PD_active)
    

    # Check whether cell is significant for the event relative to baseline
    dictFilterTrials_test_active = {'dir':'filterOff', 'trial_name':'v20a', 'fail':0}
    try:
        check_sig_cell_active=cur_cell_task.check_main_effect_motion_vs_baseline('motion_onset',dictFilterTrials_test_active,crit_value=0.01)
    except:
        check_sig_cell_active=False
        
    dictFilterTrials_PSTH_active = {'dir':'filterOff', 'trial_name':'v20a', 'fail':0,'trial_inxs':PSTH_active_inxs}
    #Average PSTH
    temp=cur_cell_task.filtTrials(dictFilterTrials_PSTH_active)
    average_PSTH_active=cur_cell_task.PSTH(window_PSTH,dictFilterTrials_PSTH_active) #PSTH of cell for given condition and direction
    average_PSTH_active=average_PSTH_active[SMOOTH_PSTH_EDGE:-SMOOTH_PSTH_EDGE]


    #PSTH
    cell_PSTH_array_active=np.empty([len(directions),win_after_PSTH-win_before_PSTH-2*SMOOTH_PSTH_EDGE])
    cell_PSTH_array_saccades=np.empty([len(directions),win_after_PSTH-win_before_PSTH-2*SMOOTH_PSTH_EDGE])

    for dir_inx,cur_dir in enumerate(directions):
        dictFilterTrials_PSTH_active['dir']=cur_dir
        average_PSTH_dir_active=cur_cell_task.PSTH(window_PSTH,dictFilterTrials_PSTH_active) #PSTH of cell for given condition and direction
        average_PSTH_dir_active=average_PSTH_dir_active[SMOOTH_PSTH_EDGE:-SMOOTH_PSTH_EDGE]
        average_PSTH_dir_active=average_PSTH_dir_active-average_PSTH_active
        cell_PSTH_array_active[dir_inx,:]=average_PSTH_dir_active


    
    cur_cell_dict['PD_active']=PD_active
    cur_cell_dict['PSTH_active']=cell_PSTH_array_active  
    
    
    #saccades
    cur_task='8dir_saccade_100_25'
    cur_cell_task=load_cell_task(cell_task_py_folder,cur_task,cell_ID)
    #divide trials to PD trials and PSTH trials to prevent double dip (bias in the population vector)
    dictFilterTrials_PD_saccades = {'dir':'filterOff', 'trial_name':'filterOff', 'fail':0}
    trials_df_saccades=cur_cell_task.filtTrials(dictFilterTrials_PD_saccades)
    inx_list=np.array(trials_df_saccades.index)
    np.random.shuffle(inx_list)
    group_size=int(np.floor(np.size(inx_list)/2))
    PD_saccades_inxs=inx_list[0:group_size]
    PSTH_saccades_inxs=inx_list[-group_size:]

    
    #PD
    dictFilterTrials_PD_saccades = {'dir':'filterOff', 'trial_name':'filterOff', 'fail':0,'trial_inxs':PD_saccades_inxs}
    PD_saccades=cur_cell_task.get_exact_PD(window_PD,dictFilterTrials_PD_saccades)

    # Check whether cell is significant for the event relative to baseline
    dictFilterTrials_test_saccade = {'dir':'filterOff', 'trial_name':'filterOff', 'fail':0}
    try:
        check_sig_cell_saccade=cur_cell_task.check_main_effect_motion_vs_baseline('motion_onset',dictFilterTrials_test_saccade,crit_value=0.01)
    except:
        check_sig_cell_saccade=False

    sig_pop_array.append(check_sig_cell_active and check_sig_cell_saccade)

        
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
              

    cur_cell_dict['PD_saccades']=PD_saccades
    cur_cell_dict['PSTH_saccades']=cell_PSTH_array_saccades 
    cells_dict_array.append(cur_cell_dict)      
        
#%% Population vector calculation
cur_monkey='both'
# PSTH active
n_cells=-1
# List to store firing rate vectors
firing_rate_vectors = []
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

# Calculate the population vector
population_vector = np.sum(firing_rate_vectors, axis=0)
population_vector_degrees_PSTH_active = np.mod(np.degrees(np.angle(population_vector)),360)

# PSTH saccade
n_cells=-1
# List to store firing rate vectors
firing_rate_vectors = []
for cell_inx,neuron_data in enumerate(cells_dict_array):
    cell_pd = neuron_data['PD_active']
    cell_rates = np.nanmean(neuron_data['PSTH_saccades'],1)
    
    if sig_pop_array[cell_inx]==False or any(np.isnan(cell_rates)) or np.isnan(cell_pd):
        continue

    n_cells=n_cells+1
    cell_vector = np.array([rate * np.exp(1j * np.radians(cell_pd)) for rate in cell_rates])
    firing_rate_vectors.append(cell_vector)

# Calculate the population vector
population_vector = np.sum(firing_rate_vectors, axis=0)
population_vector_degrees_PSTH_saccade = np.mod(np.degrees(np.angle(population_vector)),360)

#%% bar plot
# set width of bar
barWidth = 0.25
fig = plt.subplots(figsize =(12, 8))

# set height of bar
population_vector_degrees_PSTH_active = list(population_vector_degrees_PSTH_active)
population_vector_degrees_PSTH_saccade = list(population_vector_degrees_PSTH_saccade)
# Set position of bar on X axis
br1 = np.arange(len(population_vector_degrees_PSTH_active))
br2 = [x + barWidth for x in br1]
# Make the plot
plt.bar(br1, population_vector_degrees_PSTH_active, color ='b', width = barWidth, edgecolor ='grey', label ='FR active')
plt.bar(br2, population_vector_degrees_PSTH_saccade, color ='r', width = barWidth, edgecolor ='grey', label ='FR saccade')
# Adding Xticks
plt.xlabel('directions', fontweight ='bold', fontsize = 15)
plt.ylabel('Estimated directions', fontweight ='bold', fontsize = 15)
plt.xticks([r + barWidth for r in range(len(population_vector_degrees_PSTH_active))],directions)
plt.legend()
plt.show()