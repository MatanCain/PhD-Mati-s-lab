# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 14:48:12 2022

@author: Owner
"""

#In this script we compare the PSTH of cells in the learning and inlearning learning blocks (relative to washout)


import os
os.chdir("C:/Users/Owner/Documents/Matan/Mati's Lab/FEF learning project/Data_Analyses/Code analyzes/basic classes")
import numpy as np
import math
import matplotlib as mpl
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
import pandas as pd
import pickle
from scipy.linalg import svd
import random
from sklearn.linear_model import LinearRegression
os.chdir("C:/Users/Owner/Documents/Matan/Mati's Lab/FEF learning project/Data_Analyses/Code analyzes/passive learning")
from omega_passive_learning_function import* 

# General parameters
path="C:/Users/Owner/Documents/Matan/Mati's Lab/FEF learning project/Data_Analyses"
os.chdir(path)
cell_db_file='fiona_cells_db.xlsx'
#get list of cells
cell_task_py_folder="units_task_python_two_monkeys/"

behaviour_db_excel="C:/Users/Owner/Documents/Matan/Mati's Lab/FEF learning project/Data_Analyses/fiona_behaviour_db.xlsx"
behaviour_db=pd.read_excel(behaviour_db_excel)

#import the behaviour data frame for passive learning
save_path="C:/Users/Owner/Documents/Matan/Mati's Lab/FEF learning project/Data_Analyses/Code analyzes/passive learning/"
filename=save_path+'/'+'behaviour_df_passive_learning'
infile = open(filename,'rb')
passive_learning_behaviour_df= pickle.load(infile)
infile.close()
#%% In this part of the script, we create list of PSTHS fot the different passivge learning blocks and washout
cur_event='motion_onset'
win_begin_PSTH=-200
win_end_PSTH=800
dir_change=250
timecourse=np.arange(win_begin_PSTH,win_end_PSTH)

learned_direction_array=['CW','CCW']
learning_task='Motor_learning'
motor_dict_array=[]

#passive trials
trial_type_learning='v20NS'

mapping_tasks=['8dir_active_passive_interleaved','8dir_active_passive_interleaved_100_25']
washout_task='washout_100_25_cue'
cutoff_cell=8229 #cutoff between yasmin and fiona

window_begin=100
window_end=300

alternative_test='greater'
win_begin_test=100
win_end_test=300
crit_learning_value=0.05

for learned_direction in learned_direction_array:
    for learning_task2_inx,learning_task2 in enumerate([learning_task+'_'+learned_direction,learning_task+'_'+learned_direction+'_100_25_cue']): 
        cell_learning_list=os.listdir(cell_task_py_folder+learning_task2) #list of strings
        mapping_task=mapping_tasks[learning_task2_inx]
        cell_list=[int(item) for item in cell_learning_list if int(item)>cutoff_cell] #list of ints
        for cell_ID in cell_list:

            cur_cell_learning=load_cell_task(cell_task_py_folder,learning_task2,cell_ID) # load learning cell_task
            trials_df_learning=cur_cell_learning.trials_df
            
            #separate into blocks
            # Separate the trials_df to different blocks (each session includes all the dishabitaution blocks recorded during a given day)
            trials_list=trials_df_learning.loc[:]['filename_name'].tolist()
            trial_number_np=np.array([int(x.split('.',1)[1]) for x in trials_list ])
            block_end_indexes=np.where(np.diff(trial_number_np)>=80)[0]#find all trials where next trials is at least with a step of 80 trials (end of blocks)
            block_begin_indexes=np.append(np.array(0),block_end_indexes+1)#(beginning of blocks- just add 1 to end of blocks and also the first trial
            block_end_indexes=np.append(block_end_indexes,len(trials_df_learning)-1)#add the last trial to end of blocks
                   

            #for each learning block:
            for block_inx,(begin_inx,end_inx) in enumerate(zip(block_begin_indexes,block_end_indexes)):

                learning_block_df=trials_df_learning.iloc[np.arange(begin_inx,end_inx+1)]
                
                #Base and learned directions
                block_base_dir=learning_block_df.iloc[0]['screen_rotation']#Find base direction of curren block
                #select the relevant direction as learned direction in the previous washout block
                if learned_direction=='CW':
                    washout_learned_dir='270'
                    learned_direction_int=(block_base_dir-90)%360
                elif learned_direction=='CCW':
                    washout_learned_dir='90'
                    learned_direction_int=(block_base_dir+90)%360
              
                #washout block befire the learning blocks
                try:
                    cur_cell_washout=load_cell_task(cell_task_py_folder,washout_task,cell_ID)
                except:
                    continue
                session=cur_cell_learning.getSession()
                block_row=behaviour_db.loc[(behaviour_db['behaviour_session']==session) & (behaviour_db['Task']==learning_task2) & (behaviour_db['screen_rotation']==block_base_dir)].index[0] #first row of the learning block in the behavior db
                
                file_begin_washout=behaviour_db.iloc[block_row-1]['file_begin'] #find file begin of washout block preceding the learning block
                file_end_washout=behaviour_db.iloc[block_row-1]['file_end']
                               
               
                try:
                    dictFilterTrials_learning = {'dir':'filterOff', 'trial_name':trial_type_learning, 'fail':0, 'block_begin_end':[begin_inx,end_inx],'even_odd_rows':'odd'}

                    FR_learning_baseline=cur_cell_learning.get_mean_FR_event(dictFilterTrials_learning,'motion_onset',window_pre=-800,window_post=0)
                    FR_learning=cur_cell_learning.get_mean_FR_event(dictFilterTrials_learning,'motion_onset',window_pre=win_begin_test,window_post=win_end_test)
                    stat,p_learning=stats.wilcoxon(FR_learning_baseline, FR_learning,alternative=alternative_test)
                    if p_learning>crit_learning_value:
                        continue
                except:
                    continue

                try:
                    dictFilterTrials_washout_base = { 'files_begin_end':[file_begin_washout,file_end_washout], 'fail':0,'trial_name':'d0'+trial_type_learning}
                    dictFilterTrials_learning = {'dir':'filterOff', 'trial_name':trial_type_learning, 'fail':0, 'block_begin_end':[begin_inx,end_inx],'even_odd_rows':'even'}

                    FR_washout=cur_cell_washout.get_mean_FR_event(dictFilterTrials_washout_base,'motion_onset',window_pre=window_begin,window_post=window_end)
                    FR_learning=cur_cell_learning.get_mean_FR_event(dictFilterTrials_learning,'motion_onset',window_pre=window_begin,window_post=window_end)
                except:
                    continue

                
                #Stability washout - check whether the correlation in FR before MO is correlated between washout and subsequent learning block
                try:
                    dictFilterTrials_washout_base_stability = { 'dir':'filterOff', 'fail':0,'files_begin_end':[file_begin_washout,file_end_washout],'trial_name':'d0'+trial_type_learning}
                    FR_learning_washout_baseline=cur_cell_washout.get_mean_FR_event(dictFilterTrials_washout_base_stability,'motion_onset',window_pre=-800,window_post=0)
                    dictFilterTrials_learning_stability = {'dir':'filterOff', 'fail':0, 'block_begin_end':[begin_inx,end_inx], 'trial_name':trial_type_learning}
                    FR_learning_baseline2=cur_cell_learning.get_mean_FR_event(dictFilterTrials_learning_stability,'motion_onset',window_pre=-800,window_post=0)
                    r,p_stability_learning=stats.mannwhitneyu(FR_learning_baseline2, FR_learning_washout_baseline)
                    if p_stability_learning<0.05:
                        continue
                except:
                    continue                     

                block_dict=dict.fromkeys(['cell_ID', 'base direction', 'learned direction','FR learning','FR washout'])
                block_dict['cell_ID']=cell_ID
                block_dict['base direction']=block_base_dir
                block_dict['learned direction']=learned_direction
                block_dict['FR learning']=FR_learning           
                block_dict['FR washout']=FR_washout

                motor_dict_array.append(block_dict)



#%% leanring vs dishabituation in mapping            
                                                                                                                                                                                                                                                                                                        
sig_motor_array=np.array([x for x in motor_dict_array ]  )

legend_size=8

learning_FR=[]
washout_FR=[]
for block in sig_motor_array:
    learning_FR.append(np.nanmean(block['FR learning']))
    washout_FR.append(np.nanmean(block['FR washout']))

    
r,p=stats.wilcoxon(washout_FR, learning_FR,alternative=alternative_test)    
plt.scatter(learning_FR,washout_FR)
plt.axline([0,0],[1,1],color='black')
plt.ylabel('dishabituation')
plt.xlabel('learning')
plt.title(str(len(sig_motor_array))+' cells p='+str(round(p,5)))
plt.show()