# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 14:48:12 2022

@author: Owner
"""

#This script shows that cells that change their FR in 100:300 between dishabituation and congruent learning are not consistently tuned.
#To show that we use the leanring block
#Positive control - congruent blocks between 300-800
#Negative control - incongruent block 300-800

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
win_begin_PSTH=-500
win_end_PSTH=800
win_begin_PSTH2=-500
win_end_PSTH2=800
dir_change=250
timecourse=np.arange(win_begin_PSTH,win_end_PSTH)

learned_direction_array=['CW','CCW']
congruent_tasks=[]
learning_tasks=['fixation_right_probes','fixation_wrong_probes']
cong_dict_array=[]
incong_dict_array=[]
learning_dict=[cong_dict_array,incong_dict_array]

#active trials
# trial_type_learning='v20NS'
# trial_type_mapping='v20a'
#passive trials
trial_type_learning='v20S'
trial_type_mapping='v20p'
mapping_tasks=['8dir_active_passive_interleaved','8dir_active_passive_interleaved_100_25']
dishabituation_tasks=['Dishabituation','Dishabituation_100_25_cue']
washout_task='washout_100_25_cue'
crit_value_learning=0.05
cutoff_cell=8229 #cutoff between yasmin and fiona

window_begin=100
window_end=300

cur_alternative='less'

color_array=['tab:blue','tab:orange']
for learning_task_inx,learning_task in enumerate(learning_tasks):
    cur_dict_array=learning_dict[learning_task_inx]
    psth_learning_array=[]
    psth_dis_array=[]

    for learned_direction in learned_direction_array:
        for learning_task2_inx,learning_task2 in enumerate([learning_task+'_'+learned_direction,learning_task+'_'+learned_direction+'_100_25_cue']):       
            cell_learning_list=os.listdir(cell_task_py_folder+learning_task2) #list of strings
            mapping_task=mapping_tasks[learning_task2_inx]
            cell_list=[int(item) for item in cell_learning_list] #list of ints        
            for cell_ID in cell_list:

                if cell_ID<cutoff_cell:
                    #set the name of the dishabituation task according to the cell number and monkey
                    dishabituation_task=dishabituation_tasks[learning_task2_inx]
    
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
                            learning_direction_int=(block_base_dir-90)%360    
                        elif learned_direction=='CCW':
                            learning_direction_int=(block_base_dir+90)%360
                        learning_direction_null_int=int((learning_direction_int+180)%360)

                            
                        #dishabituation block before the learning block
                        try:
                            cur_cell_dishabituation=load_cell_task(cell_task_py_folder,dishabituation_task,cell_ID)
                        except:
                            continue
                        session=cur_cell_learning.getSession()
                        block_row=behaviour_db.loc[(behaviour_db['behaviour_session']==session) & (behaviour_db['Task']==learning_task2) & (behaviour_db['screen_rotation']==block_base_dir)].index[0] #first row of the learning block in the behavior db
                        file_begin_dishabituation=behaviour_db.iloc[block_row-1]['file_begin'] #find file begin of dishabituation block preceding the learning block
                        file_end_dishabituation=behaviour_db.iloc[block_row-1]['file_end']
                                       
                    
                        #Test if FR during learning block  is different than during dishabituation in critical period            
                        try:
                            dictFilterTrials_learning_test = {'dir':'filterOff', 'trial_name':trial_type_learning, 'fail':0, 'block_begin_end':[begin_inx,end_inx]}
                            dictFilterTrials_dishabituation_test = { 'dir':'filterOff', 'fail':0,'files_begin_end':[file_begin_dishabituation,file_end_dishabituation],'trial_name':'d0'+trial_type_learning}
                            FR_learning_test=cur_cell_learning.get_mean_FR_event(dictFilterTrials_learning_test,'motion_onset',window_pre=window_begin,window_post=window_end)
                            
                            stat,p_learning=stats.spearmanr(np.arange(np.size(FR_learning_test)),FR_learning_test,alternative=cur_alternative)
                            if p_learning>crit_value_learning or np.size(FR_learning_test)!=72:
                                continue
                            else:
                                window_PSTH={"timePoint":'motion_onset','timeBefore':win_begin_PSTH,'timeAfter':win_end_PSTH}
                                psth_learning=cur_cell_learning.PSTH(window_PSTH,dictFilterTrials_learning_test)
                                psth_dis=cur_cell_dishabituation.PSTH(window_PSTH,dictFilterTrials_dishabituation_test)
                        except:
                            continue                            
     
                            
                        #stability dishabituation - check whether the correlation in FR before MO is correlated between washout and subsequent learning block                        
                        try:
                            dictFilterTrials_dishabituation_stability = { 'dir':'filterOff', 'fail':0,'files_begin_end':[file_begin_dishabituation,file_end_dishabituation],'trial_name':'d0'+trial_type_learning}
                            FR_learning_dishabituation_baseline=cur_cell_dishabituation.get_mean_FR_event(dictFilterTrials_dishabituation_stability,'motion_onset',window_pre=-800,window_post=0)
                            dictFilterTrials_learning_stability = {'dir':'filterOff', 'fail':0, 'block_begin_end':[begin_inx,end_inx], 'trial_name':trial_type_learning}
                            FR_learning_baseline=cur_cell_learning.get_mean_FR_event(dictFilterTrials_learning_stability,'motion_onset',window_pre=-800,window_post=0)
                            stability_array=np.array(FR_learning_dishabituation_baseline.to_list()+FR_learning_baseline.to_list())
                            #stability based on correlation
                            r,p_stability_learning=stats.pearsonr(np.arange(len(stability_array)),stability_array)
                            #stability based on ttest
                            stat,p_stability_learning2=stats.mannwhitneyu(FR_learning_dishabituation_baseline, FR_learning_baseline)
                            if p_stability_learning<0.05:
                                continue
                        except:
                            continue
                        
                        #Behaviour - Find the learned velocity in the congruent and incongruent blocks
                        if learning_task2_inx==0:
                            cong_task='fixation_right_probes_'+learned_direction
                            incong_task='fixation_wrong_probes_'+learned_direction
                        elif learning_task2_inx==1:
                            cong_task='fixation_right_probes_'+learned_direction+'_100_25_cue'
                            incong_task='fixation_wrong_probes_'+learned_direction+'_100_25_cue'
                        
                        try:
                            cong_learned_vel=passive_learning_behaviour_df.loc[(passive_learning_behaviour_df['session']==session) & (passive_learning_behaviour_df['task']==cong_task) & (passive_learning_behaviour_df['base_direction']==block_base_dir)& (passive_learning_behaviour_df['learned_direction']==learned_direction)]['learned velocity'].to_list()
                            cong_learned_vel=np.array(cong_learned_vel)  
                            incong_learned_vel=passive_learning_behaviour_df.loc[(passive_learning_behaviour_df['session']==session) & (passive_learning_behaviour_df['task']==incong_task) & (passive_learning_behaviour_df['base_direction']==block_base_dir)& (passive_learning_behaviour_df['learned_direction']==learned_direction)]['learned velocity'].to_list()
                            incong_learned_vel=np.array(incong_learned_vel) 
                            # Behavioural test - learned velocity is higher in congruent than incongruent block
                            cong_learned_vel=cong_learned_vel[0]
                            cong_learned_vel_mean=np.nanmean(cong_learned_vel[-win_begin_PSTH2+100:-win_begin_PSTH2+300])
                            incong_learned_vel=incong_learned_vel[0]
                            incong_learned_vel_mean=np.nanmean(incong_learned_vel[-win_begin_PSTH2+100:-win_begin_PSTH2+300])
                            behaviour_effect=cong_learned_vel_mean>incong_learned_vel_mean
                        except:
                            behaviour_effect=0
                        if not behaviour_effect:
                            continue
                                                
                        psth_dis_array.append(psth_dis-np.nanmean(psth_dis[0:-win_begin_PSTH]))
                        psth_learning_array.append(psth_learning-np.nanmean(psth_learning[0:-win_begin_PSTH]))
                    
#####################################################################################################################################################################                    
                elif cell_ID>cutoff_cell:    
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
                            learning_direction_int=(block_base_dir-90)%360
    
                        elif learned_direction=='CCW':
                            learning_direction_int=(block_base_dir+90)%360
                        learning_direction_null_int=int((learning_direction_int+180)%360)

    
                            
                        #washout block before the learning block
                        try:
                            cur_cell_dishabituation=load_cell_task(cell_task_py_folder,washout_task,cell_ID)
                        except:
                            continue
                        session=cur_cell_learning.getSession()
                        block_row=behaviour_db.loc[(behaviour_db['behaviour_session']==session) & (behaviour_db['Task']==learning_task2) & (behaviour_db['screen_rotation']==block_base_dir)].index[0] #first row of the learning block in the behavior db
                        file_begin_washout=behaviour_db.iloc[block_row-1]['file_begin'] #find file begin of washout block preceding the learning block
                        file_end_washout=behaviour_db.iloc[block_row-1]['file_end']
                                       
                    
                        #Test if FR during learning block  is different than during dishabituation in critical period            
                        try:
                            dictFilterTrials_learning_test = {'dir':'filterOff', 'trial_name':trial_type_learning, 'fail':0, 'block_begin_end':[begin_inx,end_inx]}
                            dictFilterTrials_dishabituation_test = { 'dir':'filterOff', 'fail':0,'files_begin_end':[file_begin_dishabituation,file_end_dishabituation],'trial_name':'d0'+trial_type_learning}
                            FR_learning_test=cur_cell_learning.get_mean_FR_event(dictFilterTrials_learning_test,'motion_onset',window_pre=window_begin,window_post=window_end)
                            
                            stat,p_learning=stats.spearmanr(np.arange(np.size(FR_learning_test)),FR_learning_test,alternative=cur_alternative)
                            if p_learning>crit_value_learning or np.size(FR_learning_test)!=72:
                                continue
                            else:
                                window_PSTH={"timePoint":'motion_onset','timeBefore':win_begin_PSTH,'timeAfter':win_end_PSTH}
                                psth_learning=cur_cell_learning.PSTH(window_PSTH,dictFilterTrials_learning_test)
                                psth_dis=cur_cell_dishabituation.PSTH(window_PSTH,dictFilterTrials_dishabituation_test)
                        except:
                            continue   
                            
                        #stability washout - check whether the correlation in FR before MO is correlated between washout and subsequent learning block                        
                        try:
                            dictFilterTrials_washout_stability = { 'dir':'filterOff', 'fail':0,'files_begin_end':[file_begin_washout,file_end_washout],'trial_name':'d0'+trial_type_learning}
                            FR_learning_washout_baseline=cur_cell_dishabituation.get_mean_FR_event(dictFilterTrials_washout_stability,'motion_onset',window_pre=-800,window_post=0)
                            dictFilterTrials_learning_stability = {'dir':'filterOff', 'fail':0, 'block_begin_end':[begin_inx,end_inx], 'trial_name':trial_type_learning}
                            FR_learning_baseline=cur_cell_learning.get_mean_FR_event(dictFilterTrials_learning_stability,'motion_onset',window_pre=-800,window_post=0)
                            #stability based on ttest
                            stat,p_stability_learning=stats.mannwhitneyu(FR_learning_washout_baseline, FR_learning_baseline)
                            if p_stability_learning<0.05:
                                continue
                        except:
                            continue
                        
                        #Behaviour - Find the learned velocity in the congruent and incongruent blocks
                        if learning_task2_inx==0:
                            cong_task='fixation_right_probes_'+learned_direction
                            incong_task='fixation_wrong_probes_'+learned_direction
                        elif learning_task2_inx==1:
                            cong_task='fixation_right_probes_'+learned_direction+'_100_25_cue'
                            incong_task='fixation_wrong_probes_'+learned_direction+'_100_25_cue'
                        
                        try:
                            cong_learned_vel=passive_learning_behaviour_df.loc[(passive_learning_behaviour_df['session']==session) & (passive_learning_behaviour_df['task']==cong_task) & (passive_learning_behaviour_df['base_direction']==block_base_dir)& (passive_learning_behaviour_df['learned_direction']==learned_direction)]['learned velocity'].to_list()
                            cong_learned_vel=np.array(cong_learned_vel)  
                            incong_learned_vel=passive_learning_behaviour_df.loc[(passive_learning_behaviour_df['session']==session) & (passive_learning_behaviour_df['task']==incong_task) & (passive_learning_behaviour_df['base_direction']==block_base_dir)& (passive_learning_behaviour_df['learned_direction']==learned_direction)]['learned velocity'].to_list()
                            incong_learned_vel=np.array(incong_learned_vel) 
                            # Behavioural test - learned velocity is higher in congruent than incongruent block
                            cong_learned_vel=cong_learned_vel[0]
                            cong_learned_vel_mean=np.nanmean(cong_learned_vel[-win_begin_PSTH2+100:-win_begin_PSTH2+300])
                            incong_learned_vel=incong_learned_vel[0]
                            incong_learned_vel_mean=np.nanmean(incong_learned_vel[-win_begin_PSTH2+100:-win_begin_PSTH2+300])
                            behaviour_effect=cong_learned_vel_mean>incong_learned_vel_mean
                        except:
                            behaviour_effect=0
                        if not behaviour_effect:
                            continue
                        psth_dis_array.append(psth_dis-np.nanmean(psth_dis[0:-win_begin_PSTH]))
                        psth_learning_array.append(psth_learning-np.nanmean(psth_learning[0:-win_begin_PSTH]))
        
    psth_learning_mean=np.nanmean(psth_learning_array,axis=0)
    psth_dis_mean=np.nanmean(psth_dis_array,axis=0)

    plt.plot(timecourse,psth_learning_mean,color=color_array[learning_task_inx]) 
    plt.plot(timecourse,psth_dis_mean,linestyle='dashed',color=color_array[learning_task_inx])
    
    plt.axvline(100,color='black')               
    plt.axvline(300,color='black')               

plt.show()
