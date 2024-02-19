# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 14:48:12 2022

@author: Owner
"""

#In this script we compare the passive trials cells recorded in both congruent and incongruent blocks
#we compare PSTH (half trials) and learning curves


import os
os.chdir("C:/Users/Owner/Documents/Matan/Mati's Lab/FEF learning project/Data_Analyses/Code analyzes/basic classes")
import numpy as np
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
import neuron_class
from neuron_class import cell_task,filtCells,load_cell_task,smooth_data,PSTH_across_cells
from neuron_class import*

from scipy.io import savemat
from scipy.stats.stats import pearsonr
import scipy.stats as stats
from scipy.ndimage import gaussian_filter1d
from sklearn.decomposition import PCA
from scipy.stats import kruskal
import pandas as pd
import pickle
import random
from sklearn.linear_model import LinearRegression
import scipy
import pingouin as pg
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

#%% Find cells recorded in a motor block and the matching conngruent block
cur_event='motion_onset'
cong_task='fixation_right_probes'
incong_task='fixation_wrong_probes'
learned_direction_array=['CW','CCW']
window_PSTH={"timePoint":'motion_onset','timeBefore':-500,'timeAfter':800}          
timecourse=np.arange(window_PSTH['timeBefore'],window_PSTH['timeAfter'])
cutoff_cell=8229

win_begin_PSTH2=-200
win_end_PSTH2=800

behaviour_begin=100
behaviour_end=300

psth_base_array=[]
psth_learned_array=[]
FR_base_array=[]
FR_learned_array=[]

cur_alternative='greater'
half_trials=0
subtract_BL=0
trial_type_test='v20S'

trial_type_mapping='v20p'

overall_cell=0
effect_cell=0

for learned_direction in learned_direction_array:
    for learning_task2_inx,(cong_task2,incong_task2) in enumerate(zip([cong_task+'_'+learned_direction,cong_task+'_'+learned_direction+'_100_25_cue'],[incong_task+'_'+learned_direction,incong_task+'_'+learned_direction+'_100_25_cue'])): 
        cell_cong_list=os.listdir(cell_task_py_folder+cong_task2) #list of strings
        cell_incong_list=os.listdir(cell_task_py_folder+incong_task2) #list of strings
        cur_cell_list=[x for x in cell_cong_list if x in cell_incong_list]
        
        for cell_ID in cur_cell_list: #for cells recorded in both motor and passive
                cell_ID=int(cell_ID)
                cur_cell_cong=load_cell_task(cell_task_py_folder,cong_task2,cell_ID) # load learning cell_task
                trials_df_cong=cur_cell_cong.trials_df   

                #separate into blocks
                # Separate the trials_df to different blocks (each session includes all the dishabitaution blocks recorded during a given day)
                trials_list=trials_df_cong.loc[:]['filename_name'].tolist()
                trial_number_np=np.array([int(x.split('.',1)[1]) for x in trials_list ])
                block_end_indexes=np.where(np.diff(trial_number_np)>=80)[0]#find all trials where next trials is at least with a step of 80 trials (end of blocks)
                block_begin_indexes=np.append(np.array(0),block_end_indexes+1)#(beginning of blocks- just add 1 to end of blocks and also the first trial
                block_end_indexes=np.append(block_end_indexes,len(trials_df_cong)-1)#add the last trial to end of blocks
                            
                #for each learning block:
                for block_inx,(begin_inx,end_inx) in enumerate(zip(block_begin_indexes,block_end_indexes)):
                    trials_df_cong=cur_cell_cong.trials_df   
                    motor_block_df=trials_df_cong.iloc[np.arange(begin_inx,end_inx+1)]
                    #Base and learned directions
                    block_base_dir=motor_block_df.iloc[0]['screen_rotation']#Find base direction of curren block
                    #select the relevant direction as learned direction in the previous washout block
                    if learned_direction=='CW':
                        learned_direction_int=(block_base_dir-90)%360
                    elif learned_direction=='CCW':
                        learned_direction_int=(block_base_dir+90)%360
                    dictFilterTrials_cong = {'dir':'filterOff', 'trial_name':'v20S', 'fail':0, 'block_begin_end':[begin_inx,end_inx]}
                    dictFilterTrials_incong = {'dir':'filterOff', 'trial_name':'v20S', 'fail':0, 'screen_rot':block_base_dir}
                    
                    cur_cell_incong=load_cell_task(cell_task_py_folder,incong_task2,cell_ID) # load learning cell_task

                    trials_df_cong=cur_cell_cong.filtTrials(dictFilterTrials_cong)
                    trials_df_incong=cur_cell_incong.filtTrials(dictFilterTrials_incong)
                    if len(trials_df_cong)<50 or len(trials_df_incong)<50:
                        continue
                                                         
                    #Stability test - check whether the correlation in FR before MO is correlated between washout and subsequent learning blobk
                    try:
                        FR_cong_baseline=cur_cell_cong.get_mean_FR_event(dictFilterTrials_cong,'motion_onset',window_pre=-800,window_post=0)
                        FR_incong_baseline=cur_cell_incong.get_mean_FR_event(dictFilterTrials_incong,'motion_onset',window_pre=-800,window_post=0)
                        stat,p_stability=stats.mannwhitneyu(FR_cong_baseline,FR_incong_baseline)
                        if p_stability<0.05:
                            continue
                    except:
                        continue

                    #Behaviour - Find the learned velocity in the congruent and incongruent blocks
                    if learning_task2_inx==0:
                        cong_task_behaviour='fixation_right_probes_'+learned_direction
                        incong_task_behaviour='fixation_wrong_probes_'+learned_direction
                    elif learning_task2_inx==1:
                        cong_task_behaviour='fixation_right_probes_'+learned_direction+'_100_25_cue'
                        incong_task_behaviour='fixation_wrong_probes_'+learned_direction+'_100_25_cue'
                    
                    try:
                        session=cur_cell_cong.getSession()
                        cong_learned_vel=passive_learning_behaviour_df.loc[(passive_learning_behaviour_df['session']==session) & (passive_learning_behaviour_df['task']==cong_task_behaviour) & (passive_learning_behaviour_df['base_direction']==block_base_dir)& (passive_learning_behaviour_df['learned_direction']==learned_direction)]['learned velocity'].to_list()
                        cong_learned_vel=np.array(cong_learned_vel)  
                        incong_learned_vel=passive_learning_behaviour_df.loc[(passive_learning_behaviour_df['session']==session) & (passive_learning_behaviour_df['task']==incong_task_behaviour) & (passive_learning_behaviour_df['base_direction']==block_base_dir)& (passive_learning_behaviour_df['learned_direction']==learned_direction)]['learned velocity'].to_list()
                        incong_learned_vel=np.array(incong_learned_vel) 
                        # Behavioural test - learned velocity is higher in congruent than incongruent block
                        cong_learned_vel=cong_learned_vel[0]
                        cong_learned_vel_mean=np.nanmean(cong_learned_vel[-win_begin_PSTH2+100:-win_begin_PSTH2+300])
                        incong_learned_vel=incong_learned_vel[0]
                        incong_learned_vel_mean=np.nanmean(incong_learned_vel[-win_begin_PSTH2+100:-win_begin_PSTH2+300])
                        behaviour_effect=cong_learned_vel_mean>incong_learned_vel_mean
                        if not behaviour_effect:
                            continue          
                    except:
                        continue

                    dictFilterTrials_cong_test = {'dir':'filterOff', 'trial_name':trial_type_test, 'fail':0, 'block_begin_end':[begin_inx,end_inx]}
                    dictFilterTrials_incong_test = {'dir':'filterOff', 'trial_name':trial_type_test, 'fail':0, 'screen_rot':block_base_dir}

                    try:                        
                        FR_cong=cur_cell_cong.get_mean_FR_event(dictFilterTrials_cong_test,'motion_onset',window_pre=100,window_post=300)
                        FR_incong=cur_cell_incong.get_mean_FR_event(dictFilterTrials_incong_test,'motion_onset',window_pre=100,window_post=300)
                        stat,p_effect=stats.mannwhitneyu(FR_cong,FR_incong,alternative=cur_alternative)
                        overall_cell=overall_cell+1
                        if p_effect>0.05:
                            continue
                    except:
                        continue
                    

                    if learning_task2_inx==0:
                        mapping_task='8dir_active_passive_interleaved'
                    elif learning_task2_inx==1:
                        mapping_task='8dir_active_passive_interleaved_100_25'  
                    
                    try: 
                        cur_cell_mapping=load_cell_task(cell_task_py_folder,mapping_task,cell_ID)
                        trialname_base='d'+str(block_base_dir)+trial_type_mapping
                        trialname_learned='d'+str(learned_direction_int)+trial_type_mapping

                        dictFilterTrials_mapping_base = {'dir':'filterOff', 'trial_name':trialname_base, 'fail':0}
                        dictFilterTrials_mapping_learned = {'dir':'filterOff', 'trial_name':trialname_learned, 'fail':0}
                        FR_base_array.append(cur_cell_mapping.get_mean_FR_event(dictFilterTrials_mapping_base,'motion_onset',window_pre=100,window_post=300))
                        FR_learned_array.append(cur_cell_mapping.get_mean_FR_event(dictFilterTrials_mapping_learned,'motion_onset',window_pre=100,window_post=300))
                        psth_base_array.append(cur_cell_mapping.PSTH(window_PSTH,dictFilterTrials_mapping_base))
                        psth_learned_array.append(cur_cell_mapping.PSTH(window_PSTH,dictFilterTrials_mapping_learned))

                    except:
                        continue
                    effect_cell=effect_cell+1

                    
#%%

#PSTH individual cells
# for neuron_inx in np.arange(np.size(psth_cong_array,0)):
#     cur_psth_cong=psth_cong_array[neuron_inx]
#     cur_psth_incong=psth_incong_array[neuron_inx]
    
#     plt.plot(timecourse,cur_psth_cong,color='tab:blue') 
#     plt.plot(timecourse,cur_psth_incong,color='tab:orange') 
#     plt.axvline(100,color='black')
#     plt.axvline(300,color='black') 
#     plt.legend(['cong','incong'])
#     plt.show()       

#Average PSTH
psth_base=np.mean(np.array(psth_base_array),0)
psth_learned=np.mean(np.array(psth_learned_array),0)


plt.plot(timecourse,psth_base,color='tab:blue') 
plt.plot(timecourse,psth_learned,color='tab:orange') 

plt.axvline(100,color='black')
plt.axvline(300,color='black') 
plt.legend(['base','learned'])
plt.title('average'+' N='+str(effect_cell)+' cells')
plt.show()             
                 
FR_base_mean=[np.nanmean(x) for x in FR_base_array]           
FR_learned_mean=[np.nanmean(x) for x in FR_learned_array]           

#scatter plot cong vs incong
stat,p_active=stats.wilcoxon(FR_base_mean,FR_learned_mean)
p_active=round(p_active,3)

plt.scatter(FR_base_mean,FR_learned_mean)
plt.axline([0,0],[1,1],color='red')
plt.xlabel('base')
plt.ylabel('learned')
plt.title('p='+str(p_active))
plt.show()

