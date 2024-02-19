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
cell_task_py_folder="units_task_two_monkeys_python_kinematics/"

behaviour_db_excel="C:/Users/Owner/Documents/Matan/Mati's Lab/FEF learning project/Data_Analyses/fiona_behaviour_db.xlsx"
behaviour_db=pd.read_excel(behaviour_db_excel)

#import the behaviour data frame for passive learning
save_path="C:/Users/Owner/Documents/Matan/Mati's Lab/FEF learning project/Data_Analyses/Code analyzes/passive learning/"
filename=save_path+'/'+'behaviour_df_passive_learning'
infile = open(filename,'rb')
passive_learning_behaviour_df= pickle.load(infile)
infile.close()

#%% parameters
mapping_event='cue_onset'
win_begin_PSTH=-500
win_end_PSTH=1000
dir_change=250
timecourse=np.arange(win_begin_PSTH,win_end_PSTH)

learned_direction_array=['CW','CCW']
learning_task='Motor_learning'

#active trials
trial_type_learning='v20NS'
trial_type_mapping='v20a'

cutoff_cell=8229 #cutoff between yasmin and fiona

FR_begin=100
FR_end=300

behaviour_begin=100
behaviour_end=300

p_crit=0.05

correlation_parameter='less'

#%% In this part of the script, we create list of PSTHS fot the different passivge learning blocks and washout

mapping_tasks=['8dir_active_passive_interleaved','8dir_active_passive_interleaved_100_25']

overall_block_inx=0

mapping_base_cue_psths=[]
mapping_learned_cue_psths=[]
mapping_base_motion_psths=[]
mapping_learned_motion_psths=[]

FR_learned_motion=[]
FR_base_motion=[]
FR_learned_cue=[]
FR_base_cue=[]

for learned_direction in learned_direction_array:
    for learning_task2_inx,learning_task2 in enumerate([learning_task+'_'+learned_direction,learning_task+'_'+learned_direction+'_100_25_cue']): 
        cell_learning_list=os.listdir(cell_task_py_folder+learning_task2) #list of strings
        cell_list=[int(item) for item in cell_learning_list ] #list of ints
        mapping_task=mapping_tasks[learning_task2_inx]

        for cell_ID in cell_list:
            
            cur_cell_learning=load_cell_task(cell_task_py_folder,learning_task2,cell_ID) # load learning cell_task
            trials_df_learning_pre=cur_cell_learning.trials_df
           
            #separate into blocks
            # Separate the trials_df to different blocks (each session includes all the dishabitaution blocks recorded during a given day)
            trials_list=trials_df_learning_pre.loc[:]['filename_name'].tolist()
            trial_number_np=np.array([int(x.split('.',1)[1]) for x in trials_list ])
            block_end_indexes=np.where(np.diff(trial_number_np)>=80)[0]#find all trials where next trials is at least with a step of 80 trials (end of blocks)
            block_begin_indexes=np.append(np.array(0),block_end_indexes+1)#(beginning of blocks- just add 1 to end of blocks and also the first trial
            block_end_indexes=np.append(block_end_indexes,len(trials_df_learning_pre)-1)#add the last trial to end of blocks
                   
            #for each learning block:
            for block_inx,(begin_inx,end_inx) in enumerate(zip(block_begin_indexes,block_end_indexes)):

                learning_block_df=trials_df_learning_pre.iloc[np.arange(begin_inx,end_inx+1)]
                
                if len(learning_block_df)<50:
                    continue
                
                #Base and learned directions
                block_base_dir=learning_block_df.iloc[0]['screen_rotation']#Find base direction of curren block
                #select the relevant direction as learned direction in the previous washout block
                if learned_direction=='CW':
                    learned_direction_int=(block_base_dir-90)%360
                elif learned_direction=='CCW':
                    learned_direction_int=(block_base_dir+90)%360
                    
                #Cell with neural behavior correlatrion
                #calculate FR
                dictFilterTrials_learning = {'dir':'filterOff', 'trial_name':trial_type_learning, 'fail':0, 'block_begin_end':[begin_inx,end_inx]}
                trials_df_learning=cur_cell_learning.filtTrials(dictFilterTrials_learning)
                session=cur_cell_learning.getSession()
                block_row=behaviour_db.loc[(behaviour_db['behaviour_session']==session) & (behaviour_db['Task']==learning_task2) & (behaviour_db['screen_rotation']==block_base_dir)].index[0] #first row of the learning block in the behavior db
                file_begin_dishabituation=behaviour_db.iloc[block_row-1]['file_begin'] #find file begin of dishabituation block preceding the learning block
                file_end_dishabituation=behaviour_db.iloc[block_row-1]['file_end']

                #FR for scatter plot
                try:
                    FR_learning=cur_cell_learning.get_mean_FR_event(dictFilterTrials_learning,'motion_onset',window_pre=FR_begin,window_post=FR_end).to_numpy()
                except:
                    continue
                
                #calculate change in position:
                block_df=cur_cell_learning.filtTrials(dictFilterTrials_learning)
                if learned_direction_int in [0,180]:
                    LPosChange= block_df.apply(lambda row: [(row.hPos[behaviour_end+row.motion_onset])-(row.hPos[behaviour_begin+row.motion_onset])],axis=1).to_list()
                    LPosChange = np.array([item for sublist in LPosChange for item in sublist])
                    bPosChange= block_df.apply(lambda row: [(row.vPos[behaviour_end+row.motion_onset])-(row.vPos[behaviour_begin+row.motion_onset])],axis=1).to_list()
                    bPosChange = np.array([item for sublist in bPosChange for item in sublist])

                elif learned_direction_int in [90,270]:
                    LPosChange= block_df.apply(lambda row: [(row.vPos[behaviour_end+row.motion_onset])-(row.vPos[behaviour_begin+row.motion_onset])],axis=1).to_list()
                    LPosChange = np.array([item for sublist in LPosChange for item in sublist])
                    bPosChange= block_df.apply(lambda row: [(row.hPos[behaviour_end+row.motion_onset])-(row.hPos[behaviour_begin+row.motion_onset])],axis=1).to_list()
                    bPosChange = np.array([item for sublist in bPosChange for item in sublist])
                
                if learned_direction_int in [180,270]:
                    LPosChange=-LPosChange
                    
                df = pd.DataFrame({'x': LPosChange, 'y': FR_learning, 'z': bPosChange})
                res=pg.partial_corr(data=df, x='x', y='y', covar=['z'],alternative=correlation_parameter, method='pearson').round(3)
                p=res.iloc[0]['p-val']
                r=res.iloc[0]['r']
                if p>p_crit :
                    continue

                #mapping task: PSTH and FR in base direction
                window_PSTH_cue={"timePoint":'cue_onset','timeBefore':win_begin_PSTH,'timeAfter':win_end_PSTH}  
                window_PSTH_motion={"timePoint":'motion_onset','timeBefore':win_begin_PSTH,'timeAfter':win_end_PSTH}   

                try:

                    cur_cell_mapping=load_cell_task(cell_task_py_folder,mapping_task,cell_ID)
                    trialname_base='d'+str(block_base_dir)+trial_type_mapping
                    dictFilterTrials_mapping_base = {'dir':'filterOff', 'trial_name':trialname_base, 'fail':0}
                    #PSTH and FR in learned direction
                    trialname_learned='d'+str(learned_direction_int)+trial_type_mapping
                    dictFilterTrials_mapping_learned = {'dir':'filterOff', 'trial_name':trialname_learned, 'fail':0}

                    psth_mapping_learned_cue=cur_cell_mapping.PSTH(window_PSTH_cue,dictFilterTrials_mapping_learned)
                    psth_mapping_base_cue=cur_cell_mapping.PSTH(window_PSTH_cue,dictFilterTrials_mapping_base)

                    psth_mapping_learned_motion=cur_cell_mapping.PSTH(window_PSTH_motion,dictFilterTrials_mapping_learned)
                    psth_mapping_base_motion=cur_cell_mapping.PSTH(window_PSTH_motion,dictFilterTrials_mapping_base)

                    FR_mapping_learned_motion=cur_cell_mapping.get_mean_FR_event(dictFilterTrials_mapping_learned,'motion_onset',window_pre=300,window_post=800)
                    FR_mapping_base_motion=cur_cell_mapping.get_mean_FR_event(dictFilterTrials_mapping_base,'motion_onset',window_pre=300,window_post=800)

                    FR_mapping_learned_cue=cur_cell_mapping.get_mean_FR_event(dictFilterTrials_mapping_learned,'cue_onset',window_pre=300,window_post=800)
                    FR_mapping_base_cue=cur_cell_mapping.get_mean_FR_event(dictFilterTrials_mapping_base,'cue_onset',window_pre=300,window_post=800)
                    
                    FR_learned_motion.append(np.nanmean(FR_mapping_learned_motion))
                    FR_learned_cue.append(np.nanmean(FR_mapping_learned_cue))
                    FR_base_motion.append(np.nanmean(FR_mapping_base_motion))
                    FR_base_cue.append(np.nanmean(FR_mapping_base_cue))

                    
                    mapping_learned_cue_psths.append(psth_mapping_learned_cue)
                    mapping_base_cue_psths.append(psth_mapping_base_cue)
                    mapping_learned_motion_psths.append(psth_mapping_learned_motion)
                    mapping_base_motion_psths.append(psth_mapping_base_motion)
                    
                    overall_block_inx=overall_block_inx+1

                except:
                    continue
                                

#%% PSTH cue
psth_mapping_learned_mean=np.nanmean(np.array(mapping_learned_cue_psths),0)
psth_mapping_base_mean=np.nanmean(np.array(mapping_base_cue_psths),0)

plt.plot(timecourse,psth_mapping_learned_mean)
plt.plot(timecourse,psth_mapping_base_mean)
plt.axvline(100,color='black')
plt.axvline(300,color='black')
plt.legend(['learned','base'])
plt.title('Cue- '+'n='+str(overall_block_inx))
plt.show()

#%% scatter cue
stat,p=stats.wilcoxon(np.array(FR_base_cue),np.array(FR_learned_cue))
plt.scatter(FR_base_cue,FR_learned_cue)
plt.axline([0,0],slope=1,color='black')
plt.title('cue '+'p='+str(round(p,4)))
plt.xlabel('base cue')
plt.ylabel('learned cue')
plt.show()
#%% PSTH motion
psth_mapping_learned_mean=np.nanmean(np.array(mapping_learned_motion_psths),0)
psth_mapping_base_mean=np.nanmean(np.array(mapping_base_motion_psths),0)

plt.plot(timecourse,psth_mapping_learned_mean)
plt.plot(timecourse,psth_mapping_base_mean)
plt.axvline(100,color='black')
plt.axvline(300,color='black')
plt.legend(['learned','base'])
plt.title('Motion- '+'n='+str(overall_block_inx))
plt.show()

#%% scatter motion
stat,p=stats.wilcoxon(np.array(FR_base_motion),np.array(FR_learned_motion))
plt.scatter(FR_base_motion,FR_learned_motion)
plt.axline([0,0],slope=1,color='black')
plt.title('motion- p='+str(round(p,4)))
plt.xlabel('base motion')
plt.ylabel('learned motion')
plt.show()

#%% Scatter learned-base distance between cue and motion epoch
cue_distance=np.array(FR_base_cue)-np.array(FR_learned_cue)
motion_distance=np.array(FR_base_motion)-np.array(FR_learned_motion)

stat,p=stats.wilcoxon(cue_distance,motion_distance)
plt.scatter(cue_distance,motion_distance)
plt.axline([0,0],slope=1,color='black')
plt.title('p='+str(round(p,4)))
plt.xlim([-15,40])
plt.ylim([-15,40])
plt.xlabel('cue distance')
plt.ylabel('motion distance')
plt.show()



