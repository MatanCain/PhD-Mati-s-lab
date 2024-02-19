# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 14:48:12 2022

@author: Owner
"""

#In this script we compare the PSTH of cells in the learning and inlearning learning blocks (relative to dishabituation)


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

#%% In this part of the script, we create list of PSTHS fot the different passivge learning blocks and dishabituation
cur_event='motion_onset'
win_begin_PSTH=-200
win_end_PSTH=800
dir_change=250
timecourse=np.arange(win_begin_PSTH,win_end_PSTH)

learned_direction_array=['CW']
learning_task='Motor_learning'
motor_dict_array=[]

#passive trials
trial_type_learning='v20NS'
trial_type_mapping='v20a'

mapping_tasks=['8dir_active_passive_interleaved','8dir_active_passive_interleaved_100_25']
dishabituation_tasks=['Dishabituation','Dishabituation_100_25_cue']
cutoff_cell=8229 #cutoff between yasmin and fiona

window_begin=0
window_end=800

crit_learning_value=0.05

directions=[0,45,90,135,180,225,270,315]

for learned_direction in learned_direction_array:
    for learning_task2_inx,learning_task2 in enumerate([learning_task+'_'+learned_direction,learning_task+'_'+learned_direction+'_100_25_cue']): 
        cell_learning_list=os.listdir(cell_task_py_folder+learning_task2) #list of strings
        mapping_task=mapping_tasks[learning_task2_inx]
        cell_list=[int(item) for item in cell_learning_list if int(item)<cutoff_cell] #list of ints
        for cell_ID in cell_list:

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
                #select the relevant direction as learned direction in the previous dishabituation block
                if learned_direction=='CW':
                    learning_direction_int=(block_base_dir-45)%360
                elif learned_direction=='CCW':
                    learning_direction_int=(block_base_dir+45)%360
              
                session=cur_cell_learning.getSession()
                block_row=behaviour_db.loc[(behaviour_db['behaviour_session']==session) & (behaviour_db['Task']==learning_task2) & (behaviour_db['screen_rotation']==block_base_dir)].index[0] #first row of the learning block in the behavior db

                #dishabituation block before the learning block
                try:
                    cur_cell_dishabituation=load_cell_task(cell_task_py_folder,dishabituation_task,cell_ID)
                except:
                    continue
                session=cur_cell_learning.getSession()
                block_row=behaviour_db.loc[(behaviour_db['behaviour_session']==session) & (behaviour_db['Task']==learning_task2) & (behaviour_db['screen_rotation']==block_base_dir)].index[0] #first row of the learning block in the behavior db
                file_begin_dishabituation=behaviour_db.iloc[block_row-1]['file_begin'] #find file begin of dishabituation block preceding the learning block
                file_end_dishabituation=behaviour_db.iloc[block_row-1]['file_end']
                               
                    
         
                #FR in dishabituation base and learned in odd trials        
                try:
                    dictFilterTrials_learning = {'dir':'filterOff', 'trial_name':trial_type_learning, 'fail':0, 'block_begin_end':[begin_inx,end_inx]}

                    FR_learning_baseline=cur_cell_learning.get_mean_FR_event(dictFilterTrials_learning,'motion_onset',window_pre=-800,window_post=0)
                    FR_learning=cur_cell_learning.get_mean_FR_event(dictFilterTrials_learning,'motion_onset',window_pre=100,window_post=300)
                    stat,p_learning=stats.wilcoxon(FR_learning_baseline, FR_learning,alternative='less')
                    if p_learning>crit_learning_value:
                        continue
                except:
                    continue

                # try:
                #     dictFilterTrials_dishabituation = { 'dir':'filterOff', 'fail':0,'files_begin_end':[file_begin_dishabituation,file_end_dishabituation],'trial_name':'d0'+trial_type_learning}

                #     FR_dishabituation_baseline=cur_cell_dishabituation.get_mean_FR_event(dictFilterTrials_dishabituation,'motion_onset',window_pre=-800,window_post=0)
                #     FR_dishabituation=cur_cell_dishabituation.get_mean_FR_event(dictFilterTrials_dishabituation,'motion_onset',window_pre=300,window_post=800)
                #     stat,p_dis=stats.wilcoxon(FR_dishabituation_baseline, FR_dishabituation,alternative='less')
                #     if p_dis>crit_learning_value:
                #         continue
                # except:
                #     continue

                try:
                    cur_cell_mapping=load_cell_task(cell_task_py_folder,mapping_task,cell_ID)
                    dictFilterTrials_PD = {'dir':'filterOff', 'trial_name':trial_type_mapping, 'fail':0}
                    mapping_tuned=cur_cell_mapping.check_sig_dir(dictFilterTrials_PD,'motion_onset',Window_pre=0,Window_post=800,dir_array=directions)                    
                    if not mapping_tuned:
                        continue
                    FR_mapping_average=cur_cell_mapping.get_mean_FR_event(dictFilterTrials_PD,'motion_onset',window_pre=window_begin,window_post=window_end)
                    tuning_curve=[]
                                   
                    for direction in directions:
                        dictFilterTrials_PD['dir']=direction
                        FR_mapping=cur_cell_mapping.get_mean_FR_event(dictFilterTrials_PD,'motion_onset',window_pre=window_begin,window_post=window_end)-FR_mapping_average
                        tuning_curve.append(np.nanmean(FR_mapping))
                except:
                    continue

                    
                block_dict=dict.fromkeys(['cell_ID', 'base direction', 'learned direction', 'PSTH learning'\
                                          ,'FR learning','FR learning test'\
                                          ,'FR dishabituation','PD','FR_mapping','tuning_curve'])
                block_dict['cell_ID']=cell_ID
                block_dict['base direction']=block_base_dir
                block_dict['learned direction']=learning_direction_int
                block_dict['tuning curve']=tuning_curve
                motor_dict_array.append(block_dict)


#%%

sig_motor_array=np.array([x for x in motor_dict_array ])
tc_array=[]
shift_learned=3
for block_inx,cur_block_dict in enumerate(sig_motor_array):
    shift_inx=int(cur_block_dict['base direction']/45)

    tc_shifted=np.roll(cur_block_dict['tuning curve'],-shift_inx+shift_learned)
    
    # fig, ax = plt.subplots()
    # ax.set_xticks(directions)
    # ax.set_xticklabels(['B-135','B-90','L-CW','BASE','L-CCW','B+90','B+135','B+180'])
    # plt.plot(directions,tc_shifted)
    # plt.axvline(directions[shift_learned])
    # plt.title(str((cur_block_dict['cell_ID'])))
    # plt.show()
    
    tc_array.append(tc_shifted)
tc_average=np.nanmean(np.array(tc_array),axis=0)
fig, ax = plt.subplots()
ax.set_xticks(directions)
ax.set_xticklabels(['B-135','B-90','L-CW','BASE','L-CCW','B+90','B+135','B+180'])
plt.plot(directions,tc_average)
plt.axvline(directions[shift_learned])
plt.title(str(len(sig_motor_array))+'cells')
plt.show()


#%% FR early vs late

# #for dir_inx,base_dir in enumerate([0,90,180,270]):
# PV_num_x=0
# PV_den_x=0

# PV_num_y=0
# PV_den_y=0
# sig_motor_array=np.array([x for x in motor_dict_array ]  )

# for block_inx,cur_block_dict in enumerate(sig_motor_array):
          
#     PD=cur_block_dict['PD']
    
#    # mapping_FR=np.nanmean(cur_block_dict['FR mapping'][dir_inx])
#     dishabituation_FR=np.nanmean(cur_block_dict['FR dishabituation'])
#     learning_FR=np.nanmean(cur_block_dict['FR learning'])

#     PV_num_x=PV_num_x+PD[0]*learning_FR
#     PV_den_x=PV_den_x+learning_FR
    
#     PV_num_y=PV_num_y+PD[1]*learning_FR    
#     PV_den_y=PV_den_y+learning_FR    
# print(len(sig_motor_array))
# plt.scatter(PV_num_x/PV_den_x,PV_num_y/PV_den_y)
# #plt.xlim([-1,1])
# #plt.ylim([-1,1])
# plt.axvline(0,color='black')
# plt.axhline(0,color='black')
# plt.legend()
# plt.show()
