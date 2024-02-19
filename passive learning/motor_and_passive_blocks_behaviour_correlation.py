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
cell_task_py_folder="units_task_two_monkeys_python_kinematics/"

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
motor_task='Motor_learning'
passive_task='fixation_wrong_probes'
learned_direction_array=['CW','CCW']
window_PSTH={"timePoint":'motion_onset','timeBefore':-200,'timeAfter':800}          
timecourse=np.arange(-200,800)
cutoff_cell=8229

behaviour_begin=100
behaviour_end=300

psth_passive_array=[]
psth_motor_array=[]
psth_passive_array2=[]
psth_dis_array=[]
psth_dis_passive_array=[]
cur_alternative='greater'
for learned_direction in learned_direction_array:
    for learning_task2_inx,(motor_task2,passive_task2) in enumerate(zip([motor_task+'_'+learned_direction,motor_task+'_'+learned_direction+'_100_25_cue'],[passive_task+'_'+learned_direction,passive_task+'_'+learned_direction+'_100_25_cue'])): 
        cell_motor_list=os.listdir(cell_task_py_folder+motor_task2) #list of strings
        cell_passive_list=os.listdir(cell_task_py_folder+passive_task2) #list of strings
        cur_cell_list=[x for x in cell_motor_list if x in cell_passive_list]
        
        for cell_ID in cur_cell_list: #for cells recorded in both motor and passive
                cell_ID=int(cell_ID)
                cur_cell_motor=load_cell_task(cell_task_py_folder,motor_task2,cell_ID) # load learning cell_task
                trials_df_motor=cur_cell_motor.trials_df   

                #separate into blocks
                # Separate the trials_df to different blocks (each session includes all the dishabitaution blocks recorded during a given day)
                trials_list=trials_df_motor.loc[:]['filename_name'].tolist()
                trial_number_np=np.array([int(x.split('.',1)[1]) for x in trials_list ])
                block_end_indexes=np.where(np.diff(trial_number_np)>=80)[0]#find all trials where next trials is at least with a step of 80 trials (end of blocks)
                block_begin_indexes=np.append(np.array(0),block_end_indexes+1)#(beginning of blocks- just add 1 to end of blocks and also the first trial
                block_end_indexes=np.append(block_end_indexes,len(trials_df_motor)-1)#add the last trial to end of blocks
                            
                #for each learning block:
                for block_inx,(begin_inx,end_inx) in enumerate(zip(block_begin_indexes,block_end_indexes)):
                    trials_df_motor=cur_cell_motor.trials_df   
                    motor_block_df=trials_df_motor.iloc[np.arange(begin_inx,end_inx+1)]
                    #Base and learned directions
                    block_base_dir=motor_block_df.iloc[0]['screen_rotation']#Find base direction of curren block
                    #select the relevant direction as learned direction in the previous washout block
                    if learned_direction=='CW':
                        learned_direction_int=(block_base_dir-90)%360
                    elif learned_direction=='CCW':
                        learned_direction_int=(block_base_dir+90)%360
                    dictFilterTrials_motor = {'dir':'filterOff', 'trial_name':'v20NS', 'fail':0, 'block_begin_end':[begin_inx,end_inx]}
                    
                    cur_cell_passive=load_cell_task(cell_task_py_folder,passive_task2,cell_ID) # load learning cell_task
                    dictFilterTrials_passive = {'dir':'filterOff', 'trial_name':'v20S', 'fail':0, 'screen_rot':block_base_dir}
                    dictFilterTrials_passive2 = {'dir':'filterOff', 'trial_name':'v20NS', 'fail':0, 'screen_rot':block_base_dir}
                    
                    trials_df_motor=cur_cell_motor.filtTrials(dictFilterTrials_motor)
                    trials_df_passive=cur_cell_passive.filtTrials(dictFilterTrials_passive)
                    if len(trials_df_motor)<50 or len(trials_df_passive)<50:
                        continue
                    
                    #FR in dishabituation base and learned in motor blocks
                    if cell_ID>cutoff_cell:
                        dishabituation_task='washout_100_25_cue'
                    else:
                        if learning_task2_inx==0:
                            dishabituation_task='Dishabituation'
                        else:
                            dishabituation_task='Dishabituation_100_25_cue'
                    try:
                        cur_cell_dishabituation=load_cell_task(cell_task_py_folder,dishabituation_task,cell_ID)
                    except:
                        continue
                    session=cur_cell_motor.getSession()
                    block_row=behaviour_db.loc[(behaviour_db['behaviour_session']==session) & (behaviour_db['Task']==motor_task2) & (behaviour_db['screen_rotation']==block_base_dir)].index[0] #first row of the learning block in the behavior db
                    file_begin_dishabituation=behaviour_db.iloc[block_row-1]['file_begin'] #find file begin of dishabituation block preceding the learning block
                    file_end_dishabituation=behaviour_db.iloc[block_row-1]['file_end']
                    dictFilterTrials_dishabituation= { 'files_begin_end':[file_begin_dishabituation,file_end_dishabituation], 'fail':0,'trial_name':'d0v20NS'}
                    dictFilterTrials_dishabituation_passive= { 'files_begin_end':[file_begin_dishabituation,file_end_dishabituation], 'fail':0,'trial_name':'d0v20S'}
                    try:
                        psth_dis_array.append(cur_cell_dishabituation.PSTH(window_PSTH,dictFilterTrials_dishabituation))
                        psth_dis_passive_array.append(cur_cell_dishabituation.PSTH(window_PSTH,dictFilterTrials_dishabituation_passive))                    
                    except:
                        print('1')
                        continue
                    
                    #calculate change in position:
                    block_df=cur_cell_motor.filtTrials(dictFilterTrials_motor)
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
                    FR_motor=cur_cell_motor.get_mean_FR_event(dictFilterTrials_motor,'motion_onset',window_pre=100,window_post=300).to_numpy()
                    df = pd.DataFrame({'x': LPosChange, 'y': FR_motor, 'z': bPosChange})
                    res=pg.partial_corr(data=df, x='x', y='y', covar=['z'],alternative=cur_alternative, method='pearson').round(3)
                    p=res.iloc[0]['p-val']
                    r=res.iloc[0]['r']
                    if p>0.01:
                        continue
                    
                    #Stability test - check whether the correlation in FR before MO is correlated between washout and subsequent learning blobk
                    try:
                        FR_motor_baseline=cur_cell_motor.get_mean_FR_event(dictFilterTrials_motor,'motion_onset',window_pre=-800,window_post=0)
                        FR_passive_baseline=cur_cell_passive.get_mean_FR_event(dictFilterTrials_passive,'motion_onset',window_pre=-800,window_post=0)
                        stat,p_stability=stats.mannwhitneyu(FR_motor_baseline,FR_passive_baseline)
                        if p_stability<0.01:
                            continue
                    except:
                        print('2')
                        continue

                    psth_passive_array.append(cur_cell_passive.PSTH(window_PSTH,dictFilterTrials_passive))
                    psth_passive_array2.append(cur_cell_passive.PSTH(window_PSTH,dictFilterTrials_passive2))
                    psth_motor_array.append(cur_cell_motor.PSTH(window_PSTH,dictFilterTrials_motor))

 
#%%
psth_passive=np.mean(np.array(psth_passive_array),0)
psth_passive2=np.mean(np.array(psth_passive_array2),0)
psth_motor=np.mean(np.array(psth_motor_array),0)   
psth_dishabituation=np.nanmean(np.array(psth_dis_array),0)   
psth_dishabituation_passive=np.nanmean(np.array(psth_dis_passive_array),0)   


plt.plot(timecourse,psth_passive,color='tab:blue') 
plt.plot(timecourse,psth_passive2,color='tab:blue',linestyle='dashed') 
plt.plot(timecourse,psth_motor,color='tab:orange',linestyle='dashed')
plt.plot(timecourse,psth_dishabituation_passive,color='tab:green')
plt.plot(timecourse,psth_dishabituation,color='tab:green',linestyle='dashed')


plt.axvline(100,color='black')
plt.axvline(300,color='black') 
plt.legend(['passive-P','passive-A','motor','dishabituation-P','dishabituation-A'])
plt.show()             
             
                    
