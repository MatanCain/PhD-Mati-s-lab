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



#%% parameters for both monkeys
cur_event='motion_onset'
win_begin_PSTH=-800
win_end_PSTH=1000
dir_change=250
timecourse=np.arange(win_begin_PSTH,win_end_PSTH)

learned_direction_array=['CW','CCW']
learning_task='Motor_learning'
mapping_tasks=['8dir_active_passive_interleaved','8dir_active_passive_interleaved_100_25']

#active trials
trial_type_learning='v20NS'
trial_type_mapping='v20a'

cutoff_cell=8229 #cutoff between yasmin and fiona

FR_begin=100
FR_end=300

behaviour_begin=100
behaviour_end=300

p_crit=0.05

correlation_parameter='two-sided'

#stability parameter
event_stab='motion_onset'
window_pre_stab=-800
window_end_stab=0
p_stab=0.05

baseline_correction=0

#%% In this part of the script, we create list of PSTHS fot the different passivge learning blocks and washout

dishabituation_tasks=['Dishabituation','Dishabituation_100_25_cue']
overall_block_inx=0
corr_cells=[]
not_corr_cells=[]

dishabituation_psths=[] 
learning_psths=[]
FR_dis_array=[]
FR_learning_array=[]
FR_learning_curve=[]

for learned_direction in learned_direction_array:
    for learning_task2_inx,learning_task2 in enumerate([learning_task+'_'+learned_direction,learning_task+'_'+learned_direction+'_100_25_cue']): 
        cell_learning_list=os.listdir(cell_task_py_folder+learning_task2) #list of strings
        cell_list=[int(item) for item in cell_learning_list if int(item)<cutoff_cell ] #list of ints
        mapping_task=mapping_tasks[learning_task2_inx]

        for cell_ID in cell_list:
            
            cur_cell_learning=load_cell_task(cell_task_py_folder,learning_task2,cell_ID) # load learning cell_task
            trials_df_learning=cur_cell_learning.trials_df
            dishabituation_task=dishabituation_tasks[learning_task2_inx]
           
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
                
                if len(learning_block_df)<50:
                    continue
                
                #Base and learned directions
                block_base_dir=learning_block_df.iloc[0]['screen_rotation']#Find base direction of curren block
                #select the relevant direction as learned direction in the previous washout block
                if learned_direction=='CW':
                    learned_direction_int=(block_base_dir-90)%360
                elif learned_direction=='CCW':
                    learned_direction_int=(block_base_dir+90)%360
                    
                #PSTH learning
                dictFilterTrials_learning = {'dir':'filterOff', 'trial_name':trial_type_learning, 'fail':0, 'block_begin_end':[begin_inx,end_inx]}
                window_PSTH={"timePoint":cur_event,'timeBefore':win_begin_PSTH,'timeAfter':win_end_PSTH}          
                trials_df_learning=cur_cell_learning.filtTrials(dictFilterTrials_learning)
                psth_learning=cur_cell_learning.PSTH(window_PSTH,dictFilterTrials_learning)


                #dishabituation block before the learning block
                try:
                    cur_cell_dishabituation=load_cell_task(cell_task_py_folder,dishabituation_task,cell_ID)
                except:
                    continue
                session=cur_cell_learning.getSession()
                block_row=behaviour_db.loc[(behaviour_db['behaviour_session']==session) & (behaviour_db['Task']==learning_task2) & (behaviour_db['screen_rotation']==block_base_dir)].index[0] #first row of the learning block in the behavior db
                file_begin_dishabituation=behaviour_db.iloc[block_row-1]['file_begin'] #find file begin of dishabituation block preceding the learning block
                file_end_dishabituation=behaviour_db.iloc[block_row-1]['file_end']
                dictFilterTrials_dishabituation = { 'dir':'filterOff', 'fail':0,'files_begin_end':[file_begin_dishabituation,file_end_dishabituation],'trial_name':'d0'+trial_type_learning}
                psth_dishabituation=cur_cell_dishabituation.PSTH(window_PSTH,dictFilterTrials_dishabituation)


                #stability dishabituation - check whether the correlation in FR before MO is correlated between dishabituation and subsequent learning block                        
                try:
                    dictFilterTrials_dishabituation_stability = { 'dir':'filterOff', 'fail':0,'files_begin_end':[file_begin_dishabituation,file_end_dishabituation],'trial_name':'d0'+trial_type_learning}
                    FR_dishabituation_baseline=cur_cell_dishabituation.get_mean_FR_event(dictFilterTrials_dishabituation_stability,event_stab,window_pre=window_pre_stab,window_post=window_end_stab)
                    dictFilterTrials_learning_stability = {'dir':'filterOff', 'fail':0, 'block_begin_end':[begin_inx,end_inx], 'trial_name':trial_type_learning}
                    FR_learning_baseline=cur_cell_learning.get_mean_FR_event(dictFilterTrials_learning_stability,event_stab,window_pre=window_pre_stab,window_post=window_end_stab)
                    #stability based on ttest
                    r,p_stability_learning=stats.mannwhitneyu(FR_dishabituation_baseline,FR_learning_baseline)
                    if p_stability_learning<p_stab:
                        continue
                except:
                    continue     
                
                dictFilterTrials_dishabituation_stability = { 'dir':'filterOff', 'fail':0,'files_begin_end':[file_begin_dishabituation,file_end_dishabituation],'trial_name':'d0'+trial_type_learning}
                FR_dishabituation_baseline=cur_cell_dishabituation.get_mean_FR_event(dictFilterTrials_dishabituation_stability,'motion_onset',window_pre=-800,window_post=0)
                dictFilterTrials_learning_stability = {'dir':'filterOff', 'fail':0, 'block_begin_end':[begin_inx,end_inx], 'trial_name':trial_type_learning}
                FR_learning_baseline=cur_cell_learning.get_mean_FR_event(dictFilterTrials_learning_stability,'motion_onset',window_pre=-800,window_post=0)

                #FR for scatter plot
                try:
                    FR_dis=cur_cell_dishabituation.get_mean_FR_event(dictFilterTrials_dishabituation,'motion_onset',window_pre=100,window_post=300).to_numpy()
                    FR_learning=cur_cell_learning.get_mean_FR_event(dictFilterTrials_learning,'motion_onset',window_pre=100,window_post=300).to_numpy()
                except:
                    continue
                
                if baseline_correction:
                    dictFilterTrials_dishabituation_stability = { 'dir':'filterOff', 'fail':0,'files_begin_end':[file_begin_dishabituation,file_end_dishabituation],'trial_name':'d0'+trial_type_learning}
                    FR_dishabituation_baseline=cur_cell_dishabituation.get_mean_FR_event(dictFilterTrials_dishabituation_stability,'motion_onset',window_pre=-800,window_post=0)
                    dictFilterTrials_learning_stability = {'dir':'filterOff', 'fail':0, 'block_begin_end':[begin_inx,end_inx], 'trial_name':trial_type_learning}
                    FR_learning_baseline=cur_cell_learning.get_mean_FR_event(dictFilterTrials_learning_stability,'motion_onset',window_pre=-800,window_post=0)
                else:
                    FR_learning_baseline=0
                    FR_dishabituation_baseline=0
                
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
                overall_block_inx=overall_block_inx+1


                # if r<0:
                #     FR_learning=-FR_learning
                #     FR_dis=-FR_dis
                #     psth_dishabituation=-psth_dishabituation
                #     psth_learning=-psth_learning
                #     FR_learning_baseline=-FR_learning_baseline
                #     FR_dishabituation_baseline=-FR_dishabituation_baseline
                    
                if p<p_crit :
                    if r>0:
                        learning_psths.append(psth_learning-np.nanmean(FR_learning_baseline))
                        dishabituation_psths.append(psth_dishabituation-np.nanmean(FR_dishabituation_baseline))
                        FR_dis=FR_dis-np.nanmean(FR_dishabituation_baseline)
                        FR_learning=FR_learning-np.nanmean(FR_learning_baseline)

                        FR_dis_array.append(np.nanmean(FR_dis))
                        FR_learning_array.append(np.nanmean(FR_learning))
                    if r<0:
                        learning_psths.append(-psth_learning+np.nanmean(FR_learning_baseline))
                        dishabituation_psths.append(-psth_dishabituation+np.nanmean(FR_dishabituation_baseline))
                        FR_dis=-FR_dis+np.nanmean(FR_dishabituation_baseline)
                        FR_learning=-FR_learning+np.nanmean(FR_learning_baseline)
                        
                        FR_dis_array.append(np.nanmean(FR_dis))
                        FR_learning_array.append(np.nanmean(FR_learning))
                    FR_learning_curve.append(FR_learning)
                else: 
                    not_corr_cells.append(overall_block_inx)
                    continue
                                
 

                
# n_sig_cells=str(len(corr_cells))
# n_tot_cells=str(len(not_corr_cells)+len(corr_cells))
# prop_sig_cells=str(round(int(n_sig_cells)/int(n_tot_cells),3))
# learning_psths_mean=np.array(np.nanmean(learning_psths,axis=0))            
# dishabituation_psths_mean=np.array(np.nanmean(dishabituation_psths,axis=0)) 
# #learning_nan_saccades_psths_mean=np.array(np.nanmean(learning_nan_saccades_psths,axis=0)) 

# plt.title(n_sig_cells+'/'+n_tot_cells+'-'+prop_sig_cells+' cells'+'-'+correlation_parameter) 
# plt.plot(timecourse,learning_psths_mean)
# plt.plot(timecourse,dishabituation_psths_mean)
# #plt.plot(timecourse,learning_nan_saccades_psths_mean)

# plt.axvline(100,color='black')
# plt.axvline(300,color='black')
# plt.legend(['learning','dis'])
# plt.show()          

# res=stats.wilcoxon(FR_dis_array,FR_learning_array)
# p=str(round(res.pvalue,3))
# plt.scatter(FR_dis_array,FR_learning_array)
# plt.title('p='+p)
# plt.axline([0,0],slope=1,color='black')
# plt.xlabel('FR dis')
# plt.ylabel('FR learning')
# plt.show()

#%% In this part of the script, we create list of PSTHS fot the different passivge learning blocks and washout

dishabituation_task='washout_100_25_cue'

#overall_block_inx=0
# corr_cells=[]
# not_corr_cells=[]

# dishabituation_psths=[]
# learning_psths=[]
# FR_dis_array=[]
# FR_learning_array=[]

for learned_direction in learned_direction_array:
    for learning_task2_inx,learning_task2 in enumerate([learning_task+'_'+learned_direction,learning_task+'_'+learned_direction+'_100_25_cue']): 
        cell_learning_list=os.listdir(cell_task_py_folder+learning_task2) #list of strings
        cell_list=[int(item) for item in cell_learning_list if int(item)>cutoff_cell ] #list of ints
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
                window_PSTH={"timePoint":cur_event,'timeBefore':win_begin_PSTH,'timeAfter':win_end_PSTH}          

                trials_df_learning=cur_cell_learning.filtTrials(dictFilterTrials_learning)
                #psth_learning_nan_saccades=PSTH_nan_saccades(window_PSTH,dictFilterTrials_learning,trials_df_learning)
                psth_learning=cur_cell_learning.PSTH(window_PSTH,dictFilterTrials_learning)


                #dishabituation block before the learning block
                try:
                    cur_cell_dishabituation=load_cell_task(cell_task_py_folder,dishabituation_task,cell_ID)
                except:
                    continue
                session=cur_cell_learning.getSession()
                block_row=behaviour_db.loc[(behaviour_db['behaviour_session']==session) & (behaviour_db['Task']==learning_task2) & (behaviour_db['screen_rotation']==block_base_dir)].index[0] #first row of the learning block in the behavior db
                file_begin_dishabituation=behaviour_db.iloc[block_row-1]['file_begin'] #find file begin of dishabituation block preceding the learning block
                file_end_dishabituation=behaviour_db.iloc[block_row-1]['file_end']
                dictFilterTrials_dishabituation = { 'dir':'filterOff', 'fail':0,'files_begin_end':[file_begin_dishabituation,file_end_dishabituation],'trial_name':'d0'+trial_type_learning}
                psth_dishabituation=cur_cell_dishabituation.PSTH(window_PSTH,dictFilterTrials_dishabituation)
                
                #stability dishabituation - check whether the correlation in FR before MO is correlated between dishabituation and subsequent learning block                        
                try:
                    dictFilterTrials_dishabituation_stability = { 'dir':'filterOff', 'fail':0,'files_begin_end':[file_begin_dishabituation,file_end_dishabituation],'trial_name':'d0'+trial_type_learning}
                    FR_dishabituation_baseline=cur_cell_dishabituation.get_mean_FR_event(dictFilterTrials_dishabituation_stability,event_stab,window_pre=window_pre_stab,window_post=window_end_stab)
                    dictFilterTrials_learning_stability = {'dir':'filterOff', 'fail':0, 'block_begin_end':[begin_inx,end_inx], 'trial_name':trial_type_learning}
                    FR_learning_baseline=cur_cell_learning.get_mean_FR_event(dictFilterTrials_learning_stability,event_stab,window_pre=window_pre_stab,window_post=window_end_stab)
                    #stability based on ttest
                    r,p_stability_learning=stats.mannwhitneyu(FR_dishabituation_baseline,FR_learning_baseline)
                    if p_stability_learning<p_stab:
                        continue
                except:
                    continue
                
                if baseline_correction:
                    dictFilterTrials_dishabituation_stability = { 'dir':'filterOff', 'fail':0,'files_begin_end':[file_begin_dishabituation,file_end_dishabituation],'trial_name':'d0'+trial_type_learning}
                    FR_dishabituation_baseline=cur_cell_dishabituation.get_mean_FR_event(dictFilterTrials_dishabituation_stability,'motion_onset',window_pre=-800,window_post=0)
                    dictFilterTrials_learning_stability = {'dir':'filterOff', 'fail':0, 'block_begin_end':[begin_inx,end_inx], 'trial_name':trial_type_learning}
                    FR_learning_baseline=cur_cell_learning.get_mean_FR_event(dictFilterTrials_learning_stability,'motion_onset',window_pre=-800,window_post=0)
                else:
                    FR_learning_baseline=0
                    FR_dishabituation_baseline=0
                    

                #FR for scatter plot
                try:
                    FR_dis=cur_cell_dishabituation.get_mean_FR_event(dictFilterTrials_dishabituation,'motion_onset',window_pre=100,window_post=300).to_numpy()
                    FR_learning=cur_cell_learning.get_mean_FR_event(dictFilterTrials_learning,'motion_onset',window_pre=100,window_post=300).to_numpy()
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
                overall_block_inx=overall_block_inx+1


                # if r<0:
                #     FR_learning=-FR_learning
                #     FR_dis=-FR_dis
                #     psth_dishabituation=-psth_dishabituation
                #     psth_learning=-psth_learning
                #     FR_learning_baseline=-FR_learning_baseline
                #     FR_dishabituation_baseline=-FR_dishabituation_baseline
                    
                if p<p_crit :
                    
                    if r>0:
                        learning_psths.append(psth_learning-np.nanmean(FR_learning_baseline))
                        dishabituation_psths.append(psth_dishabituation-np.nanmean(FR_dishabituation_baseline))
                        FR_dis=FR_dis-np.nanmean(FR_dishabituation_baseline)
                        FR_learning=FR_learning-np.nanmean(FR_learning_baseline)
                        FR_dis_array.append(np.nanmean(FR_dis))
                        FR_learning_array.append(np.nanmean(FR_learning))
                    if r<0:
                        learning_psths.append(-psth_learning+np.nanmean(FR_learning_baseline))
                        dishabituation_psths.append(-psth_dishabituation+np.nanmean(FR_dishabituation_baseline))
                        FR_dis=-FR_dis+np.nanmean(FR_dishabituation_baseline)
                        FR_learning=-FR_learning+np.nanmean(FR_learning_baseline)
                        
                        FR_dis_array.append(np.nanmean(FR_dis))
                        FR_learning_array.append(np.nanmean(FR_learning))
                        
                    FR_learning_curve.append(FR_learning)

                else: 
                    not_corr_cells.append(overall_block_inx)
                    continue
                                
 
        
n_sig_cells=str(len(corr_cells))
n_tot_cells=str(len(not_corr_cells)+len(corr_cells))
prop_sig_cells=str(round(int(n_sig_cells)/int(n_tot_cells),3))
learning_psths_mean=np.array(np.nanmean(learning_psths,axis=0))            
dishabituation_psths_mean=np.array(np.nanmean(dishabituation_psths,axis=0)) 
#learning_nan_saccades_psths_mean=np.array(np.nanmean(learning_nan_saccades_psths,axis=0)) 

plt.title(n_sig_cells+'/'+n_tot_cells+'-'+prop_sig_cells+' cells'+'-'+correlation_parameter) 
plt.plot(timecourse,learning_psths_mean)
plt.plot(timecourse,dishabituation_psths_mean)
#plt.plot(timecourse,learning_nan_saccades_psths_mean)

plt.axvline(100,color='black')
plt.axvline(300,color='black')
plt.legend(['learning','dis'])
plt.show()          

res=stats.wilcoxon(FR_dis_array,FR_learning_array)
p=str(round(res.pvalue,3))
plt.title('p='+p)
plt.scatter(FR_dis_array,FR_learning_array)
plt.axline([0,0],slope=1,color='black')
plt.xlabel('FR dis')
plt.ylabel('FR learning')
plt.show()

#%% learning curve
FR_learning_curve=[x for x in FR_learning_curve if np.size(x)==80]
learning_curve=np.nanmean(np.array(FR_learning_curve),axis=0)
plt.plot(learning_curve)
plt.xlabel('trial number')
plt.ylabel('FR')
plt.title('tuning curve')
plt.show()

