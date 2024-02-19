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
congruent_tasks=[]
learning_tasks=['fixation_right_probes','fixation_wrong_probes']
cong_dict_array=[]
incong_dict_array=[]
learning_dict=[cong_dict_array,incong_dict_array]

#passive trials
trial_type_learning='v20S'

mapping_tasks=['8dir_active_passive_interleaved','8dir_active_passive_interleaved_100_25']
washout_task='washout_100_25_cue'
cutoff_cell=8229 #cutoff between yasmin and fiona

window_begin=100
window_end=300

crit_learning_value=0.05

for learning_task_inx,learning_task in enumerate(learning_tasks):
    cur_dict_array=learning_dict[learning_task_inx]
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
                                   
                    #FR in washout base and learned in odd trials        
                    try:
                        dictFilterTrials_washout_base_test = { 'files_begin_end':[file_begin_washout,file_end_washout], 'fail':0,'trial_name':'d0'+trial_type_learning,'even_odd_rows':'odd'}
                        dictFilterTrials_washout_learned_test = { 'files_begin_end':[file_begin_washout,file_end_washout], 'fail':0,'trial_name':'d'+washout_learned_dir+trial_type_learning,'even_odd_rows':'odd'}
                        dictFilterTrials_learning_test = {'dir':'filterOff', 'trial_name':trial_type_learning, 'fail':0, 'block_begin_end':[begin_inx,end_inx],'even_odd_rows':'odd'}

                        FR_washout_base_test=cur_cell_washout.get_mean_FR_event(dictFilterTrials_washout_base_test,'motion_onset',window_pre=100,window_post=300)
                        FR_washout_learned_test=cur_cell_washout.get_mean_FR_event(dictFilterTrials_washout_learned_test,'motion_onset',window_pre=100,window_post=300)
                        FR_learning_test=cur_cell_learning.get_mean_FR_event(dictFilterTrials_learning_test,'motion_onset',window_pre=100,window_post=300)
                        
                        stat,p_learning=stats.mannwhitneyu(FR_learning_test, FR_washout_base_test)
                        if p_learning>crit_learning_value:
                            continue
                    except:
                        continue
                   
                    #PSTHs in learning and washout blocks
                    window_PSTH={"timePoint":cur_event,'timeBefore':win_begin_PSTH,'timeAfter':win_end_PSTH}          
                    dictFilterTrials_learning_PSTH = {'dir':'filterOff', 'trial_name':trial_type_learning, 'fail':0, 'block_begin_end':[begin_inx,end_inx],'even_odd_rows':'even'}
                    dictFilterTrials_washout_base = { 'dir':'filterOff', 'fail':0,'files_begin_end':[file_begin_washout,file_end_washout],'trial_name':'d0'+trial_type_learning,'even_odd_rows':'even'}
                    dictFilterTrials_washout_learned = { 'dir':'filterOff', 'fail':0,'files_begin_end':[file_begin_washout,file_end_washout],'trial_name':'d'+washout_learned_dir+trial_type_learning,'even_odd_rows':'even'}

                    try: #only for yasmine cells
                        FR_learning=cur_cell_learning.get_mean_FR_event(dictFilterTrials_learning_PSTH,'motion_onset',window_pre=window_begin,window_post=window_end)
                        FR_washout_learned=cur_cell_washout.get_mean_FR_event(dictFilterTrials_washout_learned,'motion_onset',window_pre=window_begin,window_post=window_end)
                        FR_washout_base=cur_cell_washout.get_mean_FR_event(dictFilterTrials_washout_base,'motion_onset',window_pre=window_begin,window_post=window_end)
                        psth_learning=cur_cell_learning.PSTH(window_PSTH,dictFilterTrials_learning_PSTH)
                        psth_washout_learned=cur_cell_washout.PSTH(window_PSTH,dictFilterTrials_washout_learned)
                        psth_washout_base=cur_cell_washout.PSTH(window_PSTH,dictFilterTrials_washout_base)
                    except:
                        continue
                    
                    #Stability washout - check whether the correlation in FR before MO is correlated between washout and subsequent learning block
                    try:
                        dictFilterTrials_washout_base_stability = { 'dir':'filterOff', 'fail':0,'files_begin_end':[file_begin_washout,file_end_washout],'trial_name':'d0'+trial_type_learning,'even_odd_rows':'even'}
                        FR_learning_washout_baseline=cur_cell_washout.get_mean_FR_event(dictFilterTrials_washout_base_stability,'motion_onset',window_pre=-800,window_post=0)
                        dictFilterTrials_learning_stability = {'dir':'filterOff', 'fail':0, 'block_begin_end':[begin_inx,end_inx], 'trial_name':trial_type_learning,'even_odd_rows':'even'}
                        FR_learning_baseline2=cur_cell_learning.get_mean_FR_event(dictFilterTrials_learning_stability,'motion_onset',window_pre=-800,window_post=0)
                        r,p_stability_learning=stats.mannwhitneyu(FR_learning_baseline2, FR_learning_washout_baseline)
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
                        cong_learned_vel_mean=np.nanmean(cong_learned_vel[-win_begin_PSTH+100:-win_begin_PSTH+300])
                        incong_learned_vel=incong_learned_vel[0]
                        incong_learned_vel_mean=np.nanmean(incong_learned_vel[-win_begin_PSTH+100:-win_begin_PSTH+300])
                        behaviour_effect=cong_learned_vel_mean>incong_learned_vel_mean
                    except:
                        behaviour_effect=0
                            
                    block_dict=dict.fromkeys(['cell_ID', 'base direction', 'learned direction', 'PSTH learning','PSTH washout base'\
                                              ,'PSTH washout learned','FR learning','FR learning test','FR dishabituation','FR washout base test'\
                                              'FR washout learned test','FR washout base','FR washout learned','behaviour effect'])
                    block_dict['cell_ID']=cell_ID
                    block_dict['base direction']=block_base_dir
                    block_dict['learned direction']=learned_direction
                    block_dict['PSTH learning']=psth_learning
                    block_dict['PSTH washout base']=psth_washout_base
                    block_dict['PSTH washout learned']=psth_washout_learned
                    block_dict['FR learning']=FR_learning           
                    block_dict['FR learning test']=FR_learning_test           
                    block_dict['FR washout base']=FR_washout_base
                    block_dict['FR washout learned test']=FR_washout_learned_test
                    block_dict['FR washout base test']=FR_washout_base_test
                    block_dict['FR washout learned']=FR_washout_learned           
                    block_dict['behaviour effect']=behaviour_effect #1 if cell is significant for learned velocity betwenn washout base and learning block (100-300)
                    cur_dict_array.append(block_dict)


#%%Number of significant cells 

sig_cong_array=np.array([x for x in cong_dict_array if   x['behaviour effect']]  )
sig_incong_array=np.array([x for x in incong_dict_array if   x['behaviour effect']]  )

n_cells_cong=len(sig_cong_array)
n_cells_incong=len(sig_incong_array)
plt.bar(['congruent blocks','incongruent blocks'],[n_cells_cong,n_cells_incong])    
plt.show()


#%% leanring vs washout - inversion based on sign in Wb vs learning (yasmin only)                 
                                                                                                                                                                                                                                                                                                       
sig_cong_array=np.array([x for x in cong_dict_array if x['behaviour effect']]  )
sig_incong_array=np.array([x for x in incong_dict_array if   x['behaviour effect'] ])   

n_cells_cong=len(sig_cong_array)
n_cells_incong=len(sig_incong_array)

legend_size=8
title_array=['cong blocks','incongruent blocks']
n_cells_array=[n_cells_cong,n_cells_incong]
color_array=['tab:blue','tab:orange']

washout_base_congruent=[]
learning_congruent=[]
washout_base_incongruent=[]
learning_incongruent=[]

for array_inx,sig_array in enumerate([sig_cong_array,sig_incong_array]):
    washout_base_psths=np.empty((len(sig_array),len(timecourse)))
    washout_base_psths[:]=np.nan
    washout_learned_psths=np.empty((len(sig_array),len(timecourse)))
    washout_learned_psths[:]=np.nan
    learning_psths=np.empty((len(sig_array),len(timecourse)))
    learning_psths[:]=np.nan
    dishabituation_psths=np.empty((len(sig_array),len(timecourse)))
    dishabituation_psths[:]=np.nan

    washout_base_FR=[]
    washout_learned_FR=[]
    learning_FR=[]
    n_cells=0
    for block_inx,cur_block_dict in enumerate(sig_array):
                    
            mean_washout=np.nanmean(cur_block_dict['FR washout base test'])
            mean_learning=np.nanmean(cur_block_dict['FR learning test'])
            
            if cur_block_dict['cell_ID']<cutoff_cell:
                continue
            if mean_washout> mean_learning:
                
                learning_array=cur_block_dict['PSTH learning']
                washout_array=cur_block_dict['PSTH washout base']
                washout_learned_array=cur_block_dict['PSTH washout learned']
                washout_base_FR.append(np.nanmean(cur_block_dict['FR washout base']))
                washout_learned_FR.append(np.nanmean(cur_block_dict['FR washout learned']))
                learning_FR.append(np.nanmean(cur_block_dict['FR learning']))
            elif mean_washout< mean_learning:
                learning_array=-cur_block_dict['PSTH learning']
                washout_array=-cur_block_dict['PSTH washout base']
                washout_learned_array=-cur_block_dict['PSTH washout learned']
                washout_base_FR.append(-np.nanmean(cur_block_dict['FR washout base']))
                washout_learned_FR.append(-np.nanmean(cur_block_dict['FR washout learned']))
                learning_FR.append(-np.nanmean(cur_block_dict['FR learning']))

            washout_base_psths[block_inx,:]=washout_array
            washout_learned_psths[block_inx,:]=washout_learned_array
            learning_psths[block_inx,:]=learning_array
            n_cells=n_cells+1

    

    #check wether FR in 100-300 between mapping base and learning is significantly different         
    stat,p_learning_base=stats.wilcoxon(washout_base_FR,learning_FR)
    p_learning_base=round(p_learning_base,3)
    
    plt.plot(timecourse,np.nanmean(learning_psths,axis=0),color=color_array[array_inx])
    plt.plot(timecourse,np.nanmean(washout_learned_psths,axis=0),color='tab:green',linestyle='dashed')
    plt.plot(timecourse,np.nanmean(washout_base_psths,axis=0),color='tab:green',linestyle='dotted')
    plt.axvline(x=100,color='black')
    plt.axvline(x=300,color='black')
    plt.legend(['learning','wL','wB'], loc ="lower right",fontsize=legend_size)
    plt.title(title_array[array_inx]+' '+str(n_cells)+' cells ,p='+str(p_learning_base))
    plt.xlabel('time from MO')
    plt.ylabel('FR')
    plt.show()

    plt.scatter(learning_FR,washout_base_FR,color=color_array[array_inx])
    plt.axline((0,0),(1,1),color='black')
    plt.title(title_array[array_inx]+' '+str(n_cells)+' cells ,p='+str(p_learning_base))
    plt.xlabel('FR learning')
    plt.ylabel('FR washout base')
    plt.show()
 
#%% leanring vs washout - inversion based on sign in Wb vs wL (yasmin only)                 

sig_cong_array=np.array([x for x in cong_dict_array if x['behaviour effect']]  )
sig_incong_array=np.array([x for x in incong_dict_array if   x['behaviour effect'] ])   

n_cells_cong=len(sig_cong_array)
n_cells_incong=len(sig_incong_array)

legend_size=8
title_array=['cong blocks','incongruent blocks']
n_cells_array=[n_cells_cong,n_cells_incong]
color_array=['tab:blue','tab:orange']

washout_base_congruent=[]
learning_congruent=[]
washout_base_incongruent=[]
learning_incongruent=[]

for array_inx,sig_array in enumerate([sig_cong_array,sig_incong_array]):
    washout_base_psths=np.empty((len(sig_array),len(timecourse)))
    washout_base_psths[:]=np.nan
    washout_learned_psths=np.empty((len(sig_array),len(timecourse)))
    washout_learned_psths[:]=np.nan
    learning_psths=np.empty((len(sig_array),len(timecourse)))
    learning_psths[:]=np.nan
    dishabituation_psths=np.empty((len(sig_array),len(timecourse)))
    dishabituation_psths[:]=np.nan

    washout_base_FR=[]
    washout_learned_FR=[]
    learning_FR=[]
    n_cells=0
    for block_inx,cur_block_dict in enumerate(sig_array):
                    
            mean_washout_base=np.nanmean(cur_block_dict['FR washout base test'])
            mean_washout_learned=np.nanmean(cur_block_dict['FR washout learned test'])

            if mean_washout_base> mean_washout_learned:
                learning_array=cur_block_dict['PSTH learning']
                washout_array=cur_block_dict['PSTH washout base']
                washout_learned_array=cur_block_dict['PSTH washout learned']
                washout_base_FR.append(np.nanmean(cur_block_dict['FR washout base']))
                washout_learned_FR.append(np.nanmean(cur_block_dict['FR washout learned']))
                learning_FR.append(np.nanmean(cur_block_dict['FR learning']))
            elif mean_washout_base< mean_washout_learned:
                learning_array=-cur_block_dict['PSTH learning']
                washout_array=-cur_block_dict['PSTH washout base']
                washout_learned_array=-cur_block_dict['PSTH washout learned']
                washout_base_FR.append(-np.nanmean(cur_block_dict['FR washout base']))
                washout_learned_FR.append(-np.nanmean(cur_block_dict['FR washout learned']))
                learning_FR.append(-np.nanmean(cur_block_dict['FR learning']))

            washout_base_psths[block_inx,:]=washout_array
            washout_learned_psths[block_inx,:]=washout_learned_array
            learning_psths[block_inx,:]=learning_array
            n_cells=n_cells+1

    #check wether FR in 100-300 between mapping base and learning is significantly different         
    stat,p_learning_base=stats.wilcoxon(washout_base_FR,learning_FR)
    p_learning_base=round(p_learning_base,3)
    
    plt.plot(timecourse,np.nanmean(learning_psths,axis=0),color=color_array[array_inx])
    plt.plot(timecourse,np.nanmean(washout_learned_psths,axis=0),color='tab:green',linestyle='dashed')
    plt.plot(timecourse,np.nanmean(washout_base_psths,axis=0),color='tab:green',linestyle='dotted')
    plt.axvline(x=100,color='black')
    plt.axvline(x=300,color='black')
    plt.legend(['learning','wL','wB'], loc ="lower right",fontsize=legend_size)
    plt.title(title_array[array_inx]+' '+str(n_cells)+' cells ,p='+str(p_learning_base))
    plt.xlabel('time from MO')
    plt.ylabel('FR')
    plt.show()

    plt.scatter(learning_FR,washout_base_FR,color=color_array[array_inx])
    plt.axline((0,0),(1,1),color='black')
    plt.title(title_array[array_inx]+' '+str(n_cells)+' cells ,p='+str(p_learning_base))
    plt.xlabel('FR learning')
    plt.ylabel('FR washout base')
    #plt.xlim([-15,100])
    #plt.ylim([-15,100])
    plt.show()

    if array_inx==0:
        washout_base_congruent=np.nanmean(washout_base_psths[:,-win_begin_PSTH+100:-win_begin_PSTH+300],axis=1)
        learning_congruent=np.nanmean(learning_psths[:,-win_begin_PSTH+100:-win_begin_PSTH+300],axis=1)
    if array_inx==1:
        washout_base_incongruent=np.nanmean(washout_base_psths[:,-win_begin_PSTH+100:-win_begin_PSTH+300],axis=1)
        learning_incongruent=np.nanmean(learning_psths[:,-win_begin_PSTH+100:-win_begin_PSTH+300],axis=1)
        
x=washout_base_congruent-learning_congruent
y=washout_base_incongruent-learning_incongruent
x=x[~np.isnan(x)]
y=y[~np.isnan(y)]

stat,p_washout_learning_distance=stats.mannwhitneyu(x,y)
print(p_washout_learning_distance)

#%% Learning curve for significant cells in washout vs congruent:

# sig_cong_array=np.array([x for x in cong_dict_array if x['behaviour effect']]  )
# sig_incong_array=np.array([x for x in incong_dict_array if   x['behaviour effect'] ])   

# n_cells_cong=len(sig_cong_array)
# n_cells_incong=len(sig_incong_array)

# legend_size=8
# title_array=['cong blocks','incongruent blocks']
# n_cells_array=[n_cells_cong,n_cells_incong]
# color_array=['tab:blue','tab:orange']

# n_learning_trials=36
    
# for array_inx,sig_array in enumerate([sig_cong_array,sig_incong_array]):

#     washout_base_FR_array=[]
#     washout_learned_FR_array=[]
#     learning_FR_array=[]
#     n_cells=0
#     for block_inx,cur_block_dict in enumerate(sig_array):

#             mean_washout_base=np.nanmean(cur_block_dict['FR washout base'])
#             mean_washout_learned=np.nanmean(cur_block_dict['FR washout learned'])
#             mean_learning=np.nanmean(cur_block_dict['FR learning'])

#             mean_washout_base_test=np.nanmean(cur_block_dict['FR washout base test'])
#             mean_washout_learned_test=np.nanmean(cur_block_dict['FR washout learned test'])
#             mean_learning_test=np.nanmean(cur_block_dict['FR learning test'])
        
            
#             learning_FR=np.empty([n_learning_trials])
#             learning_FR[:]=np.nan
#             learning_FR[0:len(cur_block_dict['FR learning'])]=np.array(cur_block_dict['FR learning'])[0:n_learning_trials]
#             try:
#                 coef_cur_block = np.polyfit(np.arange(n_learning_trials),learning_FR,1)
#                 slope_cur_block=round(coef_cur_block[0],2)
#             except: 
#                 continue
#             # if slope_cur_block<0:
#             #     learning_FR=-learning_FR
#             # learning_FR_array.append(learning_FR)   
             
#             if mean_washout_base_test< mean_learning_test:
#                 learning_FR_array.append(learning_FR) 

#             elif mean_washout_base_test> mean_learning_test:
#                 learning_FR_array.append(-learning_FR)
#             n_cells=n_cells+1
    
#     learning_FR_array=np.array(learning_FR_array)
#     learning_curve=  np.nanmean(learning_FR_array,axis=0)
#     coef = np.polyfit(np.arange(n_learning_trials),learning_curve,1)
#     slope=round(coef[0],2)
#     poly1d_fn = np.poly1d(coef) 
#     plt.title('learning curve '+title_array[array_inx]+' '+str(n_cells)+' cells slope:'+str(slope))
#     plt.plot(learning_curve)
#     plt.plot(poly1d_fn(np.arange(n_learning_trials)))
#     plt.show()
    
#%% leanring and tuning index in washout (yasmin only)                 
# sig_cong_array=np.array([x for x in cong_dict_array if x['sig learned vel learning']  and x['behaviour effect']]  )
# sig_incong_array=np.array([x for x in incong_dict_array if x['sig learned vel learning']  and x['behaviour effect']]  )

# sig_cong_array=np.array([x for x in cong_dict_array if x['sig learned vel learning']  and x['behaviour effect']and x['cell stability learning']]  )
# sig_incong_array=np.array([x for x in incong_dict_array if x['sig learned vel learning']  and x['behaviour effect']and x['cell stability learning'] ]  )

# sig_cong_array=np.array([x for x in cong_dict_array if x['behaviour effect'] ]  )
# sig_incong_array=np.array([x for x in incong_dict_array if   x['behaviour effect'] ])     
                                                                                                                                                                                                                                                                                                                                                                                                
# sig_cong_array=np.array([x for x in cong_dict_array if x['behaviour effect'] and x['cell stability washout'] and  x['sig learned vel learning']]  )
# sig_incong_array=np.array([x for x in incong_dict_array if   x['behaviour effect']and x['cell stability washout']and  x['sig learned vel learning']])                                                                                                                                                                                                                                                                                                          

# legend_size=8
# title_array=['cong blocks','incongruent blocks']
# color_array=['tab:blue','tab:orange']
# p_crit=0.05
# for array_inx,sig_array in enumerate([sig_cong_array,sig_incong_array]):
#     washout_base_FR=[]
#     washout_learned_FR=[]
#     learning_FR=[]
#     n_cells=0
#     for block_inx,cur_block_dict in enumerate(sig_array):
                    
#         mean_washout_base=np.nanmean(cur_block_dict['FR washout base'])
#         mean_washout_learned=np.nanmean(cur_block_dict['FR washout learned'])
#         if np.isnan(mean_washout_learned): #skip fiona cells
#             continue

#         washout_base_FR.append(np.nanmean(cur_block_dict['FR washout base']))
#         washout_learned_FR.append(np.nanmean(cur_block_dict['FR washout learned']))
#         learning_FR.append(np.nanmean(cur_block_dict['FR learning']))
#         n_cells=n_cells+1
        
#     #Index calculations
#     learning_inx=np.array(learning_FR)-np.array(washout_base_FR)
#     tuning_inx=np.array(washout_learned_FR)-np.array(washout_base_FR)
    
#     # Perform orthogonal regression
#     result = stats.linregress(learning_inx, tuning_inx)
#     print(result)
#     # Calculate the orthogonal regression line
#     slope = result.slope
#     intercept = result.intercept
    
#     # Calculate the fitted values
#     fitted_values = slope * learning_inx + intercept
    
#     # Calculate R-squared
#     residuals = tuning_inx - fitted_values
#     ss_residual = np.sum(residuals ** 2)
#     ss_total = np.sum((tuning_inx - np.mean(tuning_inx)) ** 2)
#     r_squared = round(1 - (ss_residual / ss_total),2)

#     #plt.plot(learning_inx_fit, tuning_inx_fit, color='red', label='Orthogonal Regression Line')
#     plt.axline((0,intercept),slope=slope,color='red')
#     stat,p=stats.pearsonr(learning_inx,tuning_inx)
#     p=str(round(p,3))
#     plt.scatter(learning_inx,tuning_inx,color=color_array[array_inx])
#     plt.title(title_array[array_inx]+' '+str(n_cells)+' cells r^2='+str(r_squared))
#     plt.axline((0, 0), (0, 1), linewidth=2, color='black')
#     plt.axline((0, 0), (1, 0), linewidth=2, color='black')
#     plt.axline((0, 0), (1, 1), linewidth=2, color='black')
    
#     plt.xlabel('learning inx')
#     plt.ylabel('tuning inx')
#     #plt.xlim([-40,40])
#     #plt.ylim([-40,40])
#     plt.show()

#%% scatter: tuning in mapping vs tuning in washout in cells significant for learning             

# sig_cong_array=np.array([x for x in cong_dict_array if x['sig learned vel learning'] and  x['behaviour effect']and x['cell stability washout'] ]  )
# sig_incong_array=np.array([x for x in incong_dict_array if x['sig learned vel learning'] and   x['behaviour effect']and x['cell stability washout']])                                                                                                                                                                                                                                                                                                          


# # sig_cong_array=np.array([x for x in cong_dict_array if x['sig learned vel learning']  and x['behaviour effect']and x['cell stability learning']]  )
# # sig_incong_array=np.array([x for x in incong_dict_array if x['sig learned vel learning']  and x['behaviour effect']and x['cell stability learning'] ]  )

# title_array=['cong blocks','incongruent blocks']
# color_array=['tab:blue','tab:orange']


# for array_inx,sig_array in enumerate([sig_cong_array,sig_incong_array]):
#     washout_base_FR=[]
#     washout_learned_FR=[]
#     mapping_base_FR=[]
#     mapping_learned_FR=[]
#     learning_FR=[]
#     n_cells=0
#     for block_inx,cur_block_dict in enumerate(sig_array):

#         mean_mapping_base=np.nanmean(cur_block_dict['FR mapping base'])
#         mean_mapping_learned=np.nanmean(cur_block_dict['FR mapping learned'])
#         mean_washout_base_FR=np.nanmean(cur_block_dict['FR washout base'])
#         mean_washout_learned_FR=np.nanmean(cur_block_dict['FR washout learned'])
#         mean_learning_FR=np.nanmean(cur_block_dict['FR learning'])

#         if np.isnan(mean_washout_learned_FR) or np.isnan(mean_mapping_learned): #skip fiona cells
#             continue
        
#         mapping_base_FR.append(mean_mapping_base)
#         mapping_learned_FR.append(mean_mapping_learned)
#         washout_base_FR.append(mean_washout_base_FR)
#         washout_learned_FR.append(mean_washout_learned_FR)
#         learning_FR.append(mean_learning_FR)

#         n_cells=n_cells+1
        
#     mapping_tuning_array=np.array(mapping_base_FR)-np.array(mapping_learned_FR)
#     washout_tuning_array=np.array(washout_base_FR)-np.array(washout_learned_FR)
    
#     # mapping_tuning_array = mapping_tuning_array[~np.isnan(mapping_tuning_array)]
#     # washout_tuning_array = washout_tuning_array[~np.isnan(washout_tuning_array)]

#     plt.scatter(mapping_tuning_array,washout_tuning_array,color=color_array[array_inx])
#     plt.xlabel('mapping')
#     plt.ylabel('washout')
#     plt.title('base - learned '+title_array[array_inx]+' '+str(n_cells)+' cells')
#     plt.axvline(0,color='red')
#     plt.axhline(0,color='red')
#     #plt.xlim([-40,40])
#     #plt.ylim([-40,40])
#     plt.show()
#%% Simulation - how scatter of random cells look like (learning inx vs tuning inx)

# sig_cong_array=np.array([x for x in cong_dict_array if x['behaviour effect']]  )
# variance_array=[]
# mean_array=[]
# for block_inx,cur_block_dict in enumerate(sig_cong_array):
#     if np.isnan(np.var(cur_block_dict['FR learning'])):
#         continue
#     else:
#         variance_array.append(np.var(cur_block_dict['FR learning']))
#         mean_array.append(np.nanmean(cur_block_dict['FR learning']))


# n_repetitions=1000
# n_cells=1000
# n_learning_trials=72
# n_washout_trials=10

# learning_rep=[]
# washout_base_rep=[]
# washout_learned_rep=[]

# min_sig_cells=n_cells
# p_crit=0.05

# for rep in np.arange(n_repetitions):
    
#     washout_base=np.empty((n_cells, n_washout_trials))
#     washout_learned=np.empty((n_cells, n_washout_trials))
#     learning=np.empty((n_cells, n_learning_trials))
#     washout_base[:]=np.nan
#     washout_learned[:]=np.nan
#     learning[:]=np.nan

#     #find significant cells in washout base vs learning
#     sig_inx=[]
#     for cell_inx in np.arange(n_cells):
#         FR_variance=random.choice(variance_array)

#         random_inx=random.randint(0, len(variance_array)-1)
#         FR_variance=variance_array[random_inx]
#         FR_mean=mean_array[random_inx]

#         washout_base[cell_inx,:]=np.random.normal(FR_mean,FR_variance, size=(1, n_washout_trials))
#         washout_learned[cell_inx,:]=np.random.normal(FR_mean,FR_variance, size=(1, n_washout_trials))
#         learning[cell_inx,:]=np.random.normal(FR_mean,FR_variance, size=(1, n_learning_trials))
        
#         washout_base_cell_test=washout_base[cell_inx,::2]
#         washout_learned_cell=washout_learned[cell_inx,:]
#         learning_cell_test=learning[cell_inx,::2]
#         stat,p=stats.mannwhitneyu(washout_base_cell_test, learning_cell_test)
#         if p<p_crit:
#             sig_inx.append(cell_inx)
#             if np.mean(washout_base_cell_test) <np.mean(learning_cell_test):
#                 washout_base[cell_inx,:]=-washout_base[cell_inx,:]
#                 washout_learned[cell_inx,:]=-washout_learned[cell_inx,:]
#                 learning[cell_inx,:]=-learning[cell_inx,:]

                
#     min_sig_cells=min(len(sig_inx),min_sig_cells)   
#     #remove non significant cells
#     washout_base_sig=washout_base[sig_inx,1::2]
#     washout_learned_sig=washout_learned[sig_inx,1::2]
#     learning_sig=learning[sig_inx,1::2]
    
#     #average across trials
#     washout_base_sig=np.nanmean(washout_base_sig,axis=1)
#     washout_learned_sig=np.nanmean(washout_learned_sig,axis=1)
#     learning_sig=np.nanmean(learning_sig,axis=1)
    
 
#     conditions = ['wB','wL','learning']
#     x_pos = np.arange(len(conditions))
#     averages = [np.nanmean(washout_base_sig), np.nanmean(learning_sig), np.nanmean(washout_learned_sig)]
#     error = [stats.sem(washout_base_sig,nan_policy='omit'), stats.sem(learning_sig,nan_policy='omit'), stats.sem(washout_learned_sig,nan_policy='omit')]
    
#     # plt.bar(conditions, averages, yerr=error, align='center', alpha=0.5, ecolor='black', capsize=10)
#     # plt.show()
      
#     learning_rep.append(np.nanmean(learning_sig))
#     washout_base_rep.append(np.nanmean(washout_base_sig))
#     washout_learned_rep.append(np.nanmean(washout_learned_sig))

# averages_rep=[np.mean(washout_base_rep),np.mean(learning_rep),np.mean(washout_learned_rep)]
# error_rep = [stats.sem(washout_base_rep,nan_policy='omit'), stats.sem(learning_rep,nan_policy='omit'), stats.sem(washout_learned_rep,nan_policy='omit')]

# plt.bar(conditions, averages,yerr=error_rep, align='center', alpha=0.5, ecolor='black', capsize=10)
# plt.show()



#%% leanring vs washout (yasmin only) - random washout learned                 

# sig_cong_array=np.array([x for x in cong_dict_array if x['behaviour effect'] and x['cell stability washout']]  )
# sig_incong_array=np.array([x for x in incong_dict_array if   x['behaviour effect']and x['cell stability washout']])                                                                                                                                                                                                                                                                                                          

# # sig_cong_array=np.array([x for x in cong_dict_array if x['behaviour effect'] ]  )
# # sig_incong_array=np.array([x for x in incong_dict_array if   x['behaviour effect']])      
  
# n_cells_cong=len(sig_cong_array)
# n_cells_incong=len(sig_incong_array)

# legend_size=8
# title_array=['cong blocks','incongruent blocks']
# n_cells_array=[n_cells_cong,n_cells_incong]
# color_array=['tab:blue','tab:orange']


# washout_learned_random_FR_reps=[]
# n_reps=100
# for cur_rep in np.arange(n_reps):
#     for array_inx,sig_array in enumerate([sig_cong_array]):
#         washout_base_psths=np.empty((len(sig_array),len(timecourse)))
#         washout_base_psths[:]=np.nan
#         washout_learned_psths=np.empty((len(sig_array),len(timecourse)))
#         washout_learned_psths[:]=np.nan
#         learning_psths=np.empty((len(sig_array),len(timecourse)))
#         learning_psths[:]=np.nan
#         dishabituation_psths=np.empty((len(sig_array),len(timecourse)))
#         dishabituation_psths[:]=np.nan
#         washout_learned_random_psths=np.empty((len(sig_array),len(timecourse)))
#         washout_learned_random_psths[:]=np.nan
    
#         washout_learned_tuned_psths=np.empty((len(sig_array),len(timecourse)))
#         washout_learned_tuned_psths[:]=np.nan
    
#         washout_base_FR=[]
#         washout_learned_FR=[]
#         learning_FR=[]
#         washout_learned_random_FR=[]
#         n_cells=0
#         for block_inx,cur_block_dict in enumerate(sig_array):
                        
#                 mean_washout=np.nanmean(cur_block_dict['FR washout base test'])
#                 mean_learning=np.nanmean(cur_block_dict['FR learning test'])
                
#                 if cur_block_dict['cell_ID']<cutoff_cell:
#                     continue
#                 if mean_washout> mean_learning:
#                     learning_array=cur_block_dict['PSTH learning']
#                     washout_array=cur_block_dict['PSTH washout base']
#                     washout_learned_array=cur_block_dict['PSTH washout learned']
#                     cur_washout_base_FR=np.nanmean(cur_block_dict['FR washout base'])
#                     cur_washout_learned_FR=np.nanmean(cur_block_dict['FR washout learned'])
#                     washout_base_FR.append(cur_washout_base_FR)
#                     washout_learned_FR.append(cur_washout_learned_FR)
#                     learning_FR.append(np.nanmean(cur_block_dict['FR learning']))
#                 elif mean_washout< mean_learning:
#                     learning_array=-cur_block_dict['PSTH learning']
#                     washout_array=-cur_block_dict['PSTH washout base']
#                     washout_learned_array=-cur_block_dict['PSTH washout learned']
#                     cur_washout_base_FR=-np.nanmean(cur_block_dict['FR washout base'])
#                     cur_washout_learned_FR=-np.nanmean(cur_block_dict['FR washout learned'])
                    
#                     washout_base_FR.append(cur_washout_base_FR)
#                     washout_learned_FR.append(cur_washout_learned_FR)
#                     learning_FR.append(-np.nanmean(cur_block_dict['FR learning']))
    
    
    
#                 washout_learned_random_psths[block_inx,:]=washout_array+np.abs(washout_array-washout_learned_array)*random.choice([-1, 1])
#                 washout_learned_tuned_psths[block_inx,:]=washout_array-np.abs(washout_array-washout_learned_array)
#                 washout_learned_random_FR.append(cur_washout_base_FR+np.abs(cur_washout_base_FR-cur_washout_learned_FR)*random.choice([-1, 1]))
#                 washout_base_psths[block_inx,:]=washout_array
#                 washout_learned_psths[block_inx,:]=washout_learned_array
#                 learning_psths[block_inx,:]=learning_array
#                 #n_cells=n_cells+1    
    
#         #check wether FR in 100-300 between mapping base and learning is significantly different         
#         # stat,p_learning_base=stats.wilcoxon(washout_base_FR,learning_FR)
#         # p_learning_base=round(p_learning_base,3)
        
#         # plt.plot(timecourse,np.nanmean(learning_psths,axis=0),color=color_array[array_inx])
#         # plt.plot(timecourse,np.nanmean(washout_learned_psths,axis=0),color='tab:green',linestyle='dashed')
#         # plt.plot(timecourse,np.nanmean(washout_learned_random_psths,axis=0),color='tab:red',linestyle='dashed')
#         # #plt.plot(timecourse,np.nanmean(washout_learned_tuned_psths,axis=0),color='tab:pink',linestyle='dashed')
#         # plt.plot(timecourse,np.nanmean(washout_base_psths,axis=0),color='tab:green',linestyle='dotted')
#         # plt.axvline(x=100,color='black')
#         # plt.axvline(x=300,color='black')
#         # plt.legend(['learning','wL','wB'], loc ="lower right",fontsize=legend_size)
#         # plt.title(title_array[array_inx]+' '+str(n_cells)+' cells ,p='+str(p_learning_base))
#         # plt.xlabel('time from MO')
#         # plt.ylabel('FR')
#         # plt.show()
    
#         washout_learned_random_FR_reps.append(np.nanmean(washout_learned_random_FR))

# washout_learned_random_FR_reps_sorted=np.sort(washout_learned_random_FR_reps)
# critical_index=int(np.round(0.05*n_reps))

# plt.hist(washout_learned_random_FR_reps)
# plt.axvline(np.nanmean(washout_learned_FR),color='red')
# plt.axvline(washout_learned_random_FR_reps_sorted[critical_index],color='black')
# plt.ylabel('N cells')
# plt.show()

#%%Number of significant cells tuned in washout or mapping 

# sig_cong_array2=np.array([x for x in cong_dict_array if  x['cell stability washout2'] and x['behaviour effect'] and x['sig tuning washout']]  )
# sig_incong_array2=np.array([x for x in incong_dict_array if  x['cell stability washout2']  and x['behaviour effect']and x['sig tuning washout']]  )

# sig_cong_array2=np.array([x for x in cong_dict_array if  x['cell stability washout2'] and x['cell stability washout']  and x['behaviour effect'] and x['sig tuning washout']]  )
# sig_incong_array2=np.array([x for x in incong_dict_array if  x['cell stability washout2'] and x['cell stability washout']  and x['behaviour effect']and x['sig tuning washout']]  )

# # sig_cong_array2=np.array([x for x in cong_dict_array if  x['cell stability washout2'] and x['behaviour effect'] and x['sig mapping']]  )
# # sig_incong_array2=np.array([x for x in incong_dict_array if  x['cell stability washout2']  and x['behaviour effect']and x['sig mapping']]  )

# #select yasmin cells only:
# sig_cong_array2=[x for x in sig_cong_array2 if x['cell_ID']>cutoff_cell]
# sig_incong_array2=[x for x in sig_incong_array2 if x['cell_ID']>cutoff_cell]


# n_cells_cong2=len(sig_cong_array2)
# n_cells_incong2=len(sig_incong_array2)
# plt.bar(['congruent blocks','incongruent blocks'],[n_cells_cong2/len(sig_cong_array),n_cells_incong2/len(sig_incong_array)])    
#plt.show()
# plt.bar(['congruent blocks','incongruent blocks'],[n_cells_cong2/len(sig_cong_array),n_cells_incong2/len(sig_incong_array)])    
# plt.show()