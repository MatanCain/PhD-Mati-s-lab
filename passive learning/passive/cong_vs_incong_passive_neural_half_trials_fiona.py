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
win_begin_PSTH2=-200
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
cutoff_cell=8229 #cutoff between yasmin and fiona

window_begin=100
window_end=300
for learning_task_inx,learning_task in enumerate(learning_tasks):
    cur_dict_array=learning_dict[learning_task_inx]
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
                    #select the relevant direction as learned direction in the previous washout block
                    if learned_direction=='CW':
                        learned_direction_int=(block_base_dir-90)%360
                    elif learned_direction=='CCW':
                        learned_direction_int=(block_base_dir+90)%360
                  
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
                        dictFilterTrials_learning_test = {'dir':'filterOff', 'trial_name':trial_type_learning, 'fail':0, 'block_begin_end':[begin_inx,end_inx],'even_odd_rows':'odd'}
                        dictFilterTrials_dishabituation_test = { 'dir':'filterOff', 'fail':0,'files_begin_end':[file_begin_dishabituation,file_end_dishabituation],'trial_name':'d0'+trial_type_learning,'even_odd_rows':'odd'}

                        FR_learning_test=cur_cell_learning.get_mean_FR_event(dictFilterTrials_learning_test,'motion_onset',window_pre=window_begin,window_post=window_end)
                        FR_dishabituation_test=cur_cell_dishabituation.get_mean_FR_event(dictFilterTrials_dishabituation_test,'motion_onset',window_pre=window_begin,window_post=window_end)
                        stat,p_learning=stats.mannwhitneyu(FR_learning_test, FR_dishabituation_test)
                        if p_learning>0.05:
                            continue
                    except:
                        continue
 
                    #PSTHs in learning and dishabituation blocks
                    try:
                        window_PSTH={"timePoint":cur_event,'timeBefore':win_begin_PSTH,'timeAfter':win_end_PSTH,'even_odd_rows':'even'}          
                        dictFilterTrials_learning_PSTH = {'dir':'filterOff', 'trial_name':trial_type_learning, 'fail':0, 'block_begin_end':[begin_inx,end_inx],'even_odd_rows':'even'}
                        psth_learning=cur_cell_learning.PSTH(window_PSTH,dictFilterTrials_learning_PSTH)
                        FR_learning=cur_cell_learning.get_mean_FR_event(dictFilterTrials_learning_PSTH,'motion_onset',window_pre=window_begin,window_post=window_end)
                        dictFilterTrials_dishabituation_PSTH = { 'screen_rot':'filterOff', 'fail':0,'files_begin_end':[file_begin_dishabituation,file_end_dishabituation],'trial_name':'d0'+trial_type_learning,'even_odd_rows':'even'}
                        psth_dishabituation=cur_cell_dishabituation.PSTH(window_PSTH,dictFilterTrials_dishabituation_PSTH)  
                        FR_dishabituation=cur_cell_dishabituation.get_mean_FR_event(dictFilterTrials_dishabituation_PSTH,'motion_onset',window_pre=window_begin,window_post=window_end)
                    except:
                        continue
                        
                    #stability dishabituation - check whether the correlation in FR before MO is correlated between washout and subsequent learning block                        
                    try:
                        dictFilterTrials_dishabituation_stability = { 'dir':'filterOff', 'fail':0,'files_begin_end':[file_begin_dishabituation,file_end_dishabituation],'trial_name':'d0'+trial_type_learning}
                        FR_learning_dishabituation_baseline=cur_cell_dishabituation.get_mean_FR_event(dictFilterTrials_dishabituation_stability,'motion_onset',window_pre=-800,window_post=0)
                        dictFilterTrials_learning_stability = {'dir':'filterOff', 'fail':0, 'block_begin_end':[begin_inx,end_inx], 'trial_name':trial_type_learning}
                        FR_learning_baseline=cur_cell_learning.get_mean_FR_event(dictFilterTrials_learning_stability,'motion_onset',window_pre=-800,window_post=0)
                        #stability based on ttest
                        stat,p_stability_learning=stats.mannwhitneyu(FR_learning_dishabituation_baseline, FR_learning_baseline)
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
                            
                    block_dict=dict.fromkeys(['cell_ID', 'base direction', 'learned direction', 'PSTH learning','PSTH dishabituation','FR learning baseline','FR dishabituation baseline'\
                                              ,'FR learning','FR learning test','FR dishabituation','FR dishabituation test','behaviour effect'])
                    block_dict['cell_ID']=cell_ID
                    block_dict['base direction']=block_base_dir
                    block_dict['learned direction']=learned_direction
                    block_dict['PSTH learning']=psth_learning
                    block_dict['PSTH dishabituation']=psth_dishabituation
                    block_dict['FR learning']=FR_learning           
                    block_dict['FR learning test']=FR_learning_test           
                    block_dict['FR dishabituation']=FR_dishabituation           
                    block_dict['FR dishabituation test']=FR_dishabituation_test  
                    block_dict['behaviour effect']=behaviour_effect #1 if cell is significant for learned velocity betwenn washout base and learning block (100-300)
    
                    cur_dict_array.append(block_dict)

#%%#%% Number of significant cells 

sig_cong_array=np.array([x for x in cong_dict_array if  x['behaviour effect']]  )
sig_incong_array=np.array([x for x in incong_dict_array if x['behaviour effect']]  )

n_cells_cong=len(sig_cong_array)
n_cells_incong=len(sig_incong_array)
print(n_cells_cong,n_cells_incong)

plt.bar(['congruent blocks','incongruent blocks'],[n_cells_cong,n_cells_incong])
plt.show()

#%% leanring vs dishabituation                
                                                                                                                                                                                                                                                                                                     
sig_cong_array=np.array([x for x in cong_dict_array if x['behaviour effect']]  )
sig_incong_array=np.array([x for x in incong_dict_array if   x['behaviour effect']])    

n_cells_cong=len(sig_cong_array)
n_cells_incong=len(sig_incong_array)

legend_size=8
title_array=['cong blocks','incongruent blocks']
n_cells_array=[n_cells_cong,n_cells_incong]
color_array=['tab:blue','tab:orange']

dishabituation_congruent=[]
learning_congruent=[]
dishabituation_incongruent=[]
learning_incongruent=[]

for array_inx,sig_array in enumerate([sig_cong_array,sig_incong_array]):
    dishabituation_psths=np.empty((len(sig_array),len(timecourse)))
    dishabituation_psths[:]=np.nan
    learning_psths=np.empty((len(sig_array),len(timecourse)))
    learning_psths[:]=np.nan

    dishabituation_FR=[]
    learning_FR=[]
    n_cells=0
    for block_inx,cur_block_dict in enumerate(sig_array):
                    
            mean_dishabituation=np.nanmean(cur_block_dict['FR dishabituation test'])
            mean_learning=np.nanmean(cur_block_dict['FR learning test'])

            if mean_dishabituation> mean_learning:
                learning_array=cur_block_dict['PSTH learning']
                dishabituation_array=cur_block_dict['PSTH dishabituation']
                learning_FR.append(np.nanmean(cur_block_dict['FR learning']))
                dishabituation_FR.append(np.nanmean(cur_block_dict['FR dishabituation']))
            elif mean_dishabituation< mean_learning:
                learning_array=-cur_block_dict['PSTH learning']
                dishabituation_array=-cur_block_dict['PSTH dishabituation']
                learning_FR.append(-np.nanmean(cur_block_dict['FR learning']))
                dishabituation_FR.append(-np.nanmean(cur_block_dict['FR dishabituation']))
                
            n_cells=n_cells+1
            dishabituation_psths[block_inx,:]=dishabituation_array
            learning_psths[block_inx,:]=learning_array


    #check wether FR in 100-300 between mapping base and learning is significantly different         
    stat,p_learning_base=stats.wilcoxon(learning_FR,dishabituation_FR)
    p_learning_base=round(p_learning_base,3)
    
    plt.plot(timecourse,np.nanmean(learning_psths,axis=0),color=color_array[array_inx])
    plt.plot(timecourse,np.nanmean(dishabituation_psths,axis=0),color=color_array[array_inx],linestyle='dotted')
    plt.axvline(x=100,color='black')
    plt.axvline(x=300,color='black')
    plt.legend(['learning','dishabituation'], loc ="lower right",fontsize=legend_size)
    plt.title(title_array[array_inx]+' '+str(n_cells)+' cells ,p='+str(p_learning_base))
    plt.xlabel('time from MO')
    plt.ylabel('FR')
    plt.show()
    
    plt.scatter(learning_FR,dishabituation_FR,color=color_array[array_inx])
    plt.axline((0,0),(1,1),color='black')
    plt.title(title_array[array_inx]+' '+str(n_cells)+' cells ,p='+str(p_learning_base))
    plt.xlabel('FR learning')
    plt.ylabel('FR dishabituation')
    plt.show()

    if array_inx==0:
        washout_base_congruent=np.nanmean(dishabituation_psths[:,-win_begin_PSTH+100:-win_begin_PSTH+300],axis=1)
        learning_congruent=np.nanmean(learning_psths[:,-win_begin_PSTH+100:-win_begin_PSTH+300],axis=1)
    if array_inx==1:
        washout_base_incongruent=np.nanmean(dishabituation_psths[:,-win_begin_PSTH+100:-win_begin_PSTH+300],axis=1)
        learning_incongruent=np.nanmean(learning_psths[:,-win_begin_PSTH+100:-win_begin_PSTH+300],axis=1)
        
x=washout_base_congruent-learning_congruent
y=washout_base_incongruent-learning_incongruent
x=x[~np.isnan(x)]
y=y[~np.isnan(y)]

stat,p_washout_learning_distance=stats.mannwhitneyu(x,y)
print(p_washout_learning_distance)    

#%% leanring vs dishabituation   with baseline subtraction         


sig_cong_array=np.array([x for x in cong_dict_array if x['behaviour effect']and x['cell stability dishabituation']]  )
sig_incong_array=np.array([x for x in incong_dict_array if   x['behaviour effect']and x['cell stability dishabituation']])                                                                                                                                                                                                                                                                                                          

# sig_cong_array=np.array([x for x in cong_dict_array if x['behaviour effect']]  )
# sig_incong_array=np.array([x for x in incong_dict_array if   x['behaviour effect']])   
#select fiona cells only:
sig_cong_array=[x for x in sig_cong_array if x['cell_ID']<cutoff_cell]
sig_incong_array=[x for x in sig_incong_array if x['cell_ID']<cutoff_cell]

#select yasmin cells only:
# sig_cong_array=[x for x in sig_cong_array if x['cell_ID']>cutoff_cell]
# sig_incong_array=[x for x in sig_incong_array if x['cell_ID']>cutoff_cell]

n_cells_cong=len(sig_cong_array)
n_cells_incong=len(sig_incong_array)

legend_size=8
title_array=['cong blocks','incongruent blocks']
n_cells_array=[n_cells_cong,n_cells_incong]
color_array=['tab:blue','tab:orange']

dishabituation_congruent=[]
learning_congruent=[]
dishabituation_incongruent=[]
learning_incongruent=[]

for array_inx,sig_array in enumerate([sig_cong_array,sig_incong_array]):
    dishabituation_psths=np.empty((len(sig_array),len(timecourse)))
    dishabituation_psths[:]=np.nan
    learning_psths=np.empty((len(sig_array),len(timecourse)))
    learning_psths[:]=np.nan

    dishabituation_FR=[]
    learning_FR=[]
    n_cells=0
    for block_inx,cur_block_dict in enumerate(sig_array):
                    
            mean_dishabituation=np.nanmean(cur_block_dict['FR dishabituation test'])
            mean_learning=np.nanmean(cur_block_dict['FR learning test'])


            if mean_dishabituation> mean_learning:
                learning_array=cur_block_dict['PSTH learning']-np.nanmean(cur_block_dict['FR learning baseline'])
                dishabituation_array=cur_block_dict['PSTH dishabituation']-np.nanmean(cur_block_dict['FR dishabituation baseline'])
                learning_FR.append(np.nanmean(cur_block_dict['FR learning'])-np.nanmean(cur_block_dict['FR learning baseline']))
                dishabituation_FR.append(np.nanmean(cur_block_dict['FR dishabituation'])-np.nanmean(cur_block_dict['FR dishabituation baseline']))
            elif mean_dishabituation< mean_learning:
                learning_array=-cur_block_dict['PSTH learning']+np.nanmean(cur_block_dict['FR learning baseline'])
                dishabituation_array=-cur_block_dict['PSTH dishabituation']+np.nanmean(cur_block_dict['FR dishabituation baseline'])
                learning_FR.append(-np.nanmean(cur_block_dict['FR learning'])+np.nanmean(cur_block_dict['FR learning baseline']))
                dishabituation_FR.append(-np.nanmean(cur_block_dict['FR dishabituation'])+np.nanmean(cur_block_dict['FR dishabituation baseline']))
                
            n_cells=n_cells+1
            dishabituation_psths[block_inx,:]=dishabituation_array
            learning_psths[block_inx,:]=learning_array


    #check wether FR in 100-300 between mapping base and learning is significantly different         
    stat,p_learning_base=stats.wilcoxon(learning_FR,dishabituation_FR)
    p_learning_base=round(p_learning_base,3)
    
    plt.plot(timecourse,np.nanmean(learning_psths,axis=0),color=color_array[array_inx])
    plt.plot(timecourse,np.nanmean(dishabituation_psths,axis=0),color=color_array[array_inx],linestyle='dotted')
    plt.axvline(x=100,color='black')
    plt.axvline(x=300,color='black')
    plt.legend(['learning','dishabituation'], loc ="lower right",fontsize=legend_size)
    plt.title(title_array[array_inx]+' '+str(n_cells)+' cells')
    plt.xlabel('time from MO')
    plt.ylabel('FR')
    plt.show()


    plt.scatter(learning_FR,dishabituation_FR,color=color_array[array_inx])
    plt.axline((0,0),(1,1),color='black')
    plt.title(title_array[array_inx]+' '+str(n_cells)+' cells ,p='+str(p_learning_base))
    plt.xlabel('FR learning')
    plt.ylabel('FR dishabituation')
    plt.show()
 
#%% leanring vs dishabituation                


sig_cong_array=np.array([x for x in cong_dict_array if x['behaviour effect']and x['cell stability dishabituation']]  )
sig_incong_array=np.array([x for x in incong_dict_array if   x['behaviour effect']and x['cell stability dishabituation']])                                                                                                                                                                                                                                                                                                          

# sig_cong_array=np.array([x for x in cong_dict_array if x['behaviour effect']]  )
# sig_incong_array=np.array([x for x in incong_dict_array if   x['behaviour effect']])    

#select fiona cells only:
sig_cong_array=[x for x in sig_cong_array if x['cell_ID']<cutoff_cell]
sig_incong_array=[x for x in sig_incong_array if x['cell_ID']<cutoff_cell]

#select yasmin cells only:
# sig_cong_array=[x for x in sig_cong_array if x['cell_ID']>cutoff_cell]
# sig_incong_array=[x for x in sig_incong_array if x['cell_ID']>cutoff_cell]

n_cells_cong=len(sig_cong_array)
n_cells_incong=len(sig_incong_array)

legend_size=8
title_array=['cong blocks','incongruent blocks']
n_cells_array=[n_cells_cong,n_cells_incong]
color_array=['tab:blue','tab:orange']


for array_inx,sig_array in enumerate([sig_cong_array,sig_incong_array]):
    dishabituation_psths=np.empty((len(sig_array),len(timecourse)))
    dishabituation_psths[:]=np.nan
    learning_psths=np.empty((len(sig_array),len(timecourse)))
    learning_psths[:]=np.nan
    mapping_base_psths=np.empty((len(sig_array),len(timecourse)))
    mapping_base_psths[:]=np.nan
    mapping_learned_psths=np.empty((len(sig_array),len(timecourse)))
    mapping_learned_psths[:]=np.nan

    dishabituation_FR=[]
    learning_FR=[]
    mapping_base_FR=[]
    mapping_learned_FR=[]   
    n_cells=0
    for block_inx,cur_block_dict in enumerate(sig_array):
                    
            mean_dishabituation=np.nanmean(cur_block_dict['FR dishabituation test'])
            mean_learning=np.nanmean(cur_block_dict['FR learning test'])
            if mean_dishabituation> mean_learning:
                
                learning_array=cur_block_dict['PSTH learning']
                dishabituation_array=cur_block_dict['PSTH dishabituation']
                mapping_base_array=dishabituation_array=cur_block_dict['PSTH mapping base']
                mapping_learned_array=dishabituation_array=cur_block_dict['PSTH mapping learned']
                learning_FR.append(np.nanmean(cur_block_dict['FR learning']))
                dishabituation_FR.append(np.nanmean(cur_block_dict['FR dishabituation']))
                mapping_base_FR.append(np.nanmean(cur_block_dict['FR mapping base']))
                mapping_learned_FR.append(np.nanmean(cur_block_dict['FR mapping learned']))

            elif mean_dishabituation< mean_learning:
                learning_array=-cur_block_dict['PSTH learning']
                dishabituation_array=-cur_block_dict['PSTH dishabituation']
                mapping_base_array=dishabituation_array=-cur_block_dict['PSTH mapping base']
                mapping_learned_array=dishabituation_array=-cur_block_dict['PSTH mapping learned']
                learning_FR.append(-np.nanmean(cur_block_dict['FR learning']))
                dishabituation_FR.append(-np.nanmean(cur_block_dict['FR dishabituation']))
                mapping_base_FR.append(-np.nanmean(cur_block_dict['FR mapping base']))
                mapping_learned_FR.append(-np.nanmean(cur_block_dict['FR mapping learned']))                
            n_cells=n_cells+1
            dishabituation_psths[block_inx,:]=dishabituation_array
            learning_psths[block_inx,:]=learning_array
            mapping_base_psths[block_inx,:]=mapping_base_array
            mapping_learned_psths[block_inx,:]=mapping_learned_array


    #check wether FR in 100-300 between mapping base and learning is significantly different         
    stat,p_learning_base=stats.wilcoxon(learning_FR,dishabituation_FR)
    p_learning_base=round(p_learning_base,3)
    
    plt.plot(timecourse,np.nanmean(learning_psths,axis=0),color=color_array[array_inx])
    plt.plot(timecourse,np.nanmean(mapping_base_psths,axis=0),color=color_array[array_inx],linestyle='dotted')
    plt.plot(timecourse,np.nanmean(mapping_learned_psths,axis=0),color=color_array[array_inx],linestyle='dashed')
    plt.axvline(x=100,color='black')
    plt.axvline(x=300,color='black')
    plt.legend(['learning','mB','mL'], loc ="lower right",fontsize=legend_size)
    plt.title(title_array[array_inx]+' '+str(n_cells)+' cells ,p='+str(p_learning_base))
    plt.xlabel('time from MO')
    plt.ylabel('FR')
    plt.show()
    

#%% leanring vs washout (yasmin only)                 
# sig_cong_array=np.array([x for x in cong_dict_array if x['sig learned vel learning']  and x['behaviour effect']]  )
# sig_incong_array=np.array([x for x in incong_dict_array if x['sig learned vel learning']  and x['behaviour effect']]  )

# sig_cong_array=np.array([x for x in cong_dict_array if x['sig learned vel learning']  and x['behaviour effect']and x['cell stability learning']]  )
# sig_incong_array=np.array([x for x in incong_dict_array if x['sig learned vel learning']  and x['behaviour effect']and x['cell stability learning'] ]  )

# sig_cong_array=np.array([x for x in cong_dict_array if x['behaviour effect'] and  x['cell stability dishabituation']]  )
# sig_incong_array=np.array([x for x in incong_dict_array if   x['behaviour effect']and  x['cell stability dishabituation']])                                                                                                                                                                                                                                                                                                          

# n_cells_cong=len(sig_cong_array)
# n_cells_incong=len(sig_incong_array)

# legend_size=8
# title_array=['cong blocks','incongruent blocks']
# n_cells_array=[n_cells_cong,n_cells_incong]
# color_array=['tab:blue','tab:orange']

# washout_base_congruent=[]
# learning_congruent=[]
# washout_base_incongruent=[]
# learning_incongruent=[]

# for array_inx,sig_array in enumerate([sig_cong_array,sig_incong_array]):
#     washout_base_psths=np.empty((len(sig_array),len(timecourse)))
#     washout_base_psths[:]=np.nan
#     washout_learned_psths=np.empty((len(sig_array),len(timecourse)))
#     washout_learned_psths[:]=np.nan
#     learning_psths=np.empty((len(sig_array),len(timecourse)))
#     learning_psths[:]=np.nan
#     dishabituation_psths=np.empty((len(sig_array),len(timecourse)))
#     dishabituation_psths[:]=np.nan

#     washout_base_FR=[]
#     washout_learned_FR=[]
#     learning_FR=[]
#     n_cells=0
#     for block_inx,cur_block_dict in enumerate(sig_array):
                    
#             mean_dishabituation=np.nanmean(cur_block_dict['FR dishabituation test'])
#             mean_learning=np.nanmean(cur_block_dict['FR learning test'])
            
#             if cur_block_dict['cell_ID']<cutoff_cell:
#                 continue
#             if mean_dishabituation> mean_learning:
#                 learning_array=cur_block_dict['PSTH learning']
#                 washout_array=cur_block_dict['PSTH washout base']
#                 washout_learned_array=cur_block_dict['PSTH washout learned']
#                 washout_base_FR.append(np.nanmean(cur_block_dict['FR washout base'])-np.nanmean(cur_block_dict['FR washout base baseline']))
#                 washout_learned_FR.append(np.nanmean(cur_block_dict['FR washout learned'])-np.nanmean(cur_block_dict['FR washout learned baseline']))
#                 learning_FR.append(np.nanmean(cur_block_dict['FR learning'])-np.nanmean(cur_block_dict['FR learning baseline']))
#                 dishabituation_FR.append(np.nanmean(cur_block_dict['FR dishabituation'])-np.nanmean(cur_block_dict['FR dishabituation baseline']))
#             elif mean_dishabituation< mean_learning:
#                 learning_array=-cur_block_dict['PSTH learning']
#                 washout_array=-cur_block_dict['PSTH washout base']
#                 washout_learned_array=-cur_block_dict['PSTH washout learned']
#                 dishabituation_array=-cur_block_dict['PSTH dishabituation']
#                 washout_base_FR.append(-np.nanmean(cur_block_dict['FR washout base'])+np.nanmean(cur_block_dict['FR washout base baseline']))
#                 washout_learned_FR.append(-np.nanmean(cur_block_dict['FR washout learned'])+np.nanmean(cur_block_dict['FR washout learned baseline']))
#                 learning_FR.append(-np.nanmean(cur_block_dict['FR learning'])+np.nanmean(cur_block_dict['FR learning baseline']))
#                 dishabituation_FR.append(-np.nanmean(cur_block_dict['FR dishabituation'])+np.nanmean(cur_block_dict['FR dishabituation baseline']))

#             washout_base_psths[block_inx,:]=washout_array
#             washout_learned_psths[block_inx,:]=washout_learned_array
#             learning_psths[block_inx,:]=learning_array
#             dishabituation_psths[block_inx,:]=dishabituation_array
#             n_cells=n_cells+1

    

#     #check wether FR in 100-300 between mapping base and learning is significantly different         
#     stat,p_learning_base=stats.wilcoxon(washout_base_FR,learning_FR)
#     p_learning_base=round(p_learning_base,3)
    
#     plt.plot(timecourse,np.nanmean(learning_psths,axis=0),color=color_array[array_inx])
#     plt.plot(timecourse,np.nanmean(washout_learned_psths,axis=0),color='tab:green',linestyle='dashed')
#     #plt.plot(timecourse,np.nanmean(dishabituation_psths,axis=0),color='tab:red',linestyle='dotted')
#     plt.plot(timecourse,np.nanmean(washout_base_psths,axis=0),color='tab:green',linestyle='dotted')
#     plt.axvline(x=100,color='black')
#     plt.axvline(x=300,color='black')
#     plt.legend(['learning','wL','wB'], loc ="lower right",fontsize=legend_size)
#     plt.title(title_array[array_inx]+' '+str(n_cells)+' cells ,p='+str(p_learning_base))
#     plt.xlabel('time from MO')
#     plt.ylabel('FR')
#     plt.show()

#     plt.scatter(learning_FR,washout_base_FR,color=color_array[array_inx])
#     plt.axline((0,0),(1,1),color='black')
#     plt.title(title_array[array_inx]+' '+str(n_cells)+' cells ,p='+str(p_learning_base))
#     plt.xlabel('FR learning')
#     plt.ylabel('FR washout base')
#     plt.show()

#%% dishabituation and washout

# sig_cong_array=np.array([x for x in cong_dict_array if x['sig learned vel learning']  and x['behaviour effect']]  )
# sig_incong_array=np.array([x for x in incong_dict_array if x['sig learned vel learning']  and x['behaviour effect']]  )

# sig_cong_array=np.array([x for x in cong_dict_array if x['sig learned vel learning']  and x['behaviour effect']and x['cell stability learning']]  )
# sig_incong_array=np.array([x for x in incong_dict_array if x['sig learned vel learning']  and x['behaviour effect']and x['cell stability learning'] ]  )

# sig_cong_array=np.array([x for x in cong_dict_array if x['behaviour effect'] ]  )
# sig_incong_array=np.array([x for x in incong_dict_array if   x['behaviour effect']])                                                                                                                                                                                                                                                                                                          

# n_cells_cong=len(sig_cong_array)
# n_cells_incong=len(sig_incong_array)

# legend_size=8
# title_array=['cong blocks','incongruent blocks']
# n_cells_array=[n_cells_cong,n_cells_incong]
# color_array=['tab:blue','tab:orange']

# for array_inx,sig_array in enumerate([sig_cong_array,sig_incong_array]):

#     dishabituation_psths=np.empty((len(sig_array),len(timecourse)))
#     dishabituation_psths[:]=np.nan
#     washout_base_psths=np.empty((len(sig_array),len(timecourse)))
#     washout_base_psths[:]=np.nan
#     washout_base_FR=[]
#     dishabituation_FR=[]
#     n_cells=0
#     for block_inx,cur_block_dict in enumerate(sig_array):
                    
#         FR_dishabituation=np.nanmean(cur_block_dict['FR dishabituation'])
#         FR_washout_base=np.nanmean(cur_block_dict['FR washout base'])
        
        
#         if np.isnan(FR_dishabituation) or np.isnan(FR_washout_base):
#             continue
#         washout_base_FR.append(FR_washout_base)
#         dishabituation_FR.append(FR_dishabituation)
#         n_cells=n_cells+1
        

#     plt.scatter(dishabituation_FR,washout_base_FR,color=color_array[array_inx])
#     plt.axline((0,0),(1,1),color='black')
#     plt.title(title_array[array_inx]+' '+str(n_cells)+' cells')
#     plt.xlabel('dis')
#     plt.ylabel('washout base')
#     plt.show()
    
    
