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
#%% In this part of the script, we create list of PSTHS fot the different passivge learning blocks and dishabituation
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
trial_type_mapping='v20a'

mapping_tasks=['8dir_active_passive_interleaved','8dir_active_passive_interleaved_100_25']
dishabituation_tasks=['Dishabituation','Dishabituation_100_25_cue']
cutoff_cell=8229 #cutoff between yasmin and fiona

window_begin=100
window_end=300

crit_learning_value=0.05
n=0
not_stab=0
no_mapping=0
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
                n=n+1
                learning_block_df=trials_df_learning.iloc[np.arange(begin_inx,end_inx+1)]
                
                #Base and learned directions
                block_base_dir=learning_block_df.iloc[0]['screen_rotation']#Find base direction of curren block
                #select the relevant direction as learned direction in the previous dishabituation block
                if learned_direction=='CW':
                    learning_direction_int=(block_base_dir-90)%360
                elif learned_direction=='CCW':
                    learning_direction_int=(block_base_dir+90)%360
              
                #dishabituation block befire the learning blocks
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
                    dictFilterTrials_dishabituation_test = { 'files_begin_end':[file_begin_dishabituation,file_end_dishabituation], 'fail':0,'trial_name':'d0'+trial_type_learning}
                    dictFilterTrials_learning_test = {'dir':'filterOff', 'trial_name':trial_type_learning, 'fail':0, 'block_begin_end':[begin_inx,end_inx]}

                    FR_dishabituation_test=cur_cell_dishabituation.get_mean_FR_event(dictFilterTrials_dishabituation_test,'motion_onset',window_pre=window_begin,window_post=window_end)
                    FR_learning_test=cur_cell_learning.get_mean_FR_event(dictFilterTrials_learning_test,'motion_onset',window_pre=window_begin,window_post=window_end)
                    
                    stat,p_learning=stats.mannwhitneyu(FR_learning_test, FR_dishabituation_test)
                    if p_learning>crit_learning_value:
                        continue
                except:
                    continue

                #Stability dishabituation - check whether the correlation in FR before MO is correlated between dishabituation and subsequent learning block
                try:
                    dictFilterTrials_dishabituation_stability = { 'dir':'filterOff', 'fail':0,'files_begin_end':[file_begin_dishabituation,file_end_dishabituation],'trial_name':'d0'+trial_type_learning}
                    FR_learning_dishabituation_baseline=cur_cell_dishabituation.get_mean_FR_event(dictFilterTrials_dishabituation_stability,'motion_onset',window_pre=-800,window_post=0)
                    dictFilterTrials_learning_stability = {'dir':'filterOff', 'fail':0, 'block_begin_end':[begin_inx,end_inx], 'trial_name':trial_type_learning}
                    FR_learning_baseline2=cur_cell_learning.get_mean_FR_event(dictFilterTrials_learning_stability,'motion_onset',window_pre=-800,window_post=0)
                    r,p_stability_learning=stats.mannwhitneyu(FR_learning_baseline2, FR_learning_dishabituation_baseline)
                    if p_stability_learning<0.05:
                        not_stab=not_stab+1
                        #continue
                except:
                    continue                     

                #mapping task: PSTH and FR in base direction
                window_PSTH={"timePoint":cur_event,'timeBefore':win_begin_PSTH,'timeAfter':win_end_PSTH}   
                try:
                    cur_cell_mapping=load_cell_task(cell_task_py_folder,mapping_task,cell_ID)
                    trialname_base='d'+str(block_base_dir)+trial_type_mapping
                    dictFilterTrials_mapping_base = {'dir':'filterOff', 'trial_name':trialname_base, 'fail':0}
                    psth_mapping_base=cur_cell_mapping.PSTH(window_PSTH,dictFilterTrials_mapping_base)
                    FR_mapping_base=cur_cell_mapping.get_mean_FR_event(dictFilterTrials_mapping_base,'motion_onset',window_pre=window_begin,window_post=window_end)
                    #PSTH and FR in learned direction
                    trialname_learned='d'+str(learning_direction_int)+trial_type_mapping
                    dictFilterTrials_mapping_learned = {'dir':'filterOff', 'trial_name':trialname_learned, 'fail':0}
                    psth_mapping_learned=cur_cell_mapping.PSTH(window_PSTH,dictFilterTrials_mapping_learned)
                    FR_mapping_learned=cur_cell_mapping.get_mean_FR_event(dictFilterTrials_mapping_learned,'motion_onset',window_pre=window_begin,window_post=window_end)
                    #PSTH and FR in null of learned direction
                    null_learning_direction_int=int((learning_direction_int+180)%360)
                    trialname_null_learned='d'+str(null_learning_direction_int)+trial_type_mapping
                    dictFilterTrials_mapping_null_learned = {'dir':'filterOff', 'trial_name':trialname_null_learned, 'fail':0}
                    psth_mapping_null_learned=cur_cell_mapping.PSTH(window_PSTH,dictFilterTrials_mapping_null_learned)
                    FR_mapping_null_learned=cur_cell_mapping.get_mean_FR_event(dictFilterTrials_mapping_null_learned,'motion_onset',window_pre=window_begin,window_post=window_end)
                except:
                    no_mapping=no_mapping+1
                    #continue
                    
                block_dict=dict.fromkeys(['cell_ID', 'base direction', 'learned direction', 'PSTH learning','PSTH dishabituation'\
                                          ,'FR learning','FR learning test','FR dishabituation','FR dishabituation test'\
                                          ,'FR dishabituation','PSTH mapping base','PSTH mapping learned','PSTH mapping null learned','FR mapping base','FR mapping learned','FR mapping null learned'])
                block_dict['cell_ID']=cell_ID
                block_dict['base direction']=block_base_dir
                block_dict['learned direction']=learned_direction
                block_dict['FR learning test']=FR_learning_test           
                block_dict['FR dishabituation test']=FR_dishabituation_test
                block_dict['FR mapping base']=FR_mapping_base
                block_dict['FR mapping learned']=FR_mapping_learned
                block_dict['FR mapping null learned']=FR_mapping_null_learned

                block_dict['PSTH mapping base']=psth_mapping_base
                block_dict['PSTH mapping learned']=psth_mapping_learned
                block_dict['PSTH mapping null learned']=psth_mapping_null_learned

                motor_dict_array.append(block_dict)


#%% omega analysis
# cell_list=[x['cell_ID'] for x in motor_dict_array]
# omega_analysis(cell_list,trial_type=trial_type_mapping)

#%% leanring vs dishabituation in mapping            
                                                                                                                                                                                                                                                                                                        
sig_motor_array=np.array([x for x in motor_dict_array ]  )

legend_size=8


mapping_base_psths=np.empty((len(sig_motor_array),len(timecourse)))
mapping_base_psths[:]=np.nan
mapping_learned_psths=np.empty((len(sig_motor_array),len(timecourse)))
mapping_learned_psths[:]=np.nan
mapping_null_learned_psths=np.empty((len(sig_motor_array),len(timecourse)))
mapping_null_learned_psths[:]=np.nan

mapping_base_FR=[]
mapping_learned_FR=[]
mapping_null_learned_FR=[]
n_cells=0
for block_inx,cur_block_dict in enumerate(sig_motor_array):
                
        mean_dishabituation=np.nanmean(cur_block_dict['FR dishabituation test'])
        mean_learning=np.nanmean(cur_block_dict['FR learning test'])

        if mean_dishabituation> mean_learning:
            mapping_base_array=cur_block_dict['PSTH mapping base']
            mapping_learned_array=cur_block_dict['PSTH mapping learned']
            mapping_null_learned_array=cur_block_dict['PSTH mapping null learned']

            mapping_base_FR.append(np.nanmean(cur_block_dict['FR mapping base']))
            mapping_learned_FR.append(np.nanmean(cur_block_dict['FR mapping learned']))
            mapping_null_learned_FR.append(np.nanmean(cur_block_dict['FR mapping null learned']))

        elif mean_dishabituation<= mean_learning:                
            mapping_base_array=-cur_block_dict['PSTH mapping base']
            mapping_learned_array=-cur_block_dict['PSTH mapping learned']
            mapping_null_learned_array=-cur_block_dict['PSTH mapping null learned']

            mapping_base_FR.append(-np.nanmean(cur_block_dict['FR mapping base']))
            mapping_learned_FR.append(-np.nanmean(cur_block_dict['FR mapping learned']))
            mapping_null_learned_FR.append(-np.nanmean(cur_block_dict['FR mapping null learned']))

            mean_dishabituation=-mean_dishabituation
            mean_learning=-mean_learning


        n_cells=n_cells+1
        mapping_base_psths[block_inx,:]=mapping_base_array
        mapping_learned_psths[block_inx,:]=mapping_learned_array
        mapping_null_learned_psths[block_inx,:]=mapping_null_learned_array


#check wether FR in 100-300 between mapping base and learning is significantly different         
stat,p_learned_base=stats.wilcoxon(mapping_learned_FR,mapping_null_learned_FR)
p_learned_base=round(p_learned_base,3)
   
plt.plot(timecourse,np.nanmean(mapping_base_psths,axis=0),color='tab:blue',linestyle='dotted')
plt.plot(timecourse,np.nanmean(mapping_learned_psths,axis=0),color='tab:blue',linestyle='dashed')
plt.plot(timecourse,np.nanmean(mapping_null_learned_psths,axis=0),color='tab:blue')

plt.axvline(x=100,color='black')
plt.axvline(x=300,color='black')
plt.legend(['mapping base','mapping learned','mapping null learned'], loc ="lower right",fontsize=legend_size)
plt.title('motor blocks '+ str(n_cells)+' cells ,p='+str(p_learned_base))
plt.xlabel('time from MO')
plt.ylabel('FR')
plt.show()

plt.scatter(mapping_null_learned_FR,mapping_learned_FR,color='tab:blue')
plt.axline((0,0),(1,1),color='black')
plt.title('motor blocks '+ str(n_cells)+' cells ,p='+str(p_learned_base))
plt.xlabel('FR mapping opposite learned')
plt.ylabel('FR mapping learned')
plt.show()

#%% leanring vs dishabituation in mapping            
                                                                                                                                                                                                                                                                                                        
sig_motor_array=np.array([x for x in motor_dict_array ]  )

legend_size=8


mapping_base_psths=np.empty((len(sig_motor_array),len(timecourse)))
mapping_base_psths[:]=np.nan
mapping_learned_psths=np.empty((len(sig_motor_array),len(timecourse)))
mapping_learned_psths[:]=np.nan
mapping_null_learned_psths=np.empty((len(sig_motor_array),len(timecourse)))
mapping_null_learned_psths[:]=np.nan

mapping_base_FR=[]
mapping_learned_FR=[]
mapping_null_learned_FR=[]
n_cells=0
for block_inx,cur_block_dict in enumerate(sig_motor_array):
                
        mean_dishabituation=np.nanmean(cur_block_dict['FR dishabituation test'])
        mean_learning=np.nanmean(cur_block_dict['FR learning test'])

        if mean_dishabituation> mean_learning:
            mapping_base_array=cur_block_dict['PSTH mapping base']
            mapping_learned_array=cur_block_dict['PSTH mapping learned']
            mapping_null_learned_array=cur_block_dict['PSTH mapping null learned']

            mapping_base_FR.append(np.nanmean(cur_block_dict['FR mapping base']))
            mapping_learned_FR.append(np.nanmean(cur_block_dict['FR mapping learned']))
            mapping_null_learned_FR.append(np.nanmean(cur_block_dict['FR mapping null learned']))


            n_cells=n_cells+1
        mapping_base_psths[block_inx,:]=mapping_base_array
        mapping_learned_psths[block_inx,:]=mapping_learned_array
        mapping_null_learned_psths[block_inx,:]=mapping_null_learned_array


#check wether FR in 100-300 between mapping base and learning is significantly different         
stat,p_learned_base=stats.wilcoxon(mapping_learned_FR,mapping_null_learned_FR)
p_learned_base=round(p_learned_base,3)
   
plt.plot(timecourse,np.nanmean(mapping_base_psths,axis=0),color='tab:blue',linestyle='dotted')
plt.plot(timecourse,np.nanmean(mapping_learned_psths,axis=0),color='tab:blue',linestyle='dashed')
plt.plot(timecourse,np.nanmean(mapping_null_learned_psths,axis=0),color='tab:blue')

plt.axvline(x=100,color='black')
plt.axvline(x=300,color='black')
plt.legend(['mapping base','mapping learned','mapping null learned'], loc ="lower right",fontsize=legend_size)
plt.title('motor blocks '+ str(n_cells)+' cells ,p='+str(p_learned_base))
plt.xlabel('time from MO')
plt.ylabel('FR')
plt.show()

plt.scatter(mapping_null_learned_FR,mapping_learned_FR,color='tab:blue')
plt.axline((0,0),(1,1),color='black')
plt.title('motor blocks '+ str(n_cells)+' cells ,p='+str(p_learned_base))
plt.xlabel('FR mapping opposite learned')
plt.ylabel('FR mapping learned')
plt.show()