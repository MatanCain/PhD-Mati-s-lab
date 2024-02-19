# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 14:48:12 2022

@author: Owner
"""
################################
#For all this program:

###############################

from glob import glob
import pickle
import os
from os.path import isfile,join
path="C:/Users/Owner/Documents/Matan/Mati's Lab/FEF learning project/Data_Analyses/Code analyzes/basic classes"
os.chdir(path) 
from behavior_class import session_task
import pandas as pd
import numpy as np
import re
import math
import matplotlib
import matplotlib.pyplot as plt
from mat4py import loadmat
from neuron_class import *
import warnings
from scipy.signal import savgol_filter
import statsmodels.api as sm
import copy
from cancel_pupil_bias import *
import scipy
warnings.filterwarnings("ignore")

behaviour_py_folder="C:/Users/Owner/Documents/Matan/Mati's Lab/FEF learning project/Data_Analyses/behaviour_python_two_monkeys/"

behaviour_db_excel="C:/Users/Owner/Documents/Matan/Mati's Lab/FEF learning project/Data_Analyses/fiona_behaviour_db.xlsx"
behaviour_db=pd.read_excel(behaviour_db_excel)
    
########################################################
def load_behaviour_session(behaviour_py_folder,task,session):
    filename=behaviour_py_folder+task+'/'+str(session)
    infile = open(filename,'rb')
    cur_session = pickle.load(infile)
    infile.close()
    return cur_session
########################################################
########################################################
#This function receives a list of tasks (e.g: ['8dir_active_passive_interleaved','8dir_active_passive_interleaved_100_25']
# For all the behavioral session recorded during those tasks the function calculates the pupil biased model for vPos
    # it returns a dictionnary where keys are sessions (strings) and values a list [slope, intercept] for each session
def get_pupil_dict(Tasks): 
    #Tasks=['8dir_active_passive_interleaved','8dir_active_passive_interleaved_100_25']
    pupil_dict={}
    for task in Tasks:
        session_list=os.listdir(os.path.join(behaviour_py_folder,task))
        for session in session_list:
            if 'ya' in session: #yasmin has no pupil problems
                continue
            else: #for fiona or other monkeys:
                [slope,intercept]=get_pupil_correction_model(session,task,behaviour_py_folder,0)
                pupil_dict[session]=[slope,intercept]
    return pupil_dict     
########################################################
#%% Create a data frame for the behaviour
#Each row in the data frame is the information regarding the behaviour of a single block
#create an empty data frame:
df_keys=['monkey','session','task','trial_type','base_direction','learned_direction','base velocity','learned velocity']    
behaviour_df=pd.DataFrame(columns=df_keys,dtype=float)
    

congruent_tasks=['fixation_right_probes_CW_100_25_cue','fixation_right_probes_CCW_100_25_cue','fixation_right_probes_CW','fixation_right_probes_CCW']
incongruent_tasks=['fixation_wrong_probes_CW_100_25_cue','fixation_wrong_probes_CCW_100_25_cue','fixation_wrong_probes_CW','fixation_wrong_probes_CCW']
motor_tasks=['Motor_learning_CW_100_25_cue','Motor_learning_CCW_100_25_cue','Motor_learning_CW','Motor_learning_CCW']
dishabituation_tasks=['Dishabituation_100_25_cue','Dishabituation']
washout_tasks=['washout_100_25_cue']
task_lists_general=[motor_tasks,congruent_tasks,incongruent_tasks,dishabituation_tasks,washout_tasks]
#task_lists_general=[congruent_tasks,incongruent_tasks]

#%% get a dictionnary with moidel for pupil correction for fiona
pupil_dict_list_general=[]
for task_list in task_lists_general[:-1]: 
    pupil_dict=get_pupil_dict(task_list)
    pupil_dict_list_general.append(pupil_dict)

#%%    
#saccade parameters to remove trials with early saccades
remove_early_saccades=1 #1 to remove trials with an early saccade
saccade_motion_begin=0 #after motion onset
saccade_motion_end=300 #after motion onset

pre_MO=200 #time of trials shown before motion onset
post_MO=800 #time of trials shown after motion onset
timecourse=np.arange(-pre_MO,post_MO)

kernel_size=30#for smoothing velocity
for task_list_inx,task_list in enumerate(task_lists_general):
    if task_list!=washout_tasks:
        pupil_dict=pupil_dict_list_general[task_list_inx]
    for cur_task in task_list:
        session_list=os.listdir(os.path.join(behaviour_py_folder,cur_task))
        for cur_session in session_list:
            sessions_db=behaviour_db.loc[:]['behaviour_session'] #sessions recorded in db
            if cur_session not in sessions_db.to_list(): #remove sessions not recroded in db
                continue
            cur_monkey=cur_session[0:2]
        #Load a behaviour session:
            for trial_type in['v20NS']:
        
                cur_behavior_df=load_behaviour_session(behaviour_py_folder,cur_task,cur_session)
                trials_df=cur_behavior_df.trials_df
                #separate into blocks
                # Separate the trials_df to different blocks (each session includes all the dishabitaution blocks recorded during a given day)
                trials_list=trials_df.loc[:]['filename'].tolist()
                trial_number_np=np.array([int(x.split('.',1)[1]) for x in trials_list ])
                block_end_indexes=np.where(np.diff(trial_number_np)>=80)[0]#find all trials where next trials is at least with a step of 80 trials (end of blocks)
                block_begin_indexes=np.append(np.array(0),block_end_indexes+1)#(beginning of blocks- just add 1 to end of blocks and also the first trial
                block_end_indexes=np.append(block_end_indexes,len(trials_df)-1)#add the last trial to end of blocks
                
                #for each block:
                for block_inx,(begin_inx,end_inx) in enumerate(zip(block_begin_indexes,block_end_indexes)):
                    cur_block_df=trials_df.iloc[np.arange(begin_inx,end_inx+1)]
                    
                    #Find current row in data base
                    db_inxs=behaviour_db.index[(behaviour_db['behaviour_session']==cur_session) & (behaviour_db['Task']==cur_task)].tolist() #find the indexes of the relevant rows in the db (row with same session and task)
                    cur_row_db=db_inxs[block_inx] #Find the current block
                    
                    if cur_task=='washout_100_25_cue':
                        cur_block_df=cur_block_df.loc[cur_block_df['trialname']=='d0'+trial_type,:] #keep only trials in the base direction
                    #Remove fail trials
                    cur_block_df=cur_block_df.loc[cur_block_df['fail']==0]
                    #Keep trials for specific trial type
                    cur_block_df=cur_block_df.loc[cur_block_df.trialname.str.contains(trial_type)]
                    if len(cur_block_df)==0: #if there is no trial left we skip the block
                        continue
                    
                    #Remove trials with blinks and saccade (optionnal)
                    cur_block_df.loc[:,'blink_onset'] = cur_block_df.apply(lambda row: [blink[0] if type(blink)==list else row.blinks[0]  for blink in row.blinks],axis=1)
                    cur_block_df.loc[:,'blink_motion']= cur_block_df.apply(lambda row: any([blink_onset>row.motion_onset+saccade_motion_begin and blink_onset<row.motion_onset+saccade_motion_end   for blink_onset in row.blink_onset]) ,axis=1)
                    cur_block_df.loc[:,'blink_motion']= cur_block_df.apply(lambda row: not(row.blink_motion),axis=1)
                    if remove_early_saccades:
                        #remove trial with saccades that begin around MO
                        cur_block_df.loc[:,'saccade_onset'] = cur_block_df.apply(lambda row: [saccade[0] if type(saccade)==list else row.saccades[0]  for saccade in row.saccades],axis=1)
                        cur_block_df.loc[:,'saccade_motion']= cur_block_df.apply(lambda row: any([saccade_onset>row.motion_onset+saccade_motion_begin and saccade_onset<row.motion_onset+saccade_motion_end for saccade_onset in row.saccade_onset]) ,axis=1)
                        cur_block_df.loc[:,'saccade_motion']= cur_block_df.apply(lambda row: not(row.saccade_motion),axis=1)
                        inxs_to_keep=cur_block_df[(cur_block_df.loc[:,'saccade_motion']==1) & (cur_block_df.loc[:,'blink_motion']==1)].index
                    else:
                        inxs_to_keep=cur_block_df[(cur_block_df.loc[:,'saccade_motion']==1)].index
                    cur_block_df=cur_block_df.loc[inxs_to_keep,:]

                    if len(cur_block_df)==0: #if there is no trial left we skip the block
                        continue
                    
                    #extract eye position (hor and vel)from columns relevant 
                    hPos_serie=cur_block_df.loc[:]['hPos']
                    vPos_serie=cur_block_df.loc[:]['vPos']
                    pupil_serie=cur_block_df.loc[:]['pupil']
                    MO_serie=cur_block_df.loc[:]['motion_onset']
                
                    #extract motion onset:
                    motion_onset_list=MO_serie.to_list()
                    
                    #convert hPos to rectangular np array and slice such tat motion onset is at index 0
                    hPos_list=hPos_serie.tolist()
                    hPos_list2=[cur_trial[int(motion_onset_list[cur_trial_inx])-pre_MO:int(motion_onset_list[cur_trial_inx])+post_MO] for cur_trial_inx,cur_trial in enumerate(hPos_list)]
                    hPos_np=pd.DataFrame(hPos_list2).to_numpy() 
                    hPos_block_mean_np=np.median(hPos_np,axis=0) #take median trial within the block
                    #same for pupil
                    pupil_list=pupil_serie.tolist()
                    pupil_list2=[cur_trial[int(motion_onset_list[cur_trial_inx])-pre_MO:int(motion_onset_list[cur_trial_inx])+post_MO] for cur_trial_inx,cur_trial in enumerate(pupil_list)]
                    pupil_np=pd.DataFrame(pupil_list2).to_numpy() 
                    #same for vpos
                    vPos_list=vPos_serie.tolist()
                    vPos_list2=[cur_trial[int(motion_onset_list[cur_trial_inx])-pre_MO:int(motion_onset_list[cur_trial_inx])+post_MO] for cur_trial_inx,cur_trial in enumerate(vPos_list)]
                    vPos_np=pd.DataFrame(vPos_list2).to_numpy() 
                    if  cur_monkey=='fi': #fix pupil bias for fiona
                        #cancel pupil bias
                        cur_slope=pupil_dict[cur_session][0]
                        cur_intercept=pupil_dict[cur_session][1]
                        vPos_np=vPos_np-(cur_slope*pupil_np+cur_intercept)
                    vPos_block_mean_np=np.median(vPos_np,axis=0) #take median trial within the block
   
                    #Base and learned directions
                    base_dir=cur_block_df['screen_rotation'].iloc[0]
                    if task_list in [motor_tasks,congruent_tasks,incongruent_tasks]: 
                        if 'CCW' in cur_block_df.iloc[0]['task']:
                            learned_dir=(base_dir+90)%360
                            learned_dir_df='CCW'
                        elif 'CW' in cur_block_df.iloc[0]['task']:   
                            learned_dir=(base_dir-90)%360
                            learned_dir_df='CW'
                        else:
                            learned_dir_df=np.nan
                    
                    #If current monkey is fiona we use dishabituatiuon as baseline. If its yasmin we use washout
                    elif cur_task in dishabituation_tasks:
                        if cur_monkey=='ya':
                            continue
                            
                        elif cur_monkey=='fi':
                            behaviour_db_sess2= behaviour_db.loc[cur_row_db+1,:]
                            next_task=behaviour_db.loc[cur_row_db+1,'Task']
                            if len(behaviour_db_sess2.index)==0 or next_task not in congruent_tasks+incongruent_tasks+motor_tasks:
                                continue
                            else:
                                if 'CCW' in next_task[0]:
                                    learned_dir=(base_dir+90)%360
                                    learned_dir_df='CCW'
                                elif 'CW' in next_task[0]:   
                                    learned_dir=(base_dir-90)%360
                                    learned_dir_df='CW'

                    elif cur_task in washout_tasks:
                        behaviour_db_sess2= behaviour_db.loc[cur_row_db+1,:]
                        next_task=behaviour_db.loc[cur_row_db+1,'Task']
                        if len(behaviour_db_sess2.index)==0 or next_task not in congruent_tasks+incongruent_tasks+motor_tasks:
                            continue
                        else:
                            if 'CCW' in next_task:
                                learned_dir=(base_dir+90)%360
                                learned_dir_df='CCW'
                            elif 'CW' in next_task:   
                                learned_dir=(base_dir-90)%360
                                learned_dir_df='CW'            
                        
                    #create rotation matrix
                    base_dir_rad=-math.radians(base_dir)
                    rotation_matrix=np.array([[math.cos(base_dir_rad),-math.sin(base_dir_rad)],[math.sin(base_dir_rad),math.cos(base_dir_rad)]])
                
                    pos_block_rotated=np.matmul(rotation_matrix,np.vstack([hPos_block_mean_np,vPos_block_mean_np]))
                    vel_block_rotated=np.diff(pos_block_rotated)*1000 #Average velocity in base direction for current block
                    vel_block_rotated = np.hstack((vel_block_rotated, np.tile(vel_block_rotated[:, [-1]], 1)))#duplicate last column to compensate the lose column due to diff
                    vel_block_rotated=scipy.ndimage.gaussian_filter1d(vel_block_rotated, kernel_size,axis=1)    
                    
                    pos_base_rotated=pos_block_rotated[0,:]
                    pos_learned_rotated=pos_block_rotated[1,:]
                
                    vel_base_rotated=vel_block_rotated[0,:]
                    vel_learned_rotated=vel_block_rotated[1,:]
                    
                    if learned_dir_df=='CW':
                        pos_learned_rotated=-pos_learned_rotated
                        vel_learned_rotated=-vel_learned_rotated
                    
                    #plot current block
                    # plt.plot(timecourse,pos_base_rotated)
                    # plt.plot(timecourse,pos_learned_rotated)
                    # plt.title(cur_task+' '+str(base_dir)+' '+learned_dir_df)
                    # plt.show()

                    behaviour_df.loc[len(behaviour_df.index)] = [cur_monkey,cur_session, cur_task, trial_type,base_dir,learned_dir_df,vel_base_rotated,vel_learned_rotated] 

behaviour_df2=behaviour_df.loc[:][['monkey','session','task','trial_type','base_direction','learned_direction']] #create a copy of behaviour_df without the velocity columns. It is easier to manipulate for the memory of the computer

save_path="C:/Users/Owner/Documents/Matan/Mati's Lab/FEF learning project/Data_Analyses/Code analyzes/passive learning/"
filename=save_path+'behaviour_df_passive_learning'
outfile = open(filename,'wb')
pickle.dump(behaviour_df,outfile)
outfile.close()

#%% Plot different learning blocks as functrion of direction

#Choose a monkey
cur_monkey='ya'
fig, ax = plt.subplots(4) # one subplot for each direction and one for average across directions

time_window=[100,300]
timecourse_window=np.arange(time_window[0],time_window[1])
for dir_inx,cur_direction in enumerate([0,90,180,270]):
    cur_behaviour_df=behaviour_df.loc[:][behaviour_df['monkey']==cur_monkey]
    cur_behaviour_df=behaviour_df.loc[:][behaviour_df['base_direction']==cur_direction]
    
    
    motor_df=cur_behaviour_df.loc[:][(cur_behaviour_df['task']=='Motor_learning_CW_100_25_cue')|(cur_behaviour_df['task']=='Motor_learning_CCW_100_25_cue')]
    learned_vel=motor_df['learned velocity'].to_list()
    learned_vel=np.nanmean(np.array(learned_vel),0)
    learned_vel=learned_vel[pre_MO+time_window[0]:pre_MO+time_window[1]]
    ax[dir_inx].plot(timecourse_window,learned_vel)
    
    if dir_inx==3:
        ax[dir_inx].set_xlabel('time from MO')
        
    
    congruent_df=cur_behaviour_df.loc[:][(cur_behaviour_df['task']=='fixation_right_probes_CW_100_25_cue')|(cur_behaviour_df['task']=='fixation_right_probes_CCW_100_25_cue')]
    learned_vel=congruent_df['learned velocity'].to_list()
    learned_vel=np.nanmean(np.array(learned_vel),0)
    learned_vel=learned_vel[pre_MO+time_window[0]:pre_MO+time_window[1]]    
    ax[dir_inx].plot(timecourse_window,learned_vel)

    incongruent_df=cur_behaviour_df.loc[:][(cur_behaviour_df['task']=='fixation_wrong_probes_CW_100_25_cue')|(cur_behaviour_df['task']=='fixation_wrong_probes_CCW_100_25_cue')]
    learned_vel=incongruent_df['learned velocity'].to_list()
    learned_vel=np.nanmean(np.array(learned_vel),0)
    learned_vel=learned_vel[pre_MO+time_window[0]:pre_MO+time_window[1]]
    ax[dir_inx].plot(timecourse_window,learned_vel)
    #ax[dir_inx].set_title(cur_direction)    


    # if cur_monkey=='fi':
    #     washout_df=cur_behaviour_df.loc[:][cur_behaviour_df['task']=='Dishabituation_100_25_cue']
    # elif cur_monkey=='ya':    
    #     washout_df=cur_behaviour_df.loc[:][cur_behaviour_df['task']=='washout_100_25_cue']
    # learned_vel=washout_df['learned velocity'].to_list()
    # learned_vel=np.nanmean(np.array(learned_vel),0)
    # learned_vel=learned_vel[pre_MO+time_window[0]:pre_MO+time_window[1]]
    # ax[dir_inx].plot(timecourse_window,learned_vel)
    # ax[dir_inx].set_title(cur_direction)
    
    if dir_inx==0:
        ax[dir_inx].legend(['motor','congruent','incongruent'])
plt.suptitle('learned velocity')
fig.tight_layout()
plt.show()

#%% Plot different learning blocks as functrion of direction
#Choose a monkey
cur_monkey='ya'

time_window=[100,300]
timecourse_window=np.arange(time_window[0],time_window[1])
cur_behaviour_df=behaviour_df.loc[:][behaviour_df['monkey']==cur_monkey]


motor_df=cur_behaviour_df.loc[:][(cur_behaviour_df['task']=='Motor_learning_CW_100_25_cue')|(cur_behaviour_df['task']=='Motor_learning_CCW_100_25_cue')]
learned_vel=motor_df['learned velocity'].to_list()
learned_vel=np.array(learned_vel)
learned_vel=learned_vel[:,pre_MO+time_window[0]:pre_MO+time_window[1]]
learned_vel_sem=scipy.stats.sem(learned_vel, axis=0)
learned_vel=np.nanmean(learned_vel,0)
plt.errorbar(timecourse_window,learned_vel,learned_vel_sem)
plt.xlabel('time from MO')
plt.ylabel('learned vel')

congruent_df=cur_behaviour_df.loc[:][(cur_behaviour_df['task']=='fixation_right_probes_CW_100_25_cue')|(cur_behaviour_df['task']=='fixation_right_probes_CCW_100_25_cue')]
learned_vel=congruent_df['learned velocity'].to_list()
learned_vel=np.array(learned_vel)
learned_vel=learned_vel[:,pre_MO+time_window[0]:pre_MO+time_window[1]]
learned_vel_sem=scipy.stats.sem(learned_vel, axis=0)
learned_vel=np.nanmean(learned_vel,0)
plt.errorbar(timecourse_window,learned_vel,learned_vel_sem)
plt.xlabel('time from MO')
plt.ylabel('learned vel')

incongruent_df=cur_behaviour_df.loc[:][(cur_behaviour_df['task']=='fixation_wrong_probes_CW_100_25_cue')|(cur_behaviour_df['task']=='fixation_wrong_probes_CCW_100_25_cue')]
learned_vel=incongruent_df['learned velocity'].to_list()
learned_vel=np.array(learned_vel)
learned_vel=learned_vel[:,pre_MO+time_window[0]:pre_MO+time_window[1]]
learned_vel_sem=scipy.stats.sem(learned_vel, axis=0)
learned_vel=np.nanmean(learned_vel,0)
plt.errorbar(timecourse_window,learned_vel,learned_vel_sem)
plt.xlabel('time from MO')
plt.ylabel('learned vel')

# if cur_monkey=='fi':
#     washout_df=cur_behaviour_df.loc[:][cur_behaviour_df['task']=='Dishabituation_100_25_cue']
# elif cur_monkey=='ya':    
#     washout_df=cur_behaviour_df.loc[:][cur_behaviour_df['task']=='washout_100_25_cue']
# learned_vel=washout_df['learned velocity'].to_list()
# learned_vel=np.array(learned_vel)
# learned_vel=learned_vel[:,pre_MO+time_window[0]:pre_MO+time_window[1]]
# learned_vel_sem=scipy.stats.sem(learned_vel, axis=0)
# learned_vel=np.nanmean(learned_vel,0)
# plt.errorbar(timecourse_window,learned_vel,learned_vel_sem)
# plt.xlabel('time from MO')
# plt.ylabel('learned vel')

plt.legend(['motor','congruent','incongruent','washout'])
plt.show()

#%% Scatter plot congruent vs incongruent
cur_monkey='ya'
cur_behaviour_df=behaviour_df.loc[:][behaviour_df['monkey']==cur_monkey]

congruent_df=cur_behaviour_df.apply(lambda row: row[cur_behaviour_df['task'].isin(congruent_tasks)])
congruent_df2=congruent_df.loc[:][['monkey','session','task','trial_type','base_direction','learned_direction']] 

incongruent_df=cur_behaviour_df.apply(lambda row: row[cur_behaviour_df['task'].isin(incongruent_tasks)])
incongruent_df2=incongruent_df.loc[:][['monkey','session','task','trial_type','base_direction','learned_direction']] 

match_block_array=[]
time_window=[100,300]

# loop through the rows of congruent df and find the index of the maching block in incongruent df
for index, row in congruent_df.iterrows():
    cur_session=row['session']
    cur_base_direction=row['base_direction']
    cur_learned_direction=row['learned_direction']
    
    match_inx=incongruent_df.loc[(incongruent_df['session']==cur_session) & (incongruent_df['base_direction']==cur_base_direction)&(incongruent_df['learned_direction']==cur_learned_direction)].index
    match_block_array.append(match_inx)

cong_vs_incong=np.empty([len(match_block_array),2])
cong_vs_incong[:]=np.nan
for match_inx in np.arange(len(match_block_array)): 
    if len(match_block_array[match_inx])>0:#blocks with a match
        cur_match_inx=match_block_array[match_inx][0]
        cong_learned_vel=congruent_df.iloc[match_inx]['learned velocity'][pre_MO+time_window[0]:pre_MO+time_window[1]]
        cong_learned_vel_mean=np.mean(cong_learned_vel)
        cong_vs_incong[match_inx,0]=cong_learned_vel_mean
        incong_learned_vel=incongruent_df.loc[cur_match_inx,'learned velocity'][pre_MO+time_window[0]:pre_MO+time_window[1]]
        incong_learned_vel_mean=np.mean(incong_learned_vel)
        cong_vs_incong[match_inx,1]=incong_learned_vel_mean
    else:
        continue

#rmove nan rows. (i.e rows with no match)
cong_vs_incong2=cong_vs_incong[~np.isnan(cong_vs_incong).all(axis=1)]

(stat,pvalue)=scipy.stats.wilcoxon(cong_vs_incong2[:,0],cong_vs_incong2[:,1],alternative='greater')
#build a scatter plot
plt.scatter(cong_vs_incong2[:,0],cong_vs_incong2[:,1])
plt.axline((0, 0), (1, 1), linewidth=2, color='r')
plt.xlabel('congruent')
plt.ylabel('incongruent')
plt.title('learned velocity p='+str(round(pvalue,3)))
plt.show()
